"""
分类模型独立评价脚本，在已有指标基础上增加逐类误分类占比统计。

用法：
    python eval_cls.py <checkpoint> <mapping_json> <data_dir> <annot_csv>
        [--split test]         评价哪个 split（默认 test）
        [--batch-size 128]
        [--num-workers 4]
        [--device auto]
        [--output results.csv] 将逐类指标保存为 CSV（可选）
"""

import argparse
import json
import os
import sys

import functools
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ─────────────────────────── model loading ───────────────────────────

def load_model(ckpt_path: str, device: torch.device):
    # PyTorch 2.6+ weights_only 补丁
    _orig = torch.load
    @functools.wraps(_orig)
    def _patched(*args, **kwargs):
        kwargs["weights_only"] = False
        return _orig(*args, **kwargs)
    torch.load = _patched

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = OmegaConf.create(ckpt.get("hyper_parameters", {}))

    from marsbench.models import MODEL_REGISTRY
    model_name = cfg.model.name
    if model_name not in MODEL_REGISTRY["classification"]:
        raise ValueError(f"未知模型: {model_name}")

    model_cls = MODEL_REGISTRY["classification"][model_name]
    model = model_cls.load_from_checkpoint(ckpt_path, cfg=cfg, map_location=device)
    model.to(device).eval()
    print(f"模型: {model_name}  ({cfg.data.num_classes} 类)")
    return model, cfg


# ─────────────────────────── transform ───────────────────────────────

def build_transform(cfg):
    import albumentations as A
    h, w = (tuple(cfg.model.input_size)[1:] if cfg.model.get("input_size") else (224, 224))
    if cfg.data.get("image_type", "rgb").lower() == "rgb":
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        mean, std = [0.5], [0.5]
    return A.Compose([
        A.Resize(height=h, width=w),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        A.ToTensorV2(),
    ])


# ─────────────────────────── inference ───────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device):
    all_preds, all_labels = [], []
    for imgs, labels in tqdm(loader, desc="推理中"):
        logits = model(imgs.to(device))
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


# ─────────────────────────── reporting ───────────────────────────────

def print_misclassification(cm_norm, class_names, top_n=3):
    """
    cm_norm: 行归一化混淆矩阵，shape=[n, n]，cm_norm[i,j] = P(pred=j | true=i)
    对每个类打印：正确率 + 被误分到哪几类（占比最高的 top_n 个）
    """
    n = len(class_names)
    print("\n" + "=" * 70)
    print("逐类误分类占比（占该类总样本的百分比）")
    print("=" * 70)
    for i in range(n):
        correct_pct = cm_norm[i, i] * 100
        row = cm_norm[i].copy()
        row[i] = 0  # 去掉对角线，只看错误
        top_idx = np.argsort(row)[::-1][:top_n]
        top_str = "  |  ".join(
            f"{class_names[j]}: {row[j]*100:.1f}%"
            for j in top_idx if row[j] > 0
        )
        print(f"\n[{i:2d}] {class_names[i]}")
        print(f"     正确率: {correct_pct:.1f}%")
        if top_str:
            print(f"     主要误分 → {top_str}")
        else:
            print(f"     无误分（或该类样本不足）")


def save_csv(report_dict, cm_norm, class_names, output_path):
    import csv
    base_fields = ["class_id", "class_name", "precision", "recall", "f1-score", "support"]
    misclf_fields = [f"misclf_to_{name}" for name in class_names]
    fieldnames = base_fields + misclf_fields

    rows = []
    for i, name in enumerate(class_names):
        r = report_dict.get(name, {})
        row = {
            "class_id":   i,
            "class_name": name,
            "precision":  round(r.get("precision", 0), 4),
            "recall":     round(r.get("recall", 0), 4),
            "f1-score":   round(r.get("f1-score", 0), 4),
            "support":    int(r.get("support", 0)),
        }
        for j, jname in enumerate(class_names):
            row[f"misclf_to_{jname}"] = round(float(cm_norm[i, j]), 4)
        rows.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV 已保存: {output_path}")


# ─────────────────────────── main ────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint",   help=".ckpt 路径")
    parser.add_argument("mapping_json", help="mapping.json（label_id → class_name）")
    parser.add_argument("data_dir",     help="数据集根目录（含 images/）")
    parser.add_argument("annot_csv",    help="annotation.csv 路径")
    parser.add_argument("--split",       default="test")
    parser.add_argument("--batch-size",  type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device",      default="auto")
    parser.add_argument("--output",      default=None, help="输出 CSV 路径（可选）")
    args = parser.parse_args()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else torch.device(args.device)
    )
    print(f"设备: {device}")

    # 加载 mapping
    with open(args.mapping_json, encoding="utf-8") as f:
        mapping = {int(k): v for k, v in json.load(f).items()}
    class_names = [mapping[i] for i in sorted(mapping.keys())]
    num_classes  = len(class_names)

    # 加载模型
    model, cfg = load_model(args.checkpoint, device)

    # 覆盖 cfg 里的数据路径为命令行传入的路径
    cfg.data.data_dir  = args.data_dir
    cfg.data.annot_csv = args.annot_csv

    # 构建 transform 和 dataset
    transform = build_transform(cfg)

    from marsbench.data.classification.CTX_Geomorph_Classification import CTX_Geomorph_Classification
    dataset = CTX_Geomorph_Classification(
        cfg       = cfg,
        data_dir  = args.data_dir,
        transform = transform,
        annot_csv = args.annot_csv,
        split     = args.split,
    )
    print(f"数据集: {len(dataset)} 个样本  split={args.split}")

    loader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        pin_memory  = (device.type == "cuda"),
        shuffle     = False,
    )

    # 推理
    preds, labels = run_inference(model, loader, device)

    # ── 标准指标 ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("分类报告（sklearn）")
    print("=" * 70)
    report_str  = classification_report(labels, preds, target_names=class_names, digits=4, zero_division=0)
    report_dict = classification_report(labels, preds, target_names=class_names, digits=4,
                                        zero_division=0, output_dict=True)
    print(report_str)

    # ── 混淆矩阵 & 误分类统计 ─────────────────────────────────────────
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    row_sum = cm.sum(axis=1, keepdims=True).astype(float)
    row_sum[row_sum == 0] = 1  # 避免除 0
    cm_norm = cm / row_sum    # 行归一化

    print_misclassification(cm_norm, class_names, top_n=3)

    # ── 可选 CSV 输出 ─────────────────────────────────────────────────
    if args.output:
        save_csv(report_dict, cm_norm, class_names, args.output)


if __name__ == "__main__":
    main()
