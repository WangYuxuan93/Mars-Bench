"""
用训练好的分类模型对 CTX tile 做逐 patch 推理，生成语义分割图。
与数据集构建方式一致：全分辨率打开 CTX，逐窗口按需读取每个 patch，不整体加载。

用法：
    python ctx_segment.py <ctx_input> <checkpoint> <mapping_json> <output_png>
        [--patch-size 136]    patch 像素大小（与数据集构建时一致，默认 136）
        [--batch-size 64]     推理 batch 大小（默认 64）
        [--device auto]       推理设备（默认自动选择）
        [--alpha 0.5]         分割图叠加透明度（默认 0.5）
        [--vis-downsample 8]  可视化底图的降采样倍数（不影响推理精度，默认 8）

CTX 输入支持：
    - 直接传 .tif 文件路径
    - 传 .zip 文件路径（从中自动找 .tif）
"""

import argparse
import json
import os
import sys
import zipfile

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.enums
import rasterio.windows
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def get_vsizip_path(ctx_input: str) -> str:
    """如果是 zip，返回 /vsizip/... 路径；否则直接返回。"""
    if ctx_input.lower().endswith(".zip"):
        with zipfile.ZipFile(ctx_input) as zf:
            tif_name = next(n for n in zf.namelist() if n.lower().endswith((".tif", ".tiff")))
        abs_zip = os.path.abspath(ctx_input).replace("\\", "/")
        return f"/vsizip/{abs_zip}/{tif_name}"
    return ctx_input


def load_model_from_checkpoint(ckpt_path: str, device: torch.device):
    """从 checkpoint 自动恢复 cfg 并加载模型。"""
    # PL 内部也会调用 torch.load，打补丁强制 weights_only=False
    import functools
    _orig = torch.load
    @functools.wraps(_orig)
    def _patched(*args, **kwargs):
        kwargs["weights_only"] = False
        return _orig(*args, **kwargs)
    torch.load = _patched

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt.get("hyper_parameters", {}))

    from marsbench.models import MODEL_REGISTRY
    model_name = cfg.model.name
    if model_name not in MODEL_REGISTRY["classification"]:
        raise ValueError(f"未知模型: {model_name}，可选: {list(MODEL_REGISTRY['classification'].keys())}")

    model_cls = MODEL_REGISTRY["classification"][model_name]
    model = model_cls.load_from_checkpoint(ckpt_path, cfg=cfg, map_location=device)
    model.to(device).eval()
    print(f"模型加载完成: {model_name}  ({cfg.data.num_classes} 类)")
    return model, cfg


def build_transform(cfg):
    """构建与训练 val_transform 一致的推理预处理。"""
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


class CTXPatchDataset(Dataset):
    """每个样本为一个 patch 的 (row, col, tensor)，在 worker 进程里读取。"""

    def __init__(self, ctx_path: str, n_rows: int, n_cols: int,
                 patch_size: int, transform):
        self.ctx_path   = ctx_path
        self.n_rows     = n_rows
        self.n_cols     = n_cols
        self.patch_size = patch_size
        self.transform  = transform
        self._ds        = None   # 每个 worker 各自打开，避免多进程共享句柄

    def __len__(self):
        return self.n_rows * self.n_cols

    def _open(self):
        if self._ds is None:
            self._ds = rasterio.open(self.ctx_path)

    def __getitem__(self, idx):
        r, c = divmod(idx, self.n_cols)
        self._open()
        ds = self._ds
        col0, row0 = c * self.patch_size, r * self.patch_size
        col1 = min(col0 + self.patch_size, ds.width)
        row1 = min(row0 + self.patch_size, ds.height)
        window = rasterio.windows.Window(col0, row0, col1 - col0, row1 - row0)
        crop = ds.read(1, window=window)

        if crop.size == 0:
            crop = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)
        else:
            p2, p98 = np.percentile(crop, (2, 98))
            if p98 > p2:
                crop = np.clip((crop.astype(np.float32) - p2) / (p98 - p2) * 255,
                               0, 255).astype(np.uint8)
            else:
                crop = np.zeros_like(crop, dtype=np.uint8)
            if crop.shape != (self.patch_size, self.patch_size):
                padded = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)
                padded[:crop.shape[0], :crop.shape[1]] = crop
                crop = padded

        rgb = np.stack([crop, crop, crop], axis=-1)
        tensor = self.transform(image=rgb)["image"]
        return tensor, r, c


@torch.no_grad()
def run_inference(ctx_path: str, n_rows: int, n_cols: int, patch_size: int,
                  transform, model, device, batch_size: int, num_workers: int):
    """DataLoader 并行读取 patch，流水线推理，返回 [n_rows, n_cols] label map。"""
    dataset = CTXPatchDataset(ctx_path, n_rows, n_cols, patch_size, transform)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    label_map = np.zeros((n_rows, n_cols), dtype=np.int32)
    for tensors, rows, cols in tqdm(loader, desc="推理中"):
        preds = model(tensors.to(device)).argmax(dim=1).cpu().numpy()
        for pred, r, c in zip(preds, rows.numpy(), cols.numpy()):
            label_map[r, c] = pred

    return label_map


def read_vis_base(ctx_path: str, vis_downsample: int) -> np.ndarray:
    """读取降采样版底图，仅用于可视化，不影响推理。"""
    with rasterio.open(ctx_path) as ds:
        out_h = ds.height // vis_downsample
        out_w = ds.width  // vis_downsample
        band = ds.read(1, out_shape=(1, out_h, out_w),
                       resampling=rasterio.enums.Resampling.average).squeeze()
    p2, p98 = np.percentile(band[band > 0], (2, 98)) if band.any() else (0, 255)
    return np.clip((band.astype(np.float32) - p2) / max(p98 - p2, 1) * 255, 0, 255).astype(np.uint8)


def visualize(ctx_path, label_map, patch_size, vis_downsample,
              mapping, output_path, alpha):
    plt.rcParams["font.sans-serif"] = [
        "WenQuanYi Micro Hei", "Noto Sans CJK SC",
        "Microsoft YaHei", "SimHei", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    n_classes = max(mapping.keys()) + 1
    cmap   = plt.get_cmap("tab20", n_classes)
    colors = np.array([cmap(i)[:3] for i in range(n_classes)])

    # 底图（降采样，仅用于显示）
    base = read_vis_base(ctx_path, vis_downsample)
    base_rgb = np.stack([base, base, base], axis=-1).astype(np.float32) / 255.0

    # 将 label_map 上采样到底图分辨率
    vis_patch = patch_size // vis_downsample
    if vis_patch < 1:
        vis_patch = 1
    seg_color = colors[label_map]   # [n_rows, n_cols, 3]
    seg_full  = np.repeat(np.repeat(seg_color, vis_patch, axis=0), vis_patch, axis=1)

    # 裁到相同尺寸
    H = min(base_rgb.shape[0], seg_full.shape[0])
    W = min(base_rgb.shape[1], seg_full.shape[1])
    blended = np.clip(
        (1 - alpha) * base_rgb[:H, :W] + alpha * seg_full[:H, :W], 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].imshow(base_rgb, cmap="gray"); axes[0].set_title("CTX 原图");  axes[0].axis("off")
    axes[1].imshow(blended);              axes[1].set_title("语义分割结果"); axes[1].axis("off")

    handles = [plt.Rectangle((0, 0), 1, 1, fc=colors[i]) for i in sorted(mapping.keys())]
    axes[1].legend(handles, [mapping[i] for i in sorted(mapping.keys())],
                   loc="lower right", fontsize=6, ncol=2, framealpha=0.8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"可视化已保存: {output_path}")

    npy_path = os.path.splitext(output_path)[0] + "_labels.npy"
    np.save(npy_path, label_map)
    print(f"Label map 已保存: {npy_path}  shape={label_map.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ctx_input",    help=".tif 或 .zip 路径")
    parser.add_argument("checkpoint",   help=".ckpt 路径")
    parser.add_argument("mapping_json", help="mapping.json 路径")
    parser.add_argument("output_png",   help="输出 PNG 路径")
    parser.add_argument("--patch-size",     type=int,   default=136)
    parser.add_argument("--batch-size",     type=int,   default=64)
    parser.add_argument("--num-workers",    type=int,   default=4,
                        help="DataLoader 并行读取进程数（默认 4，Windows 建议设 0）")
    parser.add_argument("--device",         default="auto")
    parser.add_argument("--alpha",          type=float, default=0.5)
    parser.add_argument("--vis-downsample", type=int,   default=8,
                        help="可视化底图的降采样倍数（不影响推理，默认 8）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else torch.device(args.device)
    print(f"使用设备: {device}")

    ctx_path = get_vsizip_path(args.ctx_input)

    # 加载模型
    model, cfg = load_model_from_checkpoint(args.checkpoint, device)
    transform  = build_transform(cfg)

    with open(args.mapping_json, encoding="utf-8") as f:
        mapping = {int(k): v for k, v in json.load(f).items()}

    # 只读元数据，不把影像读入内存
    with rasterio.open(ctx_path) as ds:
        full_w, full_h = ds.width, ds.height
    n_rows = full_h // args.patch_size
    n_cols = full_w  // args.patch_size
    print(f"CTX 尺寸: {full_w}×{full_h} px  →  patch 网格: {n_rows}×{n_cols} = {n_rows*n_cols} 个")

    label_map = run_inference(
        ctx_path, n_rows, n_cols, args.patch_size,
        transform, model, device, args.batch_size, args.num_workers,
    )

    visualize(ctx_path, label_map, args.patch_size, args.vis_downsample,
              mapping, args.output_png, args.alpha)


if __name__ == "__main__":
    main()
