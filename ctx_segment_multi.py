"""
多尺度 CTX 语义分割：三路模型（128/256/1024px patch）推理结果融合。

三个模型的类别空间互不重叠（除 background），融合后统一到一个全局标签空间。
以 128px 网格为基准分辨率，将 256px 和 1024px 的预测上采样后按加权置信度仲裁。

用法：
    python ctx_segment_multi.py
        --ctx-input  <.tif 或 .zip>
        --ckpt-128   <128px 模型 .ckpt>  --mapping-128   <128px mapping.json>
        --ckpt-256   <256px 模型 .ckpt>  --mapping-256   <256px mapping.json>
        --ckpt-1024  <1024px 模型 .ckpt> --mapping-1024  <1024px mapping.json>
        --output-png <输出 PNG 路径>
        [--batch-size 64] [--num-workers 4] [--device auto]
        [--alpha 0.5] [--vis-downsample 8]
        [--thresh-128 0.40] [--thresh-256 0.55] [--thresh-1024 0.65]
        [--weight-128 1.0]  [--weight-256 0.8]  [--weight-1024 0.6]
        [--annotation-dir PATH]

CTX 输入支持：
    - 直接传 .tif 文件路径
    - 传 .zip 文件路径（从中自动找 .tif）
"""

import argparse
import functools
import io
import json
import math
import os
import re
import sys
import zipfile

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.enums
import rasterio.windows
import shapefile
import torch
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from omegaconf import OmegaConf
from pyproj import CRS, Transformer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── helpers (copied from ctx_segment.py) ────────────────────────────────────

def get_vsizip_path(ctx_input: str) -> str:
    if ctx_input.lower().endswith(".zip"):
        with zipfile.ZipFile(ctx_input) as zf:
            tif_name = next(n for n in zf.namelist() if n.lower().endswith((".tif", ".tiff")))
        abs_zip = os.path.abspath(ctx_input).replace("\\", "/")
        return f"/vsizip/{abs_zip}/{tif_name}"
    return ctx_input


def load_model_from_checkpoint(ckpt_path: str, device: torch.device):
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
    print(f"  模型加载: {model_name}  ({cfg.data.num_classes} 类)")
    return model, cfg


def build_transform(cfg):
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
    def __init__(self, ctx_path: str, n_rows: int, n_cols: int,
                 patch_size: int, transform):
        self.ctx_path   = ctx_path
        self.n_rows     = n_rows
        self.n_cols     = n_cols
        self.patch_size = patch_size
        self.transform  = transform
        self._ds        = None

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


def decode_name(raw: str) -> str:
    try:
        return raw.encode("cp437").decode("gbk")
    except Exception:
        return raw


def parse_task_name(filename: str) -> str:
    name = os.path.splitext(filename)[0]
    return re.sub(r'_\d{8}_\d+\s*$', '', name)


def get_tile_lonlat_bounds(ctx_path: str):
    with rasterio.open(ctx_path) as ds:
        crs_wkt = ds.crs.to_wkt()
        b = ds.bounds
    proj_crs = CRS.from_wkt(crs_wkt)
    geo_crs  = proj_crs.geodetic_crs
    t = Transformer.from_crs(proj_crs, geo_crs, always_xy=True)
    lon_min, lat_min = t.transform(b.left,  b.bottom)
    lon_max, lat_max = t.transform(b.right, b.top)
    return lon_min, lat_min, lon_max, lat_max


def load_annotations_for_tile(annotation_dir: str, lon_min, lat_min, lon_max, lat_max,
                               name_to_id: dict):
    result = {}
    for fname in sorted(os.listdir(annotation_dir)):
        if not fname.endswith(".zip"):
            continue
        task_name = parse_task_name(fname)
        if task_name not in name_to_id:
            continue
        label_id = name_to_id[task_name]
        zip_path = os.path.join(annotation_dir, fname)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                names   = {decode_name(i.filename): i.filename for i in zf.infolist()}
                shp_key = next((k for k in names if k.endswith(".shp")), None)
                if not shp_key:
                    continue
                base = shp_key[:-4]

                def part(ext):
                    raw = names.get(base + ext)
                    return io.BytesIO(zf.read(raw)) if raw else None

                prj_raw = names.get(base + ".prj")
                prj_wkt = zf.read(prj_raw).decode("utf-8") if prj_raw else None
                sf = shapefile.Reader(shp=part(".shp"), shx=part(".shx"),
                                      dbf=part(".dbf"), encoding="utf-8")
                shapes = [
                    s for s in sf.iterShapes()
                    if lon_min <= (s.bbox[0] + s.bbox[2]) / 2 <= lon_max
                    and lat_min <= (s.bbox[1] + s.bbox[3]) / 2 <= lat_max
                ]
            if shapes:
                result[label_id] = (shapes, prj_wkt)
                print(f"  annotation [{label_id}] {task_name}: {len(shapes)} 个要素")
        except Exception as e:
            print(f"  跳过 {fname}: {e}")
    return result


def draw_annotation_panel(ax, base_rgb, annotations, colors, ctx_path, vis_downsample):
    ax.imshow(base_rgb, cmap="gray")
    with rasterio.open(ctx_path) as ds:
        ctx_crs_wkt = ds.crs.to_wkt()
        affine      = ds.transform
        scale       = 1.0 / vis_downsample

    for label_id, (shapes, prj_wkt) in annotations.items():
        src_crs = CRS.from_wkt(prj_wkt) if prj_wkt else CRS.from_epsg(4326)
        dst_crs = CRS.from_wkt(ctx_crs_wkt)
        trans   = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

        def lonlat_to_px(lon, lat):
            mx, my = trans.transform(lon, lat)
            col = (mx - affine.c) / affine.a * scale
            row = (my - affine.f) / affine.e * scale
            return col, row

        patches = []
        for shape in shapes:
            if not shape.points:
                continue
            parts_idx = list(shape.parts) + [len(shape.points)]
            for i in range(len(shape.parts)):
                ring = shape.points[parts_idx[i]:parts_idx[i + 1]]
                px_coords = [lonlat_to_px(p[0], p[1]) for p in ring]
                patches.append(MplPolygon(px_coords, closed=True))

        color = colors[label_id]
        pc = PatchCollection(patches, facecolor="none",
                             edgecolor=color, linewidths=0.6, alpha=0.9)
        ax.add_collection(pc)


def read_vis_base(ctx_path: str, vis_downsample: int) -> np.ndarray:
    with rasterio.open(ctx_path) as ds:
        out_h = ds.height // vis_downsample
        out_w = ds.width  // vis_downsample
        band = ds.read(1, out_shape=(1, out_h, out_w),
                       resampling=rasterio.enums.Resampling.average).squeeze()
    p2, p98 = np.percentile(band[band > 0], (2, 98)) if band.any() else (0, 255)
    return np.clip((band.astype(np.float32) - p2) / max(p98 - p2, 1) * 255, 0, 255).astype(np.uint8)


@functools.lru_cache(maxsize=1)
def _find_cjk_font():
    import warnings
    import logging
    from matplotlib import font_manager as fm

    cjk_candidates = [
        "WenQuanYi Micro Hei", "Noto Sans CJK SC", "Noto Sans CJK TC",
        "WenQuanYi Zen Hei", "AR PL UMing CN", "AR PL UKai CN",
        "Microsoft YaHei", "SimHei", "SimSun",
    ]
    _fm_logger = logging.getLogger("matplotlib.font_manager")
    _prev_level = _fm_logger.level
    _fm_logger.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
            default_font = fm.findfont(fm.FontProperties())
            for f in cjk_candidates:
                try:
                    found = fm.findfont(fm.FontProperties(family=f))
                    if found != default_font:
                        return f
                except Exception:
                    pass
    finally:
        _fm_logger.setLevel(_prev_level)
    return None


# ── multi-scale functions ────────────────────────────────────────────────────

@torch.no_grad()
def run_inference_with_conf(ctx_path: str, n_rows: int, n_cols: int, patch_size: int,
                            transform, model, device, batch_size: int, num_workers: int):
    """逐 patch 推理，返回 label_map、conf_map、entropy_map，shape 均为 [n_rows, n_cols]。
    entropy_map 为归一化熵（0=完全确定，1=完全均匀分散），用于熵过滤。
    """
    dataset = CTXPatchDataset(ctx_path, n_rows, n_cols, patch_size, transform)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    label_map   = np.zeros((n_rows, n_cols), dtype=np.int32)
    conf_map    = np.zeros((n_rows, n_cols), dtype=np.float32)
    entropy_map = np.zeros((n_rows, n_cols), dtype=np.float32)

    for tensors, rows, cols in tqdm(loader, desc=f"  推理 patch={patch_size}px"):
        logits = model(tensors.to(device))
        probs  = torch.softmax(logits, dim=1)
        preds  = probs.argmax(dim=1).cpu().numpy()
        confs  = probs.max(dim=1).values.cpu().numpy()

        # 归一化熵：H / log(N)，范围 [0, 1]
        n_cls      = probs.shape[1]
        raw_ent    = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
        norm_ent   = (raw_ent / math.log(n_cls)).cpu().numpy()

        for pred, conf, ent, r, c in zip(preds, confs, norm_ent, rows.numpy(), cols.numpy()):
            label_map[r, c]   = pred
            conf_map[r, c]    = conf
            entropy_map[r, c] = ent

    return label_map, conf_map, entropy_map


def build_global_mapping(mapping_128: dict, mapping_256: dict, mapping_1024: dict):
    """
    将三个模型各自的 {local_id: class_name} 合并为统一全局标签空间。
    background 统一映射到全局 ID 0，其余类按 128→256→1024 顺序依次编号。
    返回：global_mapping, id_map_128, id_map_256, id_map_1024
    """
    global_mapping = {0: "background"}
    next_id = 1
    id_maps = []

    for mapping in (mapping_128, mapping_256, mapping_1024):
        id_map = {}
        for local_id, name in sorted(mapping.items()):
            if name == "background":
                id_map[local_id] = 0
            else:
                global_mapping[next_id] = name
                id_map[local_id] = next_id
                next_id += 1
        id_maps.append(id_map)

    return global_mapping, id_maps[0], id_maps[1], id_maps[2]


def _make_lut(id_map: dict) -> np.ndarray:
    """将 {local_id: global_id} 字典转为 numpy 查找数组，缺失项默认映射到 0（background）。"""
    max_local = max(id_map.keys())
    lut = np.zeros(max_local + 1, dtype=np.int32)
    for local, glob in id_map.items():
        lut[local] = glob
    return lut


def _upsample(arr: np.ndarray, factor: int, target_R: int, target_C: int) -> np.ndarray:
    """最近邻上采样整数倍，然后裁剪或补零到目标尺寸。"""
    up = np.repeat(np.repeat(arr, factor, axis=0), factor, axis=1)
    if up.shape[0] >= target_R and up.shape[1] >= target_C:
        return up[:target_R, :target_C]
    # 边缘不足时补零（视为 background，置信度 0）
    out = np.zeros((target_R, target_C), dtype=arr.dtype)
    h = min(up.shape[0], target_R)
    w = min(up.shape[1], target_C)
    out[:h, :w] = up[:h, :w]
    return out


def fuse_label_maps(
    label_128,  conf_128,  entropy_128,
    label_256,  conf_256,  entropy_256,
    label_1024, conf_1024, entropy_1024,
    id_map_128, id_map_256, id_map_1024,
    thresh_128: float,        thresh_256: float,        thresh_1024: float,
    weight_128: float,        weight_256: float,        weight_1024: float,
    entropy_thresh_128: float, entropy_thresh_256: float, entropy_thresh_1024: float,
) -> np.ndarray:
    """
    以 128px 网格为基准，将三路预测融合为全局 label map。
    每个格子满足以下全部条件才作为非 background 候选：
      1. argmax 不是 background
      2. 置信度（max softmax）>= thresh
      3. 归一化熵 <= entropy_thresh（分布足够集中，模型真的确定）
    多路候选中取加权得分（conf × weight）最高者；无候选则输出 background（0）。
    """
    R, C = label_128.shape

    lut_128  = _make_lut(id_map_128)
    lut_256  = _make_lut(id_map_256)
    lut_1024 = _make_lut(id_map_1024)

    # 全局标签（128px 网格分辨率）
    glb_128 = lut_128[label_128]                                         # [R, C]

    # 256px → 128px 网格（factor=2）
    glb_256_up     = _upsample(lut_256[label_256], 2, R, C)
    conf_256_up    = _upsample(conf_256,            2, R, C)
    entropy_256_up = _upsample(entropy_256,         2, R, C)

    # 1024px → 128px 网格（factor=8）
    glb_1024_up     = _upsample(lut_1024[label_1024], 8, R, C)
    conf_1024_up    = _upsample(conf_1024,             8, R, C)
    entropy_1024_up = _upsample(entropy_1024,          8, R, C)

    # 加权得分：三个条件均满足才有正得分，否则置 0（视为 background）
    score_128  = np.where(
        (glb_128     != 0) & (conf_128      >= thresh_128)  & (entropy_128      <= entropy_thresh_128),
        conf_128      * weight_128,  0.0).astype(np.float32)
    score_256  = np.where(
        (glb_256_up  != 0) & (conf_256_up   >= thresh_256)  & (entropy_256_up   <= entropy_thresh_256),
        conf_256_up   * weight_256,  0.0).astype(np.float32)
    score_1024 = np.where(
        (glb_1024_up != 0) & (conf_1024_up  >= thresh_1024) & (entropy_1024_up  <= entropy_thresh_1024),
        conf_1024_up  * weight_1024, 0.0).astype(np.float32)

    scores = np.stack([score_128,  score_256,  score_1024],  axis=0)    # [3, R, C]
    labels = np.stack([glb_128,    glb_256_up, glb_1024_up], axis=0)    # [3, R, C]

    best_idx   = np.argmax(scores, axis=0)                               # [R, C]
    best_score = np.max(scores,   axis=0)                                # [R, C]
    best_label = np.take_along_axis(labels, best_idx[np.newaxis], axis=0).squeeze(0)

    # 三路均无有效候选时输出 background（0）
    return np.where(best_score > 0, best_label, 0).astype(np.int32)


# ── post-processing ──────────────────────────────────────────────────────────

def majority_filter(label_map: np.ndarray, kernel_size: int) -> np.ndarray:
    """众数滤波：对每个格子取邻域内出现最多的类别，消除盐椒噪点。
    实现方式：one-hot 展开 → 对每个类别做 box 滑窗求和 → argmax。
    纯向量化，不使用 Python 循环。
    """
    from scipy.ndimage import uniform_filter
    n_classes = int(label_map.max()) + 1
    scores = np.stack([
        uniform_filter((label_map == k).astype(np.float32), size=kernel_size)
        for k in range(n_classes)
    ], axis=0)                                  # [n_classes, R, C]
    return np.argmax(scores, axis=0).astype(np.int32)


def remove_small_components(label_map: np.ndarray, min_size: int) -> np.ndarray:
    """小区域去除：面积小于 min_size 格的连通域，替换为周围邻居中最常见的类别。
    不直接改成背景，避免在图上产生新的空洞。
    """
    from scipy.ndimage import label as nd_label, binary_dilation
    result = label_map.copy()
    for class_id in np.unique(label_map):
        if class_id == 0:           # 背景本身不处理
            continue
        binary = (label_map == class_id)
        labeled, n_comp = nd_label(binary)
        for comp_id in range(1, n_comp + 1):
            mask = (labeled == comp_id)
            if mask.sum() >= min_size:
                continue
            # 向外膨胀 2 格，取邻居中最常见的类别
            dilated        = binary_dilation(mask, iterations=2)
            neighbor_vals  = result[dilated & ~mask]
            replace        = int(np.bincount(neighbor_vals.astype(np.intp)).argmax()) \
                             if neighbor_vals.size > 0 else 0
            result[mask]   = replace
    return result


def postprocess(label_map: np.ndarray, smooth_kernel: int, min_region: int) -> np.ndarray:
    """后处理入口：先众数滤波，再小区域去除。任一步设为 0 则跳过。"""
    result = label_map
    if smooth_kernel > 0:
        print(f"  众数滤波 kernel={smooth_kernel}×{smooth_kernel} ...")
        result = majority_filter(result, smooth_kernel)
    if min_region > 0:
        print(f"  小区域去除 min_size={min_region} 格 ...")
        result = remove_small_components(result, min_region)
    return result


def _make_colors(n_classes: int) -> np.ndarray:
    """为 n_classes 个类别生成颜色，循环使用 tab20 + tab20b（共 40 种）。"""
    cmap1 = plt.get_cmap("tab20")
    cmap2 = plt.get_cmap("tab20b")
    colors = []
    for i in range(n_classes):
        colors.append(cmap1(i % 20)[:3] if i < 20 else cmap2((i - 20) % 20)[:3])
    return np.array(colors)


def visualize(ctx_path, label_map, vis_downsample,
              mapping, output_path, alpha, annotations=None):
    _cjk = _find_cjk_font()
    plt.rcParams["font.sans-serif"] = [_cjk] if _cjk else ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    n_classes = max(mapping.keys()) + 1
    colors = _make_colors(n_classes)
    for label_id, name in mapping.items():
        if name == "background":
            colors[label_id] = [1.0, 1.0, 1.0]
            break

    base     = read_vis_base(ctx_path, vis_downsample)
    base_rgb = np.stack([base, base, base], axis=-1).astype(np.float32) / 255.0

    # label_map 在 128px 网格，可视化时每格对应 128/vis_downsample 个像素
    vis_patch = max(128 // vis_downsample, 1)
    seg_color = colors[label_map]
    seg_full  = np.repeat(np.repeat(seg_color, vis_patch, axis=0), vis_patch, axis=1)

    H = min(base_rgb.shape[0], seg_full.shape[0])
    W = min(base_rgb.shape[1], seg_full.shape[1])
    blended = np.clip((1 - alpha) * base_rgb[:H, :W] + alpha * seg_full[:H, :W], 0, 1)

    n_panels = 3 if annotations else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(10 * n_panels, 10))
    axes[0].imshow(base_rgb, cmap="gray"); axes[0].set_title("CTX 原图");        axes[0].axis("off")
    axes[1].imshow(blended);              axes[1].set_title("多尺度融合分割结果"); axes[1].axis("off")

    handles = [plt.Rectangle((0, 0), 1, 1, fc=colors[i]) for i in sorted(mapping.keys())]
    axes[1].legend(handles, [mapping[i] for i in sorted(mapping.keys())],
                   loc="lower right", fontsize=5, ncol=3, framealpha=0.8)

    if annotations:
        axes[2].set_title("标注 Ground Truth"); axes[2].axis("off")
        draw_annotation_panel(axes[2], base_rgb, annotations, colors, ctx_path, vis_downsample)
        ann_handles = [plt.Rectangle((0, 0), 1, 1, fc=colors[lid]) for lid in sorted(annotations)]
        axes[2].legend(ann_handles, [mapping.get(lid, str(lid)) for lid in sorted(annotations)],
                       loc="lower right", fontsize=5, ncol=2, framealpha=0.8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"可视化已保存: {output_path}")

    npy_path = os.path.splitext(output_path)[0] + "_labels.npy"
    np.save(npy_path, label_map)
    print(f"Label map 已保存: {npy_path}  shape={label_map.shape}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="多尺度 CTX 语义分割融合（128/256/1024px）")
    parser.add_argument("--ctx-input",    required=True, help=".tif 或 .zip 路径")
    parser.add_argument("--ckpt-128",     required=True, help="128px 模型 .ckpt 路径")
    parser.add_argument("--mapping-128",  required=True, help="128px 模型 mapping.json 路径")
    parser.add_argument("--ckpt-256",     required=True, help="256px 模型 .ckpt 路径")
    parser.add_argument("--mapping-256",  required=True, help="256px 模型 mapping.json 路径")
    parser.add_argument("--ckpt-1024",    required=True, help="1024px 模型 .ckpt 路径")
    parser.add_argument("--mapping-1024", required=True, help="1024px 模型 mapping.json 路径")
    parser.add_argument("--output-png",   required=True, help="输出 PNG 路径")

    parser.add_argument("--batch-size",     type=int,   default=64)
    parser.add_argument("--num-workers",    type=int,   default=4,
                        help="DataLoader 并行进程数（Windows 建议设 0）")
    parser.add_argument("--device",         default="auto")
    parser.add_argument("--alpha",          type=float, default=0.5)
    parser.add_argument("--vis-downsample", type=int,   default=8)

    # 置信度阈值（类别数越多阈值越低，见注释）
    parser.add_argument("--thresh-128",  type=float, default=0.40,
                        help="128px 模型置信度阈值（24 类，默认 0.40）")
    parser.add_argument("--thresh-256",  type=float, default=0.55,
                        help="256px 模型置信度阈值（10 类，默认 0.55）")
    parser.add_argument("--thresh-1024", type=float, default=0.65,
                        help="1024px 模型置信度阈值（6 类，默认 0.65）")

    # 尺度系数（小 patch 定位更精准，权重更高）
    parser.add_argument("--weight-128",  type=float, default=1.0,
                        help="128px 模型尺度系数（默认 1.0）")
    parser.add_argument("--weight-256",  type=float, default=0.8,
                        help="256px 模型尺度系数（默认 0.8）")
    parser.add_argument("--weight-1024", type=float, default=0.6,
                        help="1024px 模型尺度系数（默认 0.6）")

    # 熵过滤阈值：归一化熵超过此值视为"模型不确定"，强制输出 background
    # 归一化熵范围 [0, 1]：0=完全确定，1=完全均匀分散
    parser.add_argument("--entropy-thresh-128",  type=float, default=0.65,
                        help="128px 模型熵过滤阈值（默认 0.65）")
    parser.add_argument("--entropy-thresh-256",  type=float, default=0.65,
                        help="256px 模型熵过滤阈值（默认 0.65）")
    parser.add_argument("--entropy-thresh-1024", type=float, default=0.65,
                        help="1024px 模型熵过滤阈值（默认 0.65）")

    # 后处理平滑（默认均不执行）
    parser.add_argument("--smooth-kernel", type=int, default=0,
                        help="众数滤波窗口大小，0=不执行（建议值：5 或 7；值越大越平滑，但细长地物可能被抹掉）")
    parser.add_argument("--min-region",    type=int, default=0,
                        help="最小连通域面积（格子数），小于此值的孤立色块合并到周围类别，0=不执行（建议值：10~20）")

    parser.add_argument("--annotation-dir", default=None,
                        help="标注 zip 目录（可选，提供后在第三面板绘制 GT 标注）")
    args = parser.parse_args()

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    print(f"使用设备: {device}")

    ctx_path = get_vsizip_path(args.ctx_input)

    with rasterio.open(ctx_path) as ds:
        full_w, full_h = ds.width, ds.height
    print(f"CTX 尺寸: {full_w}×{full_h} px")

    # ── 三路推理 ─────────────────────────────────────────────────────────────
    scale_cfgs = [
        (128,  args.ckpt_128,  args.mapping_128,  args.thresh_128,  args.weight_128),
        (256,  args.ckpt_256,  args.mapping_256,  args.thresh_256,  args.weight_256),
        (1024, args.ckpt_1024, args.mapping_1024, args.thresh_1024, args.weight_1024),
    ]

    results  = {}   # patch_size -> (label_map, conf_map, entropy_map)
    mappings = {}   # patch_size -> {local_id: class_name}

    for patch_size, ckpt_path, mapping_path, _thresh, _weight in scale_cfgs:
        print(f"\n── {patch_size}px 模型 ──────────────────────────────────────")
        model, cfg = load_model_from_checkpoint(ckpt_path, device)
        transform  = build_transform(cfg)

        with open(mapping_path, encoding="utf-8") as f:
            mapping = {int(k): v for k, v in json.load(f).items()}
        mappings[patch_size] = mapping

        n_rows = full_h // patch_size
        n_cols = full_w  // patch_size
        print(f"  patch 网格: {n_rows}×{n_cols} = {n_rows * n_cols} 个")

        label_map, conf_map, entropy_map = run_inference_with_conf(
            ctx_path, n_rows, n_cols, patch_size,
            transform, model, device, args.batch_size, args.num_workers,
        )
        results[patch_size] = (label_map, conf_map, entropy_map)
        del model   # 及时释放显存再加载下一个模型

    # ── 构建全局标签空间 ──────────────────────────────────────────────────────
    print("\n── 构建全局标签空间 ──────────────────────────────────────────────")
    global_mapping, id_map_128, id_map_256, id_map_1024 = build_global_mapping(
        mappings[128], mappings[256], mappings[1024]
    )
    print(f"全局类别数: {len(global_mapping)}")
    for gid, name in sorted(global_mapping.items()):
        print(f"  {gid:3d}  {name}")

    # ── 融合 ──────────────────────────────────────────────────────────────────
    print("\n── 融合三路预测 ──────────────────────────────────────────────────")
    label_128,  conf_128,  entropy_128  = results[128]
    label_256,  conf_256,  entropy_256  = results[256]
    label_1024, conf_1024, entropy_1024 = results[1024]

    fused = fuse_label_maps(
        label_128,  conf_128,  entropy_128,
        label_256,  conf_256,  entropy_256,
        label_1024, conf_1024, entropy_1024,
        id_map_128, id_map_256, id_map_1024,
        args.thresh_128,  args.thresh_256,  args.thresh_1024,
        args.weight_128,  args.weight_256,  args.weight_1024,
        args.entropy_thresh_128, args.entropy_thresh_256, args.entropy_thresh_1024,
    )

    # 打印融合结果分布
    unique, counts = np.unique(fused, return_counts=True)
    print("融合结果分布:")
    for gid, cnt in zip(unique, counts):
        print(f"  [{gid:3d}] {global_mapping.get(gid, '?'):20s}  {cnt} 格")

    # ── 后处理平滑 ────────────────────────────────────────────────────────────
    if args.smooth_kernel > 0 or args.min_region > 0:
        print("\n── 后处理平滑 ────────────────────────────────────────────────")
        fused = postprocess(fused, args.smooth_kernel, args.min_region)
        unique, counts = np.unique(fused, return_counts=True)
        print("平滑后结果分布:")
        for gid, cnt in zip(unique, counts):
            print(f"  [{gid:3d}] {global_mapping.get(gid, '?'):20s}  {cnt} 格")

    # ── 可选标注 ──────────────────────────────────────────────────────────────
    annotations = None
    if args.annotation_dir:
        print("\n── 加载标注数据 ──────────────────────────────────────────────")
        lon_min, lat_min, lon_max, lat_max = get_tile_lonlat_bounds(ctx_path)
        name_to_id = {v: k for k, v in global_mapping.items()}
        annotations = load_annotations_for_tile(
            args.annotation_dir, lon_min, lat_min, lon_max, lat_max, name_to_id
        )
        print(f"共找到 {len(annotations)} 类标注")

    # ── 可视化 ────────────────────────────────────────────────────────────────
    print("\n── 生成可视化 ────────────────────────────────────────────────────")
    visualize(ctx_path, fused, args.vis_downsample,
              global_mapping, args.output_png, args.alpha, annotations=annotations)


if __name__ == "__main__":
    main()
