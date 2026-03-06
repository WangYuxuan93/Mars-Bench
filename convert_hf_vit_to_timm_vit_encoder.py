# convert_hf_vit_to_timm_encoder.py

import argparse
import os
import torch

from transformers import AutoModel
import timm


def convert(hf_dir: str, timm_name: str, out_path: str):

    # 1 Load HF model (works for ViT / ViTMAE)
    hf_model = AutoModel.from_pretrained(hf_dir)
    hf_sd = hf_model.state_dict()

    # detect prefix (MAE models have "vit.")
    if "vit.embeddings.cls_token" in hf_sd:
        prefix = "vit."
    else:
        prefix = ""

    print("Detected HF prefix:", prefix if prefix else "(none)")

    # 2 build timm backbone
    timm_model = timm.create_model(timm_name, pretrained=False)
    timm_sd = timm_model.state_dict()

    mapped = {}

    # ---- embeddings ----
    mapped["cls_token"] = hf_sd[f"{prefix}embeddings.cls_token"]
    mapped["pos_embed"] = hf_sd[f"{prefix}embeddings.position_embeddings"]

    mapped["patch_embed.proj.weight"] = hf_sd[
        f"{prefix}embeddings.patch_embeddings.projection.weight"
    ]
    mapped["patch_embed.proj.bias"] = hf_sd[
        f"{prefix}embeddings.patch_embeddings.projection.bias"
    ]

    # ---- encoder blocks ----
    num_layers = 12

    for i in range(num_layers):

        # norms
        mapped[f"blocks.{i}.norm1.weight"] = hf_sd[
            f"{prefix}encoder.layer.{i}.layernorm_before.weight"
        ]
        mapped[f"blocks.{i}.norm1.bias"] = hf_sd[
            f"{prefix}encoder.layer.{i}.layernorm_before.bias"
        ]

        mapped[f"blocks.{i}.norm2.weight"] = hf_sd[
            f"{prefix}encoder.layer.{i}.layernorm_after.weight"
        ]
        mapped[f"blocks.{i}.norm2.bias"] = hf_sd[
            f"{prefix}encoder.layer.{i}.layernorm_after.bias"
        ]

        # qkv
        qw = hf_sd[f"{prefix}encoder.layer.{i}.attention.attention.query.weight"]
        kw = hf_sd[f"{prefix}encoder.layer.{i}.attention.attention.key.weight"]
        vw = hf_sd[f"{prefix}encoder.layer.{i}.attention.attention.value.weight"]

        qb = hf_sd[f"{prefix}encoder.layer.{i}.attention.attention.query.bias"]
        kb = hf_sd[f"{prefix}encoder.layer.{i}.attention.attention.key.bias"]
        vb = hf_sd[f"{prefix}encoder.layer.{i}.attention.attention.value.bias"]

        mapped[f"blocks.{i}.attn.qkv.weight"] = torch.cat([qw, kw, vw], dim=0)
        mapped[f"blocks.{i}.attn.qkv.bias"] = torch.cat([qb, kb, vb], dim=0)

        # attention output
        mapped[f"blocks.{i}.attn.proj.weight"] = hf_sd[
            f"{prefix}encoder.layer.{i}.attention.output.dense.weight"
        ]
        mapped[f"blocks.{i}.attn.proj.bias"] = hf_sd[
            f"{prefix}encoder.layer.{i}.attention.output.dense.bias"
        ]

        # mlp
        mapped[f"blocks.{i}.mlp.fc1.weight"] = hf_sd[
            f"{prefix}encoder.layer.{i}.intermediate.dense.weight"
        ]
        mapped[f"blocks.{i}.mlp.fc1.bias"] = hf_sd[
            f"{prefix}encoder.layer.{i}.intermediate.dense.bias"
        ]

        mapped[f"blocks.{i}.mlp.fc2.weight"] = hf_sd[
            f"{prefix}encoder.layer.{i}.output.dense.weight"
        ]
        mapped[f"blocks.{i}.mlp.fc2.bias"] = hf_sd[
            f"{prefix}encoder.layer.{i}.output.dense.bias"
        ]

    # ---- final norm ----
    mapped["norm.weight"] = hf_sd[f"{prefix}layernorm.weight"]
    mapped["norm.bias"] = hf_sd[f"{prefix}layernorm.bias"]

    # 3 shape check
    missing = []
    shape_mismatch = []

    for k, v in mapped.items():

        if k not in timm_sd:
            missing.append(k)
            continue

        if timm_sd[k].shape != v.shape:
            shape_mismatch.append((k, timm_sd[k].shape, v.shape))

    if missing:
        raise RuntimeError(f"Mapped keys not found in timm model: {missing[:20]}")

    if shape_mismatch:
        msg = "\n".join([f"{k}: timm={a} hf={b}" for k, a, b in shape_mismatch[:30]])
        raise RuntimeError(f"Shape mismatch found:\n{msg}")

    # 4 load
    load_res = timm_model.load_state_dict(mapped, strict=False)

    print("Load done.")
    print("Missing keys (expected head.*):", load_res.missing_keys)
    print("Unexpected keys:", load_res.unexpected_keys)

    # 5 save
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(mapped, out_path)

    print("Saved timm-compatible encoder weights to:", out_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hf_dir",
        type=str,
        required=True,
        help="HF model name or local directory",
    )

    parser.add_argument(
        "--timm_name",
        type=str,
        default="vit_base_patch16_224.augreg_in21k",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="timm_encoder_from_hf.pth",
    )

    args = parser.parse_args()

    convert(args.hf_dir, args.timm_name, args.out)


if __name__ == "__main__":
    main()