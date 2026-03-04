# convert_vitmae_to_timm_vit_encoder.py
import argparse
import os
import torch

from transformers import ViTMAEForPreTraining
import timm


def convert(hf_dir: str, timm_name: str, out_path: str):
    # 1) Load HF ViTMAE checkpoint (your folder with config.json + model.safetensors)
    hf_model = ViTMAEForPreTraining.from_pretrained(hf_dir)
    hf_sd = hf_model.state_dict()

    # 2) Build timm ViT backbone (no pretrained weights, we'll load ours)
    timm_model = timm.create_model(timm_name, pretrained=False)

    # timm target keys
    timm_sd = timm_model.state_dict()

    mapped = {}

    # ---- Embeddings ----
    mapped["cls_token"] = hf_sd["vit.embeddings.cls_token"]
    mapped["pos_embed"] = hf_sd["vit.embeddings.position_embeddings"]  # overwrite (you requested)
    mapped["patch_embed.proj.weight"] = hf_sd["vit.embeddings.patch_embeddings.projection.weight"]
    mapped["patch_embed.proj.bias"] = hf_sd["vit.embeddings.patch_embeddings.projection.bias"]

    # ---- Encoder blocks ----
    # timm: blocks.{i}.norm1/norm2, attn.qkv, attn.proj, mlp.fc1/fc2
    # hf:   vit.encoder.layer.{i}.layernorm_before/after, attention..., intermediate/output
    num_layers = 12
    for i in range(num_layers):
        # LayerNorms
        mapped[f"blocks.{i}.norm1.weight"] = hf_sd[f"vit.encoder.layer.{i}.layernorm_before.weight"]
        mapped[f"blocks.{i}.norm1.bias"] = hf_sd[f"vit.encoder.layer.{i}.layernorm_before.bias"]
        mapped[f"blocks.{i}.norm2.weight"] = hf_sd[f"vit.encoder.layer.{i}.layernorm_after.weight"]
        mapped[f"blocks.{i}.norm2.bias"] = hf_sd[f"vit.encoder.layer.{i}.layernorm_after.bias"]

        # Attention qkv (concat on dim=0)
        qw = hf_sd[f"vit.encoder.layer.{i}.attention.attention.query.weight"]
        kw = hf_sd[f"vit.encoder.layer.{i}.attention.attention.key.weight"]
        vw = hf_sd[f"vit.encoder.layer.{i}.attention.attention.value.weight"]
        qb = hf_sd[f"vit.encoder.layer.{i}.attention.attention.query.bias"]
        kb = hf_sd[f"vit.encoder.layer.{i}.attention.attention.key.bias"]
        vb = hf_sd[f"vit.encoder.layer.{i}.attention.attention.value.bias"]

        mapped[f"blocks.{i}.attn.qkv.weight"] = torch.cat([qw, kw, vw], dim=0)
        mapped[f"blocks.{i}.attn.qkv.bias"] = torch.cat([qb, kb, vb], dim=0)

        # Attention output projection
        mapped[f"blocks.{i}.attn.proj.weight"] = hf_sd[f"vit.encoder.layer.{i}.attention.output.dense.weight"]
        mapped[f"blocks.{i}.attn.proj.bias"] = hf_sd[f"vit.encoder.layer.{i}.attention.output.dense.bias"]

        # MLP
        mapped[f"blocks.{i}.mlp.fc1.weight"] = hf_sd[f"vit.encoder.layer.{i}.intermediate.dense.weight"]
        mapped[f"blocks.{i}.mlp.fc1.bias"] = hf_sd[f"vit.encoder.layer.{i}.intermediate.dense.bias"]
        mapped[f"blocks.{i}.mlp.fc2.weight"] = hf_sd[f"vit.encoder.layer.{i}.output.dense.weight"]
        mapped[f"blocks.{i}.mlp.fc2.bias"] = hf_sd[f"vit.encoder.layer.{i}.output.dense.bias"]

    # ---- Final norm ----
    mapped["norm.weight"] = hf_sd["vit.layernorm.weight"]
    mapped["norm.bias"] = hf_sd["vit.layernorm.bias"]

    # 3) Sanity checks: shapes must match timm expected tensors
    missing = []
    shape_mismatch = []
    for k, v in mapped.items():
        if k not in timm_sd:
            missing.append(k)
            continue
        if timm_sd[k].shape != v.shape:
            shape_mismatch.append((k, timm_sd[k].shape, v.shape))

    if missing:
        raise RuntimeError(f"Mapped keys not found in timm model state_dict (unexpected): {missing[:20]} ...")
    if shape_mismatch:
        msg = "\n".join([f"{k}: timm={a} hf={b}" for k, a, b in shape_mismatch[:30]])
        raise RuntimeError(f"Shape mismatch found:\n{msg}")

    # 4) Load into timm model (strict=False because we intentionally ignore head.*)
    load_res = timm_model.load_state_dict(mapped, strict=False)

    # If anything else is missing besides head, it should be investigated
    unexpected = load_res.unexpected_keys
    missing_keys = load_res.missing_keys

    # In a vanilla timm ViT, missing should usually be head.weight/head.bias only
    # But we keep it flexible and print the results.
    print("Load done.")
    print("Missing keys (expected head.*):", missing_keys)
    print("Unexpected keys:", unexpected)

    # 5) Save mapped encoder weights (timm-style keys)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(mapped, out_path)
    print(f"Saved timm-compatible encoder weights to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_dir", type=str, required=True, help="Folder with config.json + model.safetensors")
    parser.add_argument(
        "--timm_name",
        type=str,
        default="vit_base_patch16_224.augreg_in21k",
        help="timm model name (without 'tu-')",
    )
    parser.add_argument("--out", type=str, default="timm_encoder_from_vitmae.pth")
    args = parser.parse_args()
    convert(args.hf_dir, args.timm_name, args.out)


if __name__ == "__main__":
    main()