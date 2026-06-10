from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from timm.layers import apply_rot_embed_cat
from tqdm.auto import tqdm

from src.utils.explain import attention_rollout, grad_attention_rollout, to_patch_heatmap
from src.utils.io import ensure_dir, group_renders_by_specimen, list_image_files, load_ids
from src.utils.vision import build_transform, forward_embedding, load_dinov3_model, load_image_tensor, resolve_device

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Explain DINOv3 attention for embedding formation")
    p.add_argument("--renders", type=Path, required=True)
    p.add_argument("--emb", type=Path, required=True)
    p.add_argument("--ids", type=Path, required=True)
    p.add_argument(
        "--specimen_id",
        type=str,
        required=False,
        default=None,
        help="Target specimen ID. If omitted, all specimen IDs found in both renders and --ids are processed.",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", type=str, default="dinov3_vitb16")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--crop-size", type=int, default=224)
    p.add_argument("--layers", type=str, choices=("all", "last"), default="all")
    p.add_argument(
        "--num-show",
        type=int,
        default=6,
        help="Number of views to visualize. If larger than available views, all available views are shown.",
    )
    return p.parse_args()


def _specimen_output_dir(base_out: Path, specimen_id: str) -> Path:
    parts = [p for p in PurePosixPath(specimen_id).parts if p not in ("", ".", "..")]
    if not parts:
        raise ValueError(f"Invalid specimen_id for output path: {specimen_id}")
    return base_out.joinpath(*parts)


def _collect_blocks(model: torch.nn.Module) -> list[torch.nn.Module]:
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise RuntimeError("Model has no blocks attribute for attention rollout")
    out = [b for b in blocks if hasattr(b, "attn")]
    if not out:
        raise RuntimeError("Model blocks do not expose attention modules")
    return out


def _select_blocks_for_rollout(blocks: list[torch.nn.Module], layers: str) -> list[torch.nn.Module]:
    if layers == "last":
        return [blocks[-1]]
    return blocks


def _reset_block_attn_cache(blocks: list[torch.nn.Module]) -> None:
    for blk in blocks:
        blk.attn._last_attn_map = None
        blk.attn._last_attn_grad = None


def _eva_qkv(attn_obj: torch.nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """EvaAttentionと同じ手順で q, k, v を [B, heads, N, head_dim] で取り出す。

    DINOv3(EVA系)はfused qkv(bias無し)＋別個のq/k/v bufferを使う構成と、
    分離したq_proj/k_proj/v_projを使う構成の両方があり得るため両対応する。
    """
    bsz, n_tokens, _ = x.shape
    num_heads = int(getattr(attn_obj, "num_heads", 1))

    qkv_layer = getattr(attn_obj, "qkv", None)
    if qkv_layer is not None:
        q_bias = getattr(attn_obj, "q_bias", None)
        if q_bias is None:
            qkv = qkv_layer(x)
        else:
            # base版はbias無しだが、_qkvb版は q_bias/k_bias/v_bias を結合して適用する
            qkv_bias = torch.cat((attn_obj.q_bias, attn_obj.k_bias, attn_obj.v_bias))
            if getattr(attn_obj, "qkv_bias_separate", False):
                qkv = qkv_layer(x)
                qkv = qkv + qkv_bias
            else:
                qkv = F.linear(x, weight=qkv_layer.weight, bias=qkv_bias)
        qkv = qkv.reshape(bsz, n_tokens, 3, num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
    else:
        q = attn_obj.q_proj(x).reshape(bsz, n_tokens, num_heads, -1).transpose(1, 2)
        k = attn_obj.k_proj(x).reshape(bsz, n_tokens, num_heads, -1).transpose(1, 2)
        v = attn_obj.v_proj(x).reshape(bsz, n_tokens, num_heads, -1).transpose(1, 2)
    return q, k, v


def _make_attention_forward_wrapper(attn_obj: torch.nn.Module, original_forward: Any):
    """EvaAttention.forward を忠実に再現しつつ、softmax後のattention重みを捕捉する。

    DINOv3はRoPE(回転位置埋め込み)を使うため、qkvを手動再計算する際にRoPEを
    適用しないとattentionマップが実際のモデルと一致しない。blockから渡される
    ``rope`` テンソルを受け取り、prefixトークン(CLS+register)以外に適用する。
    """
    del original_forward

    def wrapped_forward(*args: Any, **kwargs: Any) -> torch.Tensor:
        if len(args) > 0:
            x = args[0]
        else:
            x = kwargs.get("x", None)
        if x is None:
            raise ValueError("Attention wrapper received no input tensor.")

        # blockは self.attn(x, rope=rope, attn_mask=attn_mask) で呼ぶ
        rope = kwargs.get("rope", None)
        if rope is None and len(args) > 1:
            rope = args[1]

        attn_mask = kwargs.get("attn_mask", None)
        if attn_mask is not None and not getattr(attn_obj, "_warned_attn_mask_ignored", False):
            LOGGER.warning(
                "attn_mask was passed to wrapped attention, but explicit mask application is "
                "not implemented in explain wrapper. Continuing without mask."
            )
            attn_obj._warned_attn_mask_ignored = True

        bsz, n_tokens, dim = x.shape

        q, k, v = _eva_qkv(attn_obj, x)

        q_norm = getattr(attn_obj, "q_norm", None)
        if q_norm is not None:
            q = q_norm(q)
        k_norm = getattr(attn_obj, "k_norm", None)
        if k_norm is not None:
            k = k_norm(k)

        # RoPE適用: prefixトークン(CLS+register)には適用せず、patchトークンのみ回転させる
        if rope is not None:
            num_prefix = int(getattr(attn_obj, "num_prefix_tokens", 1))
            half = bool(getattr(attn_obj, "rotate_half", False))
            q = torch.cat(
                [q[:, :, :num_prefix, :], apply_rot_embed_cat(q[:, :, num_prefix:, :], rope, half=half)],
                dim=2,
            ).type_as(v)
            k = torch.cat(
                [k[:, :, :num_prefix, :], apply_rot_embed_cat(k[:, :, num_prefix:, :], rope, half=half)],
                dim=2,
            ).type_as(v)

        scale = getattr(attn_obj, "scale", None)
        if scale is None:
            scale = (dim // int(getattr(attn_obj, "num_heads", 1))) ** -0.5

        attn = (q * scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        attn_obj._last_attn_map = attn
        attn_obj._last_attn_grad = None
        if attn.requires_grad:
            attn.retain_grad()

            def _save_grad(grad: torch.Tensor) -> None:
                attn_obj._last_attn_grad = grad

            attn.register_hook(_save_grad)

        attn_drop = getattr(attn_obj, "attn_drop", None)
        attn_out = attn_drop(attn) if attn_drop is not None else attn

        x_out = (attn_out @ v).transpose(1, 2).reshape(bsz, n_tokens, -1)

        # EVAのscale_norm(self.norm)。DINOv3 base版ではIdentityなので無影響
        norm = getattr(attn_obj, "norm", None)
        if norm is not None:
            x_out = norm(x_out)

        proj = getattr(attn_obj, "proj", None)
        if proj is not None:
            x_out = proj(x_out)
        proj_drop = getattr(attn_obj, "proj_drop", None)
        if proj_drop is not None:
            x_out = proj_drop(x_out)

        return x_out

    return wrapped_forward


def _install_attention_wrappers(blocks: list[torch.nn.Module]) -> list[tuple[torch.nn.Module, Any, object]]:
    restore_state: list[tuple[torch.nn.Module, Any, object]] = []
    for blk in blocks:
        attn = blk.attn
        original_forward = attn.forward
        had_fused = hasattr(attn, "fused_attn")
        old_fused = getattr(attn, "fused_attn", None)
        if had_fused:
            attn.fused_attn = False
        attn.forward = _make_attention_forward_wrapper(attn, original_forward)
        restore_state.append((attn, original_forward, old_fused if had_fused else _MISSING))
    return restore_state


def _restore_attention_wrappers(restore_state: list[tuple[torch.nn.Module, Any, object]]) -> None:
    for attn, original_forward, old_fused in restore_state:
        attn.forward = original_forward
        if old_fused is _MISSING:
            continue
        attn.fused_attn = old_fused


def _cls_to_patch_tokens(cls_to_tokens: torch.Tensor, num_patches: int | None, image_path: Path) -> torch.Tensor:
    if num_patches is None:
        return cls_to_tokens

    available = cls_to_tokens.shape[-1]
    if available == num_patches:
        return cls_to_tokens

    if available > num_patches:
        LOGGER.warning(
            "Extra non-patch tokens detected for %s (available=%d, patches=%d). Taking last patch tokens.",
            image_path,
            available,
            num_patches,
        )
        return cls_to_tokens[..., -num_patches:]

    raise RuntimeError(
        f"Token count ({available}) smaller than expected patch count ({num_patches}) for {image_path}."
    )


def _infer_grid_size(n_patches: int, image_path: Path) -> int:
    grid = int(math.sqrt(n_patches))
    if grid * grid != n_patches:
        raise RuntimeError(
            f"Patch count is not a perfect square for {image_path}: {n_patches}. "
            "Please verify token filtering / image-size / patch-size settings."
        )
    return grid


_MISSING = object()


def main() -> None:
    args = parse_args()
    ensure_dir(args.out)

    device = resolve_device(args.device)
    model = load_dinov3_model(args.model, device)
    transform = build_transform(args.image_size, args.crop_size)

    blocks = _collect_blocks(model)
    rollout_blocks = _select_blocks_for_rollout(blocks, args.layers)
    restore_state = _install_attention_wrappers(rollout_blocks)
    # DINOv3はCLS+registerトークンを持つ。patch_embed.num_patchesは学習時の
    # デフォルト解像度に固定された値で実解像度と一致しないため使わず、
    # 実際のトークン数 T から num_patches = T - num_prefix_tokens を動的に求める。
    num_prefix_tokens = int(getattr(model, "num_prefix_tokens", 1))

    render_files = list_image_files(args.renders)
    grouped = group_renders_by_specimen(render_files, root_dir=args.renders)
    ids = load_ids(args.ids)
    embs = np.load(args.emb)
    sid_to_idx = {sid: i for i, sid in enumerate(ids)}
    if args.num_show < 1:
        raise ValueError("--num-show must be >= 1")

    if args.specimen_id is not None:
        target_specimen_ids = [args.specimen_id]
    else:
        target_specimen_ids = [sid for sid in ids if sid in grouped]

    if not target_specimen_ids:
        raise RuntimeError("No common specimen IDs found between --ids and --renders.")

    n_ok = 0
    n_skip = 0
    try:
        specimen_iter = tqdm(target_specimen_ids, desc="Specimens", unit="specimen")
        for specimen_id in specimen_iter:
            if specimen_id not in grouped:
                LOGGER.warning("Skipping %s: specimen_id not found in renders.", specimen_id)
                n_skip += 1
                continue
            if specimen_id not in sid_to_idx:
                LOGGER.warning("Skipping %s: specimen_id not found in ids.", specimen_id)
                n_skip += 1
                continue

            z_specimen = torch.from_numpy(embs[sid_to_idx[specimen_id]]).to(device).float()
            z_specimen = F.normalize(z_specimen, dim=0).detach()

            image_paths = grouped[specimen_id]
            n_show = min(args.num_show, len(image_paths))

            fig_w = max(4.0 * n_show, 8.0)
            fig_h = 8.0
            fig_roll, axs_roll = plt.subplots(2, n_show, figsize=(fig_w, fig_h), squeeze=False)
            fig_grad, axs_grad = plt.subplots(2, n_show, figsize=(fig_w, fig_h), squeeze=False)

            success_cols = 0
            view_iter = tqdm(
                enumerate(image_paths[:n_show]),
                total=n_show,
                desc=f"Views ({specimen_id})",
                unit="view",
                leave=False,
            )
            for col, ip in view_iter:
                x = load_image_tensor(ip, transform).unsqueeze(0).to(device)
                x.requires_grad_(True)
                _reset_block_attn_cache(rollout_blocks)

                for p in model.parameters():
                    p.requires_grad_(False)

                model.zero_grad(set_to_none=True)
                z_view = forward_embedding(model, x, enable_grad=True)
                z_view = F.normalize(z_view, dim=-1)
                score = F.cosine_similarity(z_view, z_specimen.unsqueeze(0), dim=-1).sum()
                score.backward()

                attn_maps: list[torch.Tensor] = []
                attn_grads: list[torch.Tensor] = []
                for blk in rollout_blocks:
                    attn_map = getattr(blk.attn, "_last_attn_map", None)
                    if attn_map is None:
                        continue

                    attn_grad = getattr(blk.attn, "_last_attn_grad", None)
                    if attn_grad is None and getattr(attn_map, "grad", None) is not None:
                        attn_grad = attn_map.grad
                    if attn_grad is None:
                        attn_grad = torch.zeros_like(attn_map)

                    attn_maps.append(attn_map)
                    attn_grads.append(attn_grad)

                if not attn_maps:
                    LOGGER.warning("No attention tensor extracted for view: %s. Skipping this view.", ip)
                    continue

                roll = attention_rollout(attn_maps)
                grad_roll = grad_attention_rollout(attn_maps, attn_grads)

                # attention行列のサイズ T からpatch数を算出（CLS+registerを除外）
                n_total_tokens = int(attn_maps[0].shape[-1])
                num_patches = n_total_tokens - num_prefix_tokens

                roll_tokens = _cls_to_patch_tokens(roll[0], num_patches, ip)
                grad_tokens = _cls_to_patch_tokens(grad_roll[0], num_patches, ip)
                grid = _infer_grid_size(int(roll_tokens.shape[-1]), ip)

                heat = to_patch_heatmap(roll_tokens, grid)
                gheat = to_patch_heatmap(grad_tokens, grid)

                img = plt.imread(ip)
                axs_roll[0, col].imshow(img)
                axs_roll[0, col].set_title(Path(ip).name)
                axs_roll[0, col].axis("off")
                axs_roll[1, col].imshow(img)
                axs_roll[1, col].imshow(heat, cmap="jet", alpha=0.45, extent=(0, img.shape[1], img.shape[0], 0))
                axs_roll[1, col].axis("off")

                axs_grad[0, col].imshow(img)
                axs_grad[0, col].set_title(Path(ip).name)
                axs_grad[0, col].axis("off")
                axs_grad[1, col].imshow(img)
                axs_grad[1, col].imshow(gheat, cmap="jet", alpha=0.45, extent=(0, img.shape[1], img.shape[0], 0))
                axs_grad[1, col].axis("off")
                success_cols += 1

            if success_cols == 0:
                LOGGER.warning(
                    "Skipping %s: no valid attention map extracted for any view. ",
                    specimen_id,
                )
                plt.close(fig_roll)
                plt.close(fig_grad)
                n_skip += 1
                continue

            fig_roll.tight_layout()
            fig_grad.tight_layout()

            specimen_out = _specimen_output_dir(args.out, specimen_id)
            ensure_dir(specimen_out)
            fig_roll.savefig(specimen_out / "attention_rollout.png", dpi=220)
            fig_grad.savefig(specimen_out / "grad_rollout_similarity_to_specimen.png", dpi=220)
            plt.close(fig_roll)
            plt.close(fig_grad)
            n_ok += 1
            LOGGER.info("Saved ViT attention explanations for %s to %s", specimen_id, specimen_out)
            specimen_iter.set_postfix(success=n_ok, skipped=n_skip)
    finally:
        _restore_attention_wrappers(restore_state)

    if n_ok == 0:
        raise RuntimeError(
            "No valid attention maps were saved for any specimen. Please check timm version/model attention outputs."
        )

    LOGGER.info("Completed ViT attention explanation. success=%d skipped=%d", n_ok, n_skip)


if __name__ == "__main__":
    main()
