from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from timm.layers import apply_rot_embed_cat
from tqdm.auto import tqdm

from src.utils.explain import attention_rollout, grad_attention_rollout, to_patch_heatmap
from src.utils.io import ensure_dir, group_renders_by_specimen, list_image_files, load_ids
from src.utils.upsampling import add_guided_upsampling_args, guided_output_suffix, guided_upsample
from src.utils.vision import (
    build_transform, forward_embedding, image_tensor_to_rgb,
    load_dinov3_model, load_image_tensor, resolve_device,
)

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
    add_guided_upsampling_args(p)
    p.add_argument(
        "--num-show",
        type=int,
        default=6,
        help="Number of views to visualize. If larger than available views, all available views are shown.",
    )
    # --- Safe-mode options (keep settings such as 768px while preventing/recovering from hangs & crashes) ---
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip specimens whose outputs already exist and resume midway (enabled by default). Lets a re-run continue from where a crash stopped.",
    )
    p.add_argument(
        "--cooldown",
        type=float,
        default=0.0,
        help="Seconds to wait after each specimen to let the GPU cool down (thermal / power-spike relief; 3-10s recommended if time allows).",
    )
    p.add_argument(
        "--vram-fraction",
        type=float,
        default=None,
        help="Upper bound on process VRAM as a fraction (0-1). When set, exceeding it raises OOM and stops safely instead of hanging. e.g. 0.9",
    )
    p.add_argument(
        "--strict-sync",
        action="store_true",
        help="Call torch.cuda.synchronize() after each view to detect CUDA errors early and at their true location (slightly slower, useful for diagnosis).",
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
    """Extract q, k, v as [B, heads, N, head_dim] following the same procedure as EvaAttention.

    DINOv3 (EVA family) may use either a fused qkv (no bias) plus separate q/k/v
    buffers, or separate q_proj/k_proj/v_proj projections, so both layouts are supported.
    """
    bsz, n_tokens, _ = x.shape
    num_heads = int(getattr(attn_obj, "num_heads", 1))

    qkv_layer = getattr(attn_obj, "qkv", None)
    if qkv_layer is not None:
        q_bias = getattr(attn_obj, "q_bias", None)
        if q_bias is None:
            qkv = qkv_layer(x)
        else:
            # The base variant has no bias; the _qkvb variant concatenates q_bias/k_bias/v_bias and applies them
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
    """Faithfully reproduce EvaAttention.forward while capturing the post-softmax attention weights.

    DINOv3 uses RoPE (rotary position embedding), so when recomputing qkv by hand,
    RoPE must be applied or the attention maps will not match the real model. The
    ``rope`` tensor passed from the block is applied to all tokens except the prefix (CLS+register).
    """
    del original_forward

    def wrapped_forward(*args: Any, **kwargs: Any) -> torch.Tensor:
        if len(args) > 0:
            x = args[0]
        else:
            x = kwargs.get("x", None)
        if x is None:
            raise ValueError("Attention wrapper received no input tensor.")

        # The block calls self.attn(x, rope=rope, attn_mask=attn_mask)
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

        # Apply RoPE: do not apply to prefix tokens (CLS+register); rotate only the patch tokens
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

        # EVA's scale_norm (self.norm). It is Identity in the DINOv3 base variant, so it has no effect
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


def _prepare_rollout_overlay(
    image_path: Path,
    model_input: torch.Tensor,
    *heatmaps: np.ndarray,
    guided_upsampling: bool = False,
    guided_radius: int = 1,
    guided_eps: float = 1e-3,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    if not guided_upsampling:
        return plt.imread(image_path), heatmaps
    # Use the exact input crop for BOTH guidance and display. Stretching a map
    # over the uncropped original would incorrectly align attention with edges.
    image = image_tensor_to_rgb(model_input)
    return image, tuple(
        guided_upsample(heat, image, radius=guided_radius, eps=guided_eps)
        for heat in heatmaps
    )


def explain_specimens(
    model: torch.nn.Module,
    target_specimen_ids: list[str],
    grouped: dict[str, list[Path]],
    get_specimen_embedding: Callable[[str], torch.Tensor | None],
    transform: Any,
    out: Path,
    *,
    layers: str = "all",
    num_show: int = 6,
    device: torch.device,
    guided_upsampling: bool = False,
    guided_radius: int = 1,
    guided_eps: float = 1e-3,
) -> tuple[int, int]:
    """Render attention/grad-rollout figures for each target specimen.

    ``get_specimen_embedding(sid)`` returns the grad-rollout REFERENCE embedding
    (a normalized, detached ``[D]`` tensor on ``device``) or ``None`` to skip that
    specimen. Decoupling the embedding source from the rollout machinery lets the
    frozen tool pass stored ``embeddings.npy`` vectors while the out-of-scope
    fine-tuning experiment (``experiments/explain_finetuned.py``) passes vectors
    recomputed with the SAME fine-tuned model, so grad-rollout is not a
    frozen/fine-tuned hybrid. Returns ``(n_ok, n_skip)``; raises if nothing saved.
    """
    if num_show < 1:
        raise ValueError("--num-show must be >= 1")
    output_suffix = guided_output_suffix(guided_upsampling, guided_radius, guided_eps)
    heatmap_range = {"vmin": 0, "vmax": 1} if guided_upsampling else {}

    blocks = _collect_blocks(model)
    rollout_blocks = _select_blocks_for_rollout(blocks, layers)
    restore_state = _install_attention_wrappers(rollout_blocks)
    # DINOv3 has CLS+register tokens. patch_embed.num_patches is fixed to the
    # training-time default resolution and does not match the actual resolution,
    # so derive num_patches = T - num_prefix_tokens dynamically from the actual token count T.
    num_prefix_tokens = int(getattr(model, "num_prefix_tokens", 1))

    n_ok = 0
    n_skip = 0
    try:
        specimen_iter = tqdm(target_specimen_ids, desc="Specimens", unit="specimen")
        for specimen_id in specimen_iter:
            if specimen_id not in grouped:
                LOGGER.warning("Skipping %s: specimen_id not found in renders.", specimen_id)
                n_skip += 1
                continue

            z_specimen = get_specimen_embedding(specimen_id)
            if z_specimen is None:
                LOGGER.warning("Skipping %s: no reference embedding available.", specimen_id)
                n_skip += 1
                continue

            image_paths = grouped[specimen_id]
            n_show = min(num_show, len(image_paths))

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

                # Derive the patch count from the attention matrix size T (excluding CLS+register).
                n_total_tokens = int(attn_maps[0].shape[-1])
                num_patches = n_total_tokens - num_prefix_tokens

                roll_tokens = _cls_to_patch_tokens(roll[0], num_patches, ip)
                grad_tokens = _cls_to_patch_tokens(grad_roll[0], num_patches, ip)
                grid = _infer_grid_size(int(roll_tokens.shape[-1]), ip)

                heat = to_patch_heatmap(roll_tokens, grid)
                gheat = to_patch_heatmap(grad_tokens, grid)

                img, (heat, gheat) = _prepare_rollout_overlay(
                    ip, x[0], heat, gheat,
                    guided_upsampling=guided_upsampling,
                    guided_radius=guided_radius, guided_eps=guided_eps,
                )
                axs_roll[0, col].imshow(img)
                axs_roll[0, col].set_title(Path(ip).name)
                axs_roll[0, col].axis("off")
                axs_roll[1, col].imshow(img)
                axs_roll[1, col].imshow(heat, cmap="jet", alpha=0.45, extent=(0, img.shape[1], img.shape[0], 0), **heatmap_range)
                axs_roll[1, col].axis("off")

                axs_grad[0, col].imshow(img)
                axs_grad[0, col].set_title(Path(ip).name)
                axs_grad[0, col].axis("off")
                axs_grad[1, col].imshow(img)
                axs_grad[1, col].imshow(gheat, cmap="jet", alpha=0.45, extent=(0, img.shape[1], img.shape[0], 0), **heatmap_range)
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

            specimen_out = _specimen_output_dir(out, specimen_id)
            ensure_dir(specimen_out)
            fig_roll.savefig(specimen_out / f"attention_rollout{output_suffix}.png", dpi=220)
            fig_grad.savefig(specimen_out / f"grad_rollout_similarity_to_specimen{output_suffix}.png", dpi=220)
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
    return n_ok, n_skip


def main() -> None:
    args = parse_args()
    ensure_dir(args.out)
    output_suffix = guided_output_suffix(args.guided_upsampling, args.guided_radius, args.guided_eps)
    heatmap_range = {"vmin": 0, "vmax": 1} if args.guided_upsampling else {}

    device = resolve_device(args.device)
    model = load_dinov3_model(args.model, device)
    transform = build_transform(args.image_size, args.crop_size)

    # Safe mode: capping VRAM as a fraction makes it stop with OOM instead of hanging when the cap is exceeded
    if device.type == "cuda" and args.vram_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(args.vram_fraction, device.index or 0)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    blocks = _collect_blocks(model)
    rollout_blocks = _select_blocks_for_rollout(blocks, args.layers)
    restore_state = _install_attention_wrappers(rollout_blocks)
    # DINOv3 has CLS+register tokens. patch_embed.num_patches is fixed to the
    # training-time default resolution and does not match the actual resolution,
    # so instead derive num_patches = T - num_prefix_tokens dynamically from the actual token count T.
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
    n_resumed = 0
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

            # Safe mode (resume): if both outputs already exist, skip without recomputing
            specimen_out = _specimen_output_dir(args.out, specimen_id)
            roll_out_path = specimen_out / f"attention_rollout{output_suffix}.png"
            grad_out_path = specimen_out / f"grad_rollout_similarity_to_specimen{output_suffix}.png"
            if args.resume and roll_out_path.exists() and grad_out_path.exists():
                n_skip += 1
                n_resumed += 1
                specimen_iter.set_postfix(success=n_ok, skipped=n_skip)
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
                try:
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
                    if args.strict_sync and device.type == "cuda":
                        torch.cuda.synchronize(device)

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

                    # Derive the patch count from the attention matrix size T (excluding CLS+register)
                    n_total_tokens = int(attn_maps[0].shape[-1])
                    num_patches = n_total_tokens - num_prefix_tokens

                    roll_tokens = _cls_to_patch_tokens(roll[0], num_patches, ip)
                    grad_tokens = _cls_to_patch_tokens(grad_roll[0], num_patches, ip)
                    grid = _infer_grid_size(int(roll_tokens.shape[-1]), ip)

                    heat = to_patch_heatmap(roll_tokens, grid)
                    gheat = to_patch_heatmap(grad_tokens, grid)

                    img, (heat, gheat) = _prepare_rollout_overlay(
                        ip, x[0], heat, gheat,
                        guided_upsampling=args.guided_upsampling,
                        guided_radius=args.guided_radius, guided_eps=args.guided_eps,
                    )
                    axs_roll[0, col].imshow(img)
                    axs_roll[0, col].set_title(Path(ip).name)
                    axs_roll[0, col].axis("off")
                    axs_roll[1, col].imshow(img)
                    axs_roll[1, col].imshow(heat, cmap="jet", alpha=0.45, extent=(0, img.shape[1], img.shape[0], 0), **heatmap_range)
                    axs_roll[1, col].axis("off")

                    axs_grad[0, col].imshow(img)
                    axs_grad[0, col].set_title(Path(ip).name)
                    axs_grad[0, col].axis("off")
                    axs_grad[1, col].imshow(img)
                    axs_grad[1, col].imshow(gheat, cmap="jet", alpha=0.45, extent=(0, img.shape[1], img.shape[0], 0), **heatmap_range)
                    axs_grad[1, col].axis("off")
                    success_cols += 1

                    # Safe mode: explicitly free the large per-view tensors to keep the VRAM peak at ~one view
                    del x, z_view, score, attn_maps, attn_grads, roll, grad_roll
                    del roll_tokens, grad_tokens, heat, gheat, img
                except torch.cuda.OutOfMemoryError:
                    # OOM is recoverable: give up only this view, free the cache, and keep going
                    LOGGER.warning(
                        "Out of VRAM; skipping view: %s. If this happens often, reduce --num-show or "
                        "lower --image-size/--crop-size.",
                        ip,
                    )
                finally:
                    # Reliably drop attention-map references between views and free the cache to prevent hangs
                    _reset_block_attn_cache(rollout_blocks)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

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

            ensure_dir(specimen_out)
            fig_roll.savefig(roll_out_path, dpi=220)
            fig_grad.savefig(grad_out_path, dpi=220)
            plt.close(fig_roll)
            plt.close(fig_grad)
            n_ok += 1
            LOGGER.info("Saved ViT attention explanations for %s to %s", specimen_id, specimen_out)

            # Safe mode: record and free the VRAM peak, and wait for cool-down if requested
            postfix: dict[str, Any] = {"success": n_ok, "skipped": n_skip}
            if device.type == "cuda":
                peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3
                postfix["peak_gb"] = round(peak_gb, 2)
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.empty_cache()
            specimen_iter.set_postfix(**postfix)
            if args.cooldown > 0:
                time.sleep(args.cooldown)
    finally:
        _restore_attention_wrappers(restore_state)

    if n_ok == 0 and n_resumed == 0:
        raise RuntimeError(
            "No valid attention maps were saved for any specimen. Please check timm version/model attention outputs."
        )

    LOGGER.info("Completed ViT attention explanation. success=%d skipped=%d", n_ok, n_skip)


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError as exc:
        LOGGER.error(
            "Aborted due to out of VRAM (%s). Re-run with --resume (default) to continue from where it stopped. "
            "If it persists, reduce --num-show or set --vram-fraction 0.9 to cap memory.",
            exc,
        )
        sys.exit(2)
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "cuda" in msg or "device-side" in msg or "out of memory" in msg:
            # A corrupted CUDA context (e.g. unknown error) cannot be recovered in-process; exit safely and let the restart loop take over
            LOGGER.error(
                "Aborted due to a fatal GPU error (%s). The CUDA context is corrupted, so exiting safely. "
                "Re-run with --resume (default) to continue from where it stopped. If it recurs, use "
                "scripts/run_safe.sh (auto-restart loop), cap the GPU power (nvidia-smi -pl), and set "
                "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.",
                exc,
            )
            sys.exit(1)
        raise
