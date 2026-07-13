"""Side experiment: frozen vs fine-tuned grad-rollout difference for one specimen.

Scope (read experiments/README.md)
----------------------------------
An optional side track to this repository's frozen explainability tool. This compares
where a pristine (frozen) DINOv3 attends versus a LoRA/blocks/full fine-tuned one, on
the SAME specimen at the SAME resolution, and draws the difference. It is a
qualitative side-by-side; keep it separate from frozen-baseline claims.

What it does
------------
For one specimen it computes the grad-rollout heatmap (similarity to the
specimen's own pooled embedding) with BOTH models at the fine-tune resolution,
using each model's OWN recomputed reference embedding (so neither map is a
frozen/fine-tuned hybrid), and saves a 4-row figure: view / frozen / fine-tuned /
difference (fine-tuned - frozen). Positive (red) = fine-tuned attends more.

Held-out safety
---------------
By default it visualizes a HELD-OUT specimen (read from the adapter dir's
heldout_ids.txt) so the attention reflects generalization, not memorization of a
training specimen. If --specimen_id names a training specimen, it warns.

Run from the repo root::

    python -m experiments.compare_rollout \\
        --renders data/renders/fish_bone \\
        --adapter-dir results/finetune_fish_bone_lora_ce \\
        --out results/compare_rollout/fish_bone --num-show 6
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from experiments.explain_finetuned import load_finetuned_model
from experiments._finetune_utils import extract_specimen_embeddings
from src.explain_vit_attention import (
    _cls_to_patch_tokens,
    _collect_blocks,
    _infer_grid_size,
    _install_attention_wrappers,
    _reset_block_attn_cache,
    _restore_attention_wrappers,
    _select_blocks_for_rollout,
)
from src.utils.explain import grad_attention_rollout, to_patch_heatmap
from src.utils.io import ensure_dir, group_renders_by_specimen, list_image_files, setup_logging
from src.utils.vision import (
    build_transform,
    forward_embedding,
    load_dinov3_model,
    load_image_tensor,
    resolve_device,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Frozen vs fine-tuned grad-rollout difference for one specimen (optional side track)"
    )
    p.add_argument("--renders", type=Path, required=True)
    p.add_argument("--adapter-dir", type=Path, required=True, help="finetune_dinov3 --out dir")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--specimen_id", type=str, default=None,
                   help="default: first held-out specimen present under --renders")
    p.add_argument("--num-show", type=int, default=6)
    p.add_argument("--layers", type=str, choices=("all", "last"), default="all")
    p.add_argument("--image-size", type=int, default=None, help="override; default = fine-tune image_size")
    p.add_argument("--crop-size", type=int, default=None, help="override; default = fine-tune crop_size")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def grad_rollout_heatmaps(model, z_specimen, paths, transform, layers, device):
    """Per-view grad-rollout heatmaps (normalized to [0,1]). Identical to explain_specimens' per-view computation."""
    blocks = _select_blocks_for_rollout(_collect_blocks(model), layers)
    restore = _install_attention_wrappers(blocks)
    num_prefix = int(getattr(model, "num_prefix_tokens", 1))
    heats = []
    try:
        for ip in paths:
            x = load_image_tensor(ip, transform).unsqueeze(0).to(device)
            x.requires_grad_(True)
            _reset_block_attn_cache(blocks)
            for p in model.parameters():
                p.requires_grad_(False)
            model.zero_grad(set_to_none=True)
            zv = F.normalize(forward_embedding(model, x, enable_grad=True), dim=-1)
            score = F.cosine_similarity(zv, z_specimen.unsqueeze(0), dim=-1).sum()
            score.backward()
            amaps, agrads = [], []
            for blk in blocks:
                am = getattr(blk.attn, "_last_attn_map", None)
                if am is None:
                    continue
                ag = getattr(blk.attn, "_last_attn_grad", None)
                if ag is None:
                    ag = torch.zeros_like(am)
                amaps.append(am)
                agrads.append(ag)
            gr = grad_attention_rollout(amaps, agrads)
            n_patch = int(amaps[0].shape[-1]) - num_prefix
            gt = _cls_to_patch_tokens(gr[0], n_patch, ip)
            heats.append(to_patch_heatmap(gt, _infer_grid_size(int(gt.shape[-1]), ip)))
    finally:
        _restore_attention_wrappers(restore)
    return heats


def _reference(model, sid, groups, transform, device, batch_size):
    """Build the specimen's reference embedding (token mean -> view mean, normalized) with the given model."""
    vec = extract_specimen_embeddings(model, [sid], groups, transform, device, batch_size)[0]
    return F.normalize(torch.from_numpy(np.asarray(vec)).to(device).float(), dim=0).detach()


def main() -> None:
    setup_logging()
    args = parse_args()
    ensure_dir(args.out)
    device = resolve_device(args.device)

    ftm, summary = load_finetuned_model(args.adapter_dir, device)
    cfg = summary["config"]
    image_size = args.image_size if args.image_size is not None else int(cfg["image_size"])
    crop_size = args.crop_size if args.crop_size is not None else int(cfg["crop_size"])
    transform = build_transform(image_size, crop_size)

    groups = group_renders_by_specimen(list_image_files(args.renders), root_dir=args.renders)

    # Held-out set (specimens not used for training). By default the visualization target is picked from here.
    heldout_path = args.adapter_dir / "heldout_ids.txt"
    heldout = set(heldout_path.read_text().split()) if heldout_path.exists() else set()

    if args.specimen_id is not None:
        sid = args.specimen_id
        if heldout and sid not in heldout:
            LOGGER.warning(
                "specimen '%s' is NOT in heldout_ids.txt -> likely a TRAINING specimen; "
                "rollout may reflect memorization, not generalization.", sid,
            )
    else:
        candidates = [s for s in sorted(heldout) if s in groups] or sorted(groups)
        if not candidates:
            raise RuntimeError("no specimen found under --renders")
        sid = candidates[0]
        LOGGER.info("auto-selected held-out specimen: %s", sid)

    if sid not in groups:
        raise RuntimeError(f"specimen '{sid}' not found under --renders")
    paths = groups[sid][: args.num_show]

    LOGGER.info("visualizing %s at %dpx (%d views); adapter mode=%s",
                sid, image_size, len(paths), cfg["finetune"])

    pm = load_dinov3_model(cfg["model"], device)
    Hf = grad_rollout_heatmaps(pm, _reference(pm, sid, groups, transform, device, args.batch_size),
                               paths, transform, args.layers, device)
    Ht = grad_rollout_heatmaps(ftm, _reference(ftm, sid, groups, transform, device, args.batch_size),
                               paths, transform, args.layers, device)

    n = len(paths)
    fig, axs = plt.subplots(4, n, figsize=(4 * n, 15), squeeze=False)
    row_labels = ["view", "frozen grad-rollout", "fine-tuned grad-rollout", "diff = fine-tuned - frozen"]
    diff_im = None
    stats = []
    for c, ip in enumerate(paths):
        img = plt.imread(ip)
        ext = (0, img.shape[1], img.shape[0], 0)
        axs[0, c].imshow(img)
        axs[0, c].set_title(Path(ip).name, fontsize=8)
        axs[1, c].imshow(img)
        axs[1, c].imshow(Hf[c], cmap="jet", alpha=0.45, extent=ext)
        axs[2, c].imshow(img)
        axs[2, c].imshow(Ht[c], cmap="jet", alpha=0.45, extent=ext)
        d = Ht[c] - Hf[c]
        axs[3, c].imshow(img, alpha=0.35)
        diff_im = axs[3, c].imshow(d, cmap="bwr", vmin=-1, vmax=1, alpha=0.75, extent=ext)
        for r in range(4):
            axs[r, c].axis("off")
        stats.append((Path(ip).name, float(np.abs(d).mean()), float(d.max()), float(d.min())))
    for r, lbl in enumerate(row_labels):
        axs[r, 0].text(-0.04, 0.5, lbl, transform=axs[r, 0].transAxes,
                       rotation=90, va="center", ha="right", fontsize=11)
    fig.suptitle(
        f"{sid}: grad-rollout (similarity to specimen) - frozen vs fine-tuned ({cfg['finetune']}) @{image_size}\n"
        "diff row: red = fine-tuned attends MORE, blue = LESS "
        "(each heatmap min-max normalized to [0,1]; relative shift)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.02, 0.03, 1, 0.95))
    fig.colorbar(diff_im, ax=axs[3, :].tolist(), orientation="horizontal", fraction=0.05, pad=0.02).set_label(
        "fine-tuned - frozen (relative attention shift)"
    )
    safe = sid.replace("/", "_")
    outpath = args.out / f"{safe}_grad_rollout_diff.png"
    fig.savefig(outpath, dpi=170)
    plt.close(fig)

    for name, m, mx, mn in stats:
        LOGGER.info("%s: mean|diff|=%.3f max+=%+.3f min-=%+.3f", name, m, mx, mn)
    LOGGER.info("saved %s", outpath)


if __name__ == "__main__":
    main()
