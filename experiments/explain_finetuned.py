"""Side experiment: attention-rollout visualization of a FINE-TUNED DINOv3.

Scope (read experiments/README.md)
----------------------------------
An optional side track to this repository's main tool, which explains a FROZEN
DINOv3 (see ``src/explain_vit_attention.py`` and ``../README.md``). It visualizes a
model whose backbone was updated with labels by the sister repo's fine-tuner
(``MultiView3D-DINOv2/experiments/finetune_dinov3.py``). A fine-tuned rollout answers
"where did the model learn to look to separate THESE classes on THIS data", a
different object from the frozen rollout's "where does a general-purpose backbone
attend" -- keep the two separate and never mix them in a single claim.

Why a separate script (not just re-running src/explain_vit_attention.py)
------------------------------------------------------------------------
Two things must change versus the frozen tool, and this script does both:

  1. Reconstruct the fine-tuned model. The sister repo's ``finetune_dinov3.py`` saves
     only the trained delta (``finetuned_adapter.pt`` = LoRA adapters / unfrozen params
     + head), never the whole backbone. We load a pristine DINOv3, re-apply the same
     structure recorded in ``summary.json`` (LoRA injection with the same
     rank/alpha/targets, or a no-op for the blocks/full modes), then load the delta on
     top. The pieces needed to reconstruct it live in ``experiments/_finetune_utils.py``.
  2. Regenerate the grad-rollout REFERENCE embedding with that SAME fine-tuned
     model (token-mean over [CLS + patch] per view, then mean over views -- the
     frozen pipeline's pooling). Pointing grad-rollout at a stored frozen
     ``embeddings.npy`` while visualizing a fine-tuned model would paint a
     frozen/fine-tuned hybrid quantity; recomputing keeps it coherent.

The shared attention-capture / rollout machinery is imported from
``src.explain_vit_attention`` (``explain_specimens``); only the two differences
above live here.

Run from the repo root (src/ is a namespace package; ``-m`` keeps imports working)::

    python -m experiments.explain_finetuned \\
        --renders data/renders/fish_bone \\
        --adapter-dir results/finetune_fish_bone_lora_ce \\
        --out results/explain_finetuned_fish_bone \\
        --num-show 8
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments._finetune_utils import _LORA_TARGET_SETS, extract_specimen_embeddings, inject_lora
from src.explain_vit_attention import explain_specimens
from src.utils.io import ensure_dir, group_renders_by_specimen, list_image_files, setup_logging
from src.utils.upsampling import add_guided_upsampling_args
from src.utils.vision import build_transform, load_dinov3_model, resolve_device

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Explain a FINE-TUNED DINOv3 via attention rollout (optional side track)"
    )
    p.add_argument("--renders", type=Path, required=True, help="dir of {sid}_viewNN.png")
    p.add_argument(
        "--adapter-dir",
        type=Path,
        required=True,
        help="a finetune_dinov3 --out dir (must contain finetuned_adapter.pt + summary.json)",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--specimen_id",
        type=str,
        default=None,
        help="Target specimen ID. If omitted, all specimens found under --renders are processed.",
    )
    p.add_argument("--layers", type=str, choices=("all", "last"), default="all")
    add_guided_upsampling_args(p)
    p.add_argument("--num-show", type=int, default=6)
    # Resolution defaults to summary.json (same as training); override only to experiment.
    p.add_argument("--image-size", type=int, default=None, help="override; default = fine-tune image_size")
    p.add_argument("--crop-size", type=int, default=None, help="override; default = fine-tune crop_size")
    p.add_argument("--batch-size", type=int, default=16, help="batch size for reference-embedding extraction")
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def load_finetuned_model(adapter_dir: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Reconstruct the fine-tuned model from ``finetuned_adapter.pt`` + ``summary.json``.

    The fine-tuner saves only the trained delta (not the whole backbone). So we load a
    pristine DINOv3, rebuild the same structure recorded in summary.json (for LoRA,
    inject with the same rank/alpha/targets; blocks/full need no structural change),
    then overlay the delta on top.
    """
    summary_path = adapter_dir / "summary.json"
    ckpt_path = adapter_dir / "finetuned_adapter.pt"
    if not summary_path.exists() or not ckpt_path.exists():
        raise FileNotFoundError(f"{adapter_dir} must contain both summary.json and finetuned_adapter.pt")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cfg = summary["config"]
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["backbone_trainable"]

    model = load_dinov3_model(cfg["model"], device)

    mode = cfg["finetune"]
    if mode == "lora":
        n = inject_lora(
            model,
            _LORA_TARGET_SETS[cfg["lora_targets"]],
            int(cfg["lora_rank"]),
            int(cfg["lora_alpha"]),
            0.0,  # dropout is irrelevant at eval
        )
        model.to(device)  # move the new LoRA layers to device
        LOGGER.info(
            "re-injected %d LoRA layers (targets=%s r=%d alpha=%d)",
            n, cfg["lora_targets"], cfg["lora_rank"], cfg["lora_alpha"],
        )
    else:
        LOGGER.info("finetune mode=%s: no structural change, loading trained params on top", mode)

    # Overlay only the delta (missing keys = still-frozen, fine; unexpected keys = structure mismatch, fatal)
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if unexpected:
        raise RuntimeError(
            "adapter contains keys not present in the reconstructed model "
            f"(structure mismatch vs summary.json): e.g. {unexpected[:6]}"
        )
    model.eval()  # disable dropout / stochastic depth for deterministic visualization
    LOGGER.info("loaded fine-tuned delta: %d tensors (mode=%s)", len(state), mode)
    return model, summary


def main() -> None:
    setup_logging()
    args = parse_args()
    ensure_dir(args.out)
    device = resolve_device(args.device)

    model, summary = load_finetuned_model(args.adapter_dir, device)
    cfg = summary["config"]

    image_size = args.image_size if args.image_size is not None else int(cfg["image_size"])
    crop_size = args.crop_size if args.crop_size is not None else int(cfg["crop_size"])
    transform = build_transform(image_size, crop_size)
    LOGGER.info(
        "visualizing at image_size=%d crop_size=%d (fine-tune used %s/%s)",
        image_size, crop_size, cfg["image_size"], cfg["crop_size"],
    )

    render_files = list_image_files(args.renders)
    if not render_files:
        raise FileNotFoundError(f"no render images under {args.renders}")
    grouped = group_renders_by_specimen(render_files, root_dir=args.renders)

    if args.specimen_id is not None:
        target_specimen_ids = [args.specimen_id]
    else:
        target_specimen_ids = sorted(grouped)

    # Recompute the reference embedding with the fine-tuned model (same aggregation as the
    # frozen pipeline: token mean -> view mean). This is the grad-rollout target; using the
    # frozen embeddings.npy would make it a frozen/fine-tuned hybrid and muddy the interpretation.
    in_renders = [s for s in target_specimen_ids if s in grouped]
    if not in_renders:
        raise RuntimeError("None of the target specimens were found under --renders.")
    ref = extract_specimen_embeddings(model, in_renders, grouped, transform, device, args.batch_size)
    sid_to_ref = {sid: ref[i] for i, sid in enumerate(in_renders)}
    LOGGER.info("recomputed fine-tuned reference embeddings for %d specimens", len(sid_to_ref))

    def get_specimen_embedding(specimen_id: str) -> torch.Tensor | None:
        vec = sid_to_ref.get(specimen_id)
        if vec is None:
            return None
        z = torch.from_numpy(np.asarray(vec)).to(device).float()
        return F.normalize(z, dim=0).detach()

    # Record provenance (make it explicit next to the figure that this is a fine-tuned visualization)
    provenance = {
        "note": "Attention rollout of a FINE-TUNED DINOv3 (optional side track); keep separate from frozen-baseline claims.",
        "adapter_dir": str(args.adapter_dir),
        "finetune_config": cfg,
        "visualization": {
            "image_size": image_size,
            "crop_size": crop_size,
            "layers": args.layers,
            "guided_upsampling": args.guided_upsampling,
            "guided_radius": args.guided_radius,
            "guided_eps": args.guided_eps,
            "num_show": args.num_show,
            "n_target_specimens": len(in_renders),
        },
        "reference_embedding": "recomputed with the fine-tuned model (token-mean per view, then view-mean)",
    }
    (args.out / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")

    n_ok, n_skip = explain_specimens(
        model,
        target_specimen_ids,
        grouped,
        get_specimen_embedding,
        transform,
        args.out,
        layers=args.layers,
        num_show=args.num_show,
        device=device,
        guided_upsampling=args.guided_upsampling,
        guided_radius=args.guided_radius,
        guided_eps=args.guided_eps,
    )
    LOGGER.info("done: success=%d skipped=%d -> %s", n_ok, n_skip, args.out)


if __name__ == "__main__":
    main()
