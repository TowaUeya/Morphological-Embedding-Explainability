"""Self-contained helpers to RECONSTRUCT and READ a fine-tuned DINOv3 for rollout.

Why this file exists
--------------------
The rollout scripts in this directory consume artifacts produced by the sister
repository's fine-tuner (``MultiView3D-DINOv2/experiments/finetune_dinov3.py``):
a ``finetuned_adapter.pt`` delta plus a ``summary.json``. To turn those files back
into a runnable model and to recompute the grad-rollout reference embedding, the
rollout side needs the SAME LoRA-injection structure and the SAME
token-mean/view-mean pooling the fine-tuner used.

Rather than importing across repositories (which would couple the two clones on
disk), the few pieces required to *read back* a fine-tuned model are duplicated
here so this repo runs standalone. The contract between the two repos is the
artifact format on disk, not a shared Python import -- consistent with the
ecosystem's file-passing design.

Keep in sync with ``finetune_dinov3.py`` in the sister repo: ``_LORA_TARGET_SETS``,
``LoRALinear``, ``inject_lora`` and the pooling in ``extract_specimen_embeddings``
must match the fine-tuner, or a saved adapter will not reload cleanly.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.utils.vision import forward_embedding

# Attention/MLP Linear layer names LoRA can wrap (matched by the child attribute name).
_LORA_TARGET_SETS = {
    "attn": {"qkv", "proj"},               # default: attention only (standard LoRA target)
    "attn_mlp": {"qkv", "proj", "fc1", "fc2"},  # attention + MLP (more capacity, more params)
}


class LoRALinear(nn.Module):
    """Wrap an existing ``nn.Linear`` with a low-rank adapter: y = W0 x + (alpha/r) * B(A(x)).

    The base weight is frozen and only A/B train. B is zero-initialized, so at the
    start of training the output matches the original Linear exactly (identity).
    This mirrors ``LoRALinear`` in the sister repo's fine-tuner; the two must stay
    in sync so a saved adapter reloads cleanly here.
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = int(r)
        self.scale = float(alpha) / float(r)
        self.a = nn.Linear(base.in_features, r, bias=False)
        self.b = nn.Linear(r, base.out_features, bias=False)
        self.drop = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * self.b(self.a(self.drop(x)))


def inject_lora(model: nn.Module, target_names: set[str], r: int, alpha: int, dropout: float) -> int:
    """Replace every Linear under ``blocks.*`` whose attribute name is in ``target_names``
    with a ``LoRALinear``. Must reproduce the fine-tuner's injection exactly so the
    reconstructed model has the parameter names the saved adapter expects. Returns
    the number of layers wrapped.
    """
    wrapped = 0
    for module_name, module in list(model.named_modules()):
        # Only wrap Linear layers inside transformer blocks (leave patch_embed etc. alone).
        if "blocks." not in module_name:
            continue
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in target_names:
                setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                wrapped += 1
    return wrapped


def image_embedding(model: nn.Module, batch: torch.Tensor, *, enable_grad: bool) -> torch.Tensor:
    """Per-image representation = token mean over [CLS+patch] -> [B, D].

    Matches the frozen pipeline pooling (extract_features -> pool_embeddings) so the
    recomputed reference embedding is the SAME quantity the retrieval metric uses.
    """
    feats = forward_embedding(model, batch, enable_grad=enable_grad)  # [B, T, D] (registers already dropped)
    return feats.mean(dim=1)  # token mean -> [B, D]


@torch.no_grad()
def extract_specimen_embeddings(
    model: nn.Module,
    specimens: list[str],
    render_groups: dict[str, list[Path]],
    transform,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Aggregate each specimen to [D] (token mean -> view mean).

    Same aggregation as the frozen pipeline, so the reference the grad-rollout score
    points at is coherent with how the fine-tuned model would be evaluated.
    """
    model.eval()
    out = np.zeros((len(specimens), int(getattr(model, "embed_dim", 0) or model.num_features)), dtype=np.float64)
    for i, sid in enumerate(specimens):
        paths = render_groups[sid]
        view_embs: list[torch.Tensor] = []
        imgs = [transform(Image.open(p).convert("RGB")) for p in paths]
        views = torch.stack(imgs)  # [V, C, H, W]
        for idx in range(0, views.shape[0], batch_size):
            bt = views[idx: idx + batch_size].to(device)
            view_embs.append(image_embedding(model, bt, enable_grad=False).float().cpu())
        emb = torch.cat(view_embs, dim=0).mean(dim=0)  # view mean -> [D]
        out[i] = emb.numpy().astype(np.float64)
    return out
