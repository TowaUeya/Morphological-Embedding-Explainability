"""RGB-guided joint upsampling for normalized patch heatmaps.

Fit the guided filter's local linear model at patch resolution, bilinearly
upsample its coefficients, and evaluate them against the full-resolution RGB
guide (He et al., Guided Image Filtering, joint upsampling application):
https://people.csail.mit.edu/kaiming/publications/pami12guidedfilter.pdf
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be an integer >= 1")
    return number


def _positive_float(value: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and > 0")
    return number


def add_guided_upsampling_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--guided-upsampling", action="store_true",
        help="Upsample heatmaps using RGB guidance from the model's resized/center-cropped input. Save separate guided PNGs.",
    )
    parser.add_argument(
        "--guided-radius", type=_positive_int, default=1,
        help="Guided-filter window radius in PATCH cells (>= 1; default: 1, a 3x3 window).",
    )
    parser.add_argument(
        "--guided-eps", type=_positive_float, default=1e-3,
        help="Guided-filter regularization for RGB in [0,1] (> 0; default: 0.001). Larger values weaken edge guidance.",
    )


def guided_output_suffix(enabled: bool, radius: int = 1, eps: float = 1e-3) -> str:
    """Keep ordinary/guided outputs and different guided settings distinct for resume."""
    return f"_guided_r{radius}_eps{float(eps)}" if enabled else ""


def _box_mean(array: np.ndarray, radius: int) -> np.ndarray:
    """Spatial box mean with clipped windows, including for tiny patch grids."""
    height, width = array.shape[:2]
    integral = np.pad(
        array, ((1, 0), (1, 0)) + ((0, 0),) * (array.ndim - 2),
    ).cumsum(axis=0).cumsum(axis=1)
    ys, xs = np.arange(height), np.arange(width)
    y0, y1 = np.maximum(ys - radius, 0), np.minimum(ys + radius + 1, height)
    x0, x1 = np.maximum(xs - radius, 0), np.minimum(xs + radius + 1, width)
    total = (
        integral[y1[:, None], x1] - integral[y0[:, None], x1]
        - integral[y1[:, None], x0] + integral[y0[:, None], x0]
    )
    count = (y1 - y0)[:, None] * (x1 - x0)
    return total / count.reshape(count.shape + (1,) * (array.ndim - 2))


def guided_upsample(
    heatmap: np.ndarray,
    guidance: np.ndarray,
    *,
    radius: int = 1,
    eps: float = 1e-3,
) -> np.ndarray:
    """Return an HxW float32 map using an aligned HxWx3 RGB guide in [0,1].

    ``radius`` is measured in patch cells, not output pixels. All computations
    run on the CPU after rollout; neither model predictions nor gradients change.
    Clip filter overshoot to [0,1] without another min-max normalization, so a
    constant/zero attention map stays constant/zero.
    """
    if isinstance(radius, bool) or not isinstance(radius, (int, np.integer)) or radius < 1:
        raise ValueError("radius must be an integer >= 1")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and > 0")
    heat = np.asarray(heatmap, dtype=np.float64)
    guide = np.asarray(guidance, dtype=np.float64)
    if heat.ndim != 2 or 0 in heat.shape:
        raise ValueError("heatmap must be a nonempty 2D array")
    if guide.ndim != 3 or guide.shape[2] != 3 or 0 in guide.shape:
        raise ValueError("guidance must be a nonempty HxWx3 RGB array")
    if not np.isfinite(heat).all() or heat.min() < 0 or heat.max() > 1:
        raise ValueError("heatmap must contain finite values in [0,1]")
    if not np.isfinite(guide).all() or guide.min() < 0 or guide.max() > 1:
        raise ValueError("guidance must contain finite RGB values in [0,1]")

    # Average RGB over each patch, matching the support of its attention value.
    low_guide = F.interpolate(
        torch.from_numpy(guide).permute(2, 0, 1).unsqueeze(0),
        size=heat.shape, mode="area",
    )[0].permute(1, 2, 0).numpy()
    mean_i = _box_mean(low_guide, radius)
    mean_p = _box_mean(heat, radius)
    covariance = (
        _box_mean(low_guide[..., :, None] * low_guide[..., None, :], radius)
        - mean_i[..., :, None] * mean_i[..., None, :]
    )
    covariance_ip = _box_mean(low_guide * heat[..., None], radius) - mean_i * mean_p[..., None]
    # Solve the regularized RGB system in float64, also stable for flat guides.
    a = np.linalg.solve(covariance + eps * np.eye(3), covariance_ip[..., None])[..., 0]
    b = mean_p - (a * mean_i).sum(axis=-1)
    coefficients = np.concatenate((a, b[..., None]), axis=-1)
    full = F.interpolate(
        torch.from_numpy(coefficients).permute(2, 0, 1).unsqueeze(0),
        size=guide.shape[:2], mode="bilinear", align_corners=False,
    )[0].permute(1, 2, 0).numpy()
    result = (full[..., :3] * guide).sum(axis=-1) + full[..., 3]
    return np.clip(result, 0.0, 1.0).astype(np.float32)
