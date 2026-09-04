# Morphological Embedding Explainability: Attention rollout visualization for multi-view DINOv2 representations
Explainability toolkit for visualizing attention rollout, gradient-weighted attention rollout, and image-level cues associated with ViT-based embedding formation.

This repository does not compute embeddings or perform clustering. It visualizes attention-based cues using rendered views and precomputed embeddings.

## Requirements
- Python 3.10+
- PyTorch-compatible environment (CPU or CUDA)
- Dependencies in `requirements.txt`

## Installation
```bash
pip install -r requirements.txt
```

## Input Data
Prepare the following files before running:
- Rendered multi-view images directory (`--renders`)
- `embeddings.npy` (`--emb`)
- `ids.txt` (`--ids`)
- Optional `clusters.csv` (`--clusters`)

Minimal example layout:
```text
data/
  renders/
  embeddings/
    embeddings.npy
    ids.txt
    clusters.csv   # optional
```

## Usage
Run explainability visualization with:

```bash
python -m src.explain_vit_attention \
  --renders ../MultiView3D-DINOv2/data/renders \
  --emb ../MultiView3D-DINOv2/data/embeddings/embeddings.npy \
  --ids ../MultiView3D-DINOv2/data/embeddings/ids.txt \
  --out results/explain \
  --image-size 768 \
  --crop-size 768 \
  --num-show 12
```

### Guided upsampling

Add `--guided-upsampling` to produce RGB-guided, input-resolution heatmaps for
both attention rollout and gradient-weighted attention rollout:

```bash
python -m src.explain_vit_attention \
  --renders data/renders \
  --emb data/embeddings/embeddings.npy \
  --ids data/embeddings/ids.txt \
  --out results/explain \
  --image-size 768 --crop-size 768 --num-show 12 \
  --guided-upsampling --guided-radius 1 --guided-eps 0.001
```

| Option | Default | Purpose |
|--------|---------|---------|
| `--guided-upsampling` | off | Enable RGB-guided joint upsampling for both heatmaps. |
| `--guided-radius` | `1` | Local window radius in **patch cells**, not pixels; `1` uses a 3×3 patch window. Must be ≥ 1. |
| `--guided-eps` | `0.001` | Positive regularization for RGB values in [0,1]. Larger values reduce the influence of image edges. |

This uses the joint upsampling procedure from
[He et al., Guided Image Filtering](https://people.csail.mit.edu/kaiming/eccv10/index.html):
fit local linear coefficients on the patch grid, bilinearly upsample the
coefficients, and evaluate against the RGB input. Guidance and overlay backgrounds
use the exact resized/center-cropped model input, so edges remain aligned even
for non-square source images. Output heatmaps have `crop_size × crop_size` pixels.
The filter runs on the CPU and requires no additional dependencies.

Guided outputs have a settings suffix, for example
`attention_rollout_guided_r1_eps0.001.png` and
`grad_rollout_similarity_to_specimen_guided_r1_eps0.001.png`. This keeps them
alongside ordinary outputs and makes `--resume` distinguish guided settings.
Use `--no-resume` or a new `--out` directory when changing other settings such as
resolution, layers, model, or selected views. Without `--guided-upsampling`,
the existing rendering and filenames are retained.

The same options are available in `experiments.explain_finetuned` and
`experiments.compare_rollout`. Upsampling is visualization postprocessing;
fine edge structure comes from the RGB guide and does not add evidence of
pixel-level model attribution. Values are clipped to [0,1] after filtering,
without another min-max normalization.

## Safe execution (high-resolution / long runs)
High-resolution settings (e.g. `--image-size 768`) make this pass VRAM-heavy:
attention maps are materialized explicitly (fused attention is disabled so the
maps can be captured), so the peak can approach the GPU limit. On memory-limited
GPUs this may cause CUDA crashes or full system hangs. The options below make
long runs robust **without lowering the resolution**:

| Option | Default | Purpose |
|--------|---------|---------|
| `--resume` / `--no-resume` | `--resume` | Skip specimens whose two output PNGs already exist (resume after a crash). |
| `--cooldown SECONDS` | `0` | Sleep after each specimen to let the GPU cool down (thermal / power-spike relief). |
| `--vram-fraction F` | off | Cap process VRAM to fraction `F` (0–1); exceeding it raises OOM instead of hanging. |
| `--strict-sync` | off | `cuda.synchronize()` after each view to surface CUDA errors at their true location. |

Example — a manual run that keeps the 768px settings and adds the safety options:

```bash
python -m src.explain_vit_attention \
  --renders data/renders \
  --emb data/embeddings/embeddings.npy \
  --ids data/embeddings/ids.txt \
  --out results/explain \
  --image-size 768 \
  --crop-size 768 \
  --num-show 12 \
  --resume \
  --cooldown 5 \
  --vram-fraction 0.9
```

Per-view tensors are now freed and `torch.cuda.empty_cache()` runs between views,
keeping the VRAM peak near a single view's footprint; the peak (GB) is shown in
the progress bar. Out-of-memory on one view is caught and that view is skipped.

For unattended runs, use the auto-restart wrapper, which relaunches the process
(resuming where it stopped) if the GPU crashes — a corrupted CUDA context cannot
be recovered in-process, so restarting is the only reliable recovery:

```bash
bash scripts/run_safe.sh
```

To enable guided upsampling through the wrapper:

```bash
GUIDED_UPSAMPLING=1 GUIDED_RADIUS=1 GUIDED_EPS=0.001 bash scripts/run_safe.sh
```

It also exports `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce
fragmentation. To guard against power-spike blackouts, also consider capping the
GPU power limit on the host (e.g. `nvidia-smi -pl 200`).

## Outputs
For each specimen, the tool writes attention-visualization artifacts such as:
- `attention_rollout.png`
- `grad_rollout_similarity_to_specimen.png`
- Per-view overlays and summary plots

## Interpretation Notes
- Attention maps are **not** direct identification of taxonomic or morphological traits.
- These outputs are auxiliary visualizations of image-level cues associated with ViT-based embedding formation.

## Citation
```bibtex
@software{morphological_embedding_explainability,
  title  = {Morphological Embedding Explainability},
  author = {Ueya, Towa and Iba, Yasuhiro},
  year   = {2026},
  url    = {https://github.com/TowaUeya/Morphological-Embedding-Explainability},
  doi    = {10.5281/zenodo.20258440},
  note   = {Attention rollout visualization toolkit for multi-view DINOv2 representations}
}
```

## Links
* Source code: [https://github.com/TowaUeya/Morphological-Embedding-Explainability](https://github.com/TowaUeya/Morphological-Embedding-Explainability)
* Archived version: [https://doi.org/10.5281/zenodo.20258440](https://doi.org/10.5281/zenodo.20258440)

## Related Repositories

Morphological-Embedding-Explainability is the interpretability component of this ecosystem. It uses rendered views and precomputed embeddings to visualize attention-based image-level cues associated with ViT-based embedding formation.

This repository is part of a small research software ecosystem for morphology-based analysis of 3D specimen models.

- **Embedding generation**  
  **MultiView3D-DINOv2**  
  [https://github.com/TowaUeya/MultiView3D-DINOv2](https://github.com/TowaUeya/MultiView3D-DINOv2)  
  Renders multi-view images from 3D specimen models and extracts frozen DINOv2 features, producing specimen-level embeddings and rendered views for downstream analysis and visualization.

- **Embedding-space analysis**  
  **Morphological-Embedding-Space-Analyzer**  
  [https://github.com/TowaUeya/Morphological-Embedding-Space-Analyzer](https://github.com/TowaUeya/Morphological-Embedding-Space-Analyzer)  
  Performs downstream analysis of specimen-level embeddings, including retrieval evaluation, HDBSCAN-based clustering, leaf-core and residual sample extraction, embedding-space visualization, and publication-oriented figure generation.

- **Embedding explainability**  
  **Morphological-Embedding-Explainability**  
  [https://github.com/TowaUeya/Morphological-Embedding-Explainability](https://github.com/TowaUeya/Morphological-Embedding-Explainability)  
  Uses rendered multi-view images, embeddings, specimen IDs, and optional cluster information to visualize attention rollout and image-level visual cues associated with ViT-based embedding formation.
