# Morphological Embedding Explainability: Attention rollout visualization for multi-view DINOv2 representations
Explainability toolkit for visualizing attention rollout, gradient-weighted attention rollout, and visual evidence associated with ViT-based embedding formation.

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
  --renders data/renders \
  --emb data/embeddings/embeddings.npy \
  --ids data/embeddings/ids.txt \
  --clusters data/embeddings/clusters.csv \
  --out results/explain \
  --model dinov2_vits14 \
  --device auto \
  --image-size 224 \
  --crop-size 224 \
  --layers all \
  --num-show 6
```

## Outputs
For each specimen, the tool writes visual explainability artifacts such as:
- `attention_rollout.png`
- `grad_rollout_similarity_to_specimen.png`
- Per-view overlays and summary plots

## Interpretation Notes
- Attention maps are **not** direct identification of taxonomic or morphological traits.
- These outputs are auxiliary visual evidence of cues contributing to ViT-based embedding formation.

## Citation
```bibtex
@software{morphological_embedding_explainability,
  title = {Morphological Embedding Explainability},
  year = {2026},
  note = {Attention rollout visualization toolkit for multi-view DINOv2 representations}
}
```
