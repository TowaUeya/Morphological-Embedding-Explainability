# Morphological Embedding Explainability: Attention rollout visualization for multi-view DINOv2 representations
Explainability toolkit for visualizing attention rollout, gradient-weighted attention rollout, and image-level cues associated with ViT-based embedding formation.

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
  --image-size 518 \
  --crop-size 518 \
  --layers all \
  --num-show 12
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
  author = {Ueya, Towa and Iba, Yasuhiro},
  year   = {2026},
  url    = {https://github.com/TowaUeya/Morphological-Embedding-Explainability},
  doi    = {10.5281/zenodo.20258440},
  note = {Attention rollout visualization toolkit for multi-view DINOv2 representations}
}
```

## Links
* Source code: [https://github.com/TowaUeya/Morphological-Embedding-Explainability](https://github.com/TowaUeya/Morphological-Embedding-Explainability)
* Archived version: [https://doi.org/10.5281/zenodo.20258440](https://doi.org/10.5281/zenodo.20258440)

## Related Repositories

Morphological-Embedding-Explainability is the interpretability component of a small research software ecosystem for morphology-based analysis of 3D specimen models. It complements the embedding-generation and embedding-space analysis repositories by visualizing image-level evidence associated with ViT-based embedding formation.

- **Embedding generation**  
  **MultiView3D-DINOv2**  
  [https://github.com/TowaUeya/MultiView3D-DINOv2](https://github.com/TowaUeya/MultiView3D-DINOv2)  
  Renders multi-view images from 3D specimen models and extracts frozen DINOv2 features, producing specimen-level embeddings and rendered views that can be used for explainability visualization.

- **Embedding-space analysis**  
  **Morphological-Embedding-Space-Analyzer**  
  [https://github.com/TowaUeya/Morphological-Embedding-Space-Analyzer](https://github.com/TowaUeya/Morphological-Embedding-Space-Analyzer)  
  Performs downstream analysis of specimen-level embeddings, including nearest-neighbor retrieval, HDBSCAN-based auxiliary structure extraction, leaf-core and residual analysis, visualization, and publication-oriented figure generation.

- **Embedding explainability**  
  **Morphological-Embedding-Explainability**  
  [https://github.com/TowaUeya/Morphological-Embedding-Explainability](https://github.com/TowaUeya/Morphological-Embedding-Explainability)  
  Uses rendered multi-view images, embeddings, specimen IDs, and optional cluster information to visualize attention rollout and related evidence, helping interpret which image-level cues contribute to ViT-based embedding formation.
