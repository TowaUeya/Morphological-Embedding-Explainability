# experiments/ — Attention rollout of a fine-tuned DINOv3

An optional side track to the frozen explainability tool in `src/`. Where
`src/explain_vit_attention.py` visualizes where a **frozen** DINOv3 attends, the
scripts here visualize where a **fine-tuned** DINOv3 attends — and how that
attention shifted relative to the frozen model.

## This is one half of a two-repo fine-tuning workflow

The fine-tuning capability is split across the two sister repositories so each one
keeps its own concern (feature extraction there, explainability here):

1. **Sister repo (`MultiView3D-DINOv2`) — the feature-extraction side.**
   `experiments/finetune_dinov3.py` fine-tunes the backbone and saves the trained
   delta (`finetuned_adapter.pt`) plus a `summary.json`. **Run it first** — it
   produces the `--adapter-dir` the scripts here consume. See that repo's
   `experiments/README.md`.
2. **Here (`Morphological-Embedding-Explainability`) — the rollout side.**
   `explain_finetuned.py` and `compare_rollout.py` reconstruct the fine-tuned model
   from that adapter directory and draw its attention rollout.

The two repos connect through the **adapter files on disk, not a shared Python
import**. So this repo can reload a fine-tuned model on its own, the few pieces
needed to reconstruct it (LoRA injection, the token-mean/view-mean pooling) are
kept locally in `experiments/_finetune_utils.py` — a small, deliberate duplicate of
the fine-tuner's structure. The contract between the repos is the on-disk artifact
format (`finetuned_adapter.pt` + `summary.json`), which both sides must keep in sync.

## Scope

The main tool (`../README.md`) keeps DINOv3 **frozen**. These scripts visualize a
model whose backbone was updated with labels, which is a capability demo — keep its
figures **separate** from the frozen-baseline claims (see [Interpretation notes](#interpretation-notes)).

## Requirements

Everything here uses only the base tool's dependencies (parent `requirements.txt`:
`torch`, `timm`, `numpy`, `pillow`, `matplotlib`). No `pandas` / `scikit-learn` is
needed — those are training-only and live in the sister repo. `_finetune_utils.py`
uses only `torch` / `numpy` / `pillow`.

## Data layout

- `--renders`: a directory of `{specimen_id}_viewNN.png` multi-view renders.
- `--adapter-dir`: a `finetune_dinov3 --out` directory from the sister repo. Must
  contain `finetuned_adapter.pt` + `summary.json`; `compare_rollout.py` additionally
  reads `heldout_ids.txt` (written by `finetune_dinov3.py --save-embeddings`).

Run everything **from the repo root** (`src/` is a namespace package, so `-m` keeps
imports working).

---

## 1) explain_finetuned.py — rollout of the fine-tuned model

Reconstructs the fine-tuned model from a `finetune_dinov3.py` output directory and
renders attention-rollout heatmaps for it. It takes care of the two things that make
a fine-tuned rollout correct (see [Interpretation notes](#interpretation-notes)):

1. **Model reconstruction** — loads a pristine DINOv3 and re-applies the exact
   structure recorded in `summary.json` (same LoRA rank/alpha/targets, or a no-op for
   `blocks`/`full`), then loads the `finetuned_adapter.pt` delta on top. Fails fast if
   the delta doesn't match.
2. **Reference-embedding regeneration** — the grad-rollout score is the cosine
   similarity of each view to a **reference specimen embedding**; this script recomputes
   that reference **with the fine-tuned model** (not any frozen `embeddings.npy`), so the
   figure isn't a frozen/fine-tuned hybrid.

```bash
# 1) First fine-tune in the sister repo (MultiView3D-DINOv2) to produce an adapter:
#    python -m experiments.finetune_dinov3 \
#      --renders data/renders/fish_bone --labels data/labels_bone.csv \
#      --out results/finetune_fish_bone_lora_ce --finetune lora --loss ce --image-size 448

# 2) Then, here, visualize the fine-tuned model's evidence:
python -m experiments.explain_finetuned \
  --renders data/renders/fish_bone \
  --adapter-dir /path/to/results/finetune_fish_bone_lora_ce \
  --out results/explain_finetuned_fish_bone \
  --num-show 12
#   One specimen:  --specimen_id <sid>     Final block only:  --layers last
```

### Arguments

| flag | default | meaning |
|---|---|---|
| `--renders` | (required) | dir of `{sid}_viewNN.png` |
| `--adapter-dir` | (required) | a `finetune_dinov3 --out` dir (needs `finetuned_adapter.pt` + `summary.json`) |
| `--out` | (required) | output directory |
| `--specimen_id` | all | target specimen; if omitted, every specimen under `--renders` is processed |
| `--layers` | `all` | rollout source: `all` blocks or `last` block only |
| `--guided-upsampling` | off | RGB-guided upsampling of both heatmaps to model input resolution |
| `--guided-radius` | `1` | guided-filter radius in patch cells (≥ 1) |
| `--guided-eps` | `0.001` | positive regularization for RGB guidance in [0,1] |
| `--num-show` | `6` | number of views to render per specimen |
| `--image-size` | = fine-tune | override; defaults to `summary.json` (avoids a train/eval distribution shift) |
| `--crop-size` | = fine-tune | override; defaults to `summary.json` |
| `--batch-size` | `16` | batch size for reference-embedding extraction |
| `--device` | `auto` | `auto` / `cuda` / `cpu` |

### Outputs (`--out`)

Per specimen:

- `attention_rollout.png` — rollout from the raw attention weights.
- `grad_rollout_similarity_to_specimen.png` — rollout weighted by the gradient of the
  view-to-specimen similarity score.

Plus `provenance.json`, which records the adapter, config, and resolution used.
It also records guided upsampling settings. With `--guided-upsampling`, PNG
filenames gain `_guided_r1_eps0.001` (using the selected radius/epsilon) before
`.png`; ordinary outputs are kept separately. Guidance and display use the exact
resized/center-cropped input. See [Guided upsampling](../README.md#guided-upsampling)
for an example and interpretation notes.

---

## 2) compare_rollout.py — frozen vs fine-tuned difference

For **one specimen**, computes the grad-rollout heatmap (similarity to the specimen's
own pooled embedding) with **both** the frozen and the fine-tuned model at the
fine-tune resolution, and draws the difference. Each model uses its **own** recomputed
reference embedding, so neither map is a frozen/fine-tuned hybrid. The result is a
4-row figure — view / frozen / fine-tuned / difference (fine-tuned − frozen) — where
red in the diff row means the fine-tuned model attends **more**, blue **less**.

**Held-out safety.** By default it visualizes a **held-out** specimen (read from the
adapter dir's `heldout_ids.txt`), so the attention reflects generalization rather than
memorization of a training specimen. If `--specimen_id` names a training specimen, it
warns.

```bash
python -m experiments.compare_rollout \
  --renders data/renders/fish_bone \
  --adapter-dir /path/to/results/finetune_fish_bone_lora_ce \
  --out results/compare_rollout/fish_bone --num-show 12
```

### Arguments

| flag | default | meaning |
|---|---|---|
| `--renders` | (required) | dir of `{sid}_viewNN.png` |
| `--adapter-dir` | (required) | a `finetune_dinov3 --out` dir (needs `heldout_ids.txt` for auto-selection) |
| `--out` | (required) | output directory |
| `--specimen_id` | first held-out | target specimen; default = first held-out specimen present under `--renders` |
| `--num-show` | `6` | number of views to render |
| `--layers` | `all` | rollout source: `all` blocks or `last` block only |
| `--guided-upsampling` | off | apply the same RGB-guided upsampling to both models before computing the difference |
| `--guided-radius` | `1` | guided-filter radius in patch cells (≥ 1) |
| `--guided-eps` | `0.001` | positive regularization for RGB guidance in [0,1] |
| `--image-size` | = fine-tune | override; defaults to `summary.json` |
| `--crop-size` | = fine-tune | override; defaults to `summary.json` |
| `--batch-size` | `16` | batch size for reference-embedding extraction |
| `--device` | `auto` | `auto` / `cuda` / `cpu` |

### Output (`--out`)

- `{specimen_id}_grad_rollout_diff.png` — the 4-row view / frozen / fine-tuned / diff
  figure. Each heatmap is min-max normalized to [0,1], so the diff row shows a
  **relative** shift in attention, not an absolute one.

With `--guided-upsampling`, the filename gains `_guided_r1_eps0.001` (using the
selected values). The difference and logged statistics then describe the
upsampled, clipped maps at input resolution, with no further normalization.

---

## Interpretation notes

Everything you need to read these figures honestly.

### Fine-tuning changes what attention rollout *means*

Attention rollout is a function of the model's weights — how attention is laid out
across the network. Fine-tuning moves those weights, so **applying attention rollout
to a fine-tuned model changes the meaning of the visualization itself**:

- **Frozen model** → "where a general-purpose, pre-trained ViT *generally* attends."
  This is a property of the pre-trained representation, and it is what
  `src/explain_vit_attention.py` shows.
- **Fine-tuned model** → "where the model *learned to look in order to separate these
  classes on this data*." This is task- and data-specific discriminative evidence.

These are **different objects.** Do not read a fine-tuned rollout as if it were the
frozen one, and never mix the two in a single claim.

The two figures `explain_finetuned.py` writes react to fine-tuning differently:

- `attention_rollout.png` depends only on the attention weights, so it moves directly
  as the weights move.
- `grad_rollout_similarity_to_specimen.png` also depends on the **reference embedding**
  the similarity score points at. If that reference were a frozen `embeddings.npy` while
  the model is fine-tuned, the figure would be a frozen/fine-tuned hybrid and the
  interpretation would be muddy. `explain_finetuned.py` avoids this by recomputing the
  reference with the fine-tuned model, so what you see is a coherent "fine-tuned view
  pulled toward a fine-tuned target." `compare_rollout.py` does the same for both models.

### Rollout is descriptive, not proof of importance

- The **quantitative** evidence that the fine-tuning data mattered is the held-out
  **retrieval lift** reported by the sister repo's `finetune_dinov3.py`
  (`summary.json` → `lift_knn`), **not** the rollout. These figures (really the
  fine-tuned − frozen difference) are only a **qualitative** illustration of how
  attention shifted.
- "Attends to a region" ≠ "that region is diagnostically important." With little
  labelled data the model can latch onto **shortcuts** (render/lighting quirks,
  background, orientation, silhouette, size) and still produce a clean heatmap.
- To claim an attended region is morphologically meaningful you need: (1) a real
  held-out lift, (2) agreement between the attended region and domain knowledge, and
  ideally (3) an occlusion/ablation check.
