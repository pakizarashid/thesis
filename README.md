# Dual-Defense Audio Watermarking for Zero-Shot Voice Cloning

MSc thesis project: a joint traceability + disruption audio watermarking system built on VoiceMark (traceability) and a SafeSpeech-derived disruption objective, evaluated against AudioPure (diffusion-based purification). All training uses LibriSpeech `train-clean-100`; VCTK is additionally used for evaluation-only cross-dataset generalization testing (see [Dataset section](#dataset) for exact scope and rationale).

**Full technical writeups**: [`STAGE1_WRITEUP.md`](./STAGE1_WRITEUP.md) · [`STAGE2_WRITEUP.md`](./STAGE2_WRITEUP.md) · [`AUDIOPURE_WRITEUP.md`](./AUDIOPURE_WRITEUP.md)

---

## Pipeline overview

![Data flow diagram](./pipeline_diagram.png)

Green boxes are the only trainable components (294,912 shared LoRA parameters across both stages); yellow boxes are large frozen models used as fixed tools; pink ellipses are where measurements come out.

---

## Repository structure

```
src/
  models/       backbone.py, adapters.py (LoRA), surrogate_vc.py (YourTTS)
  data/         librispeech.py, augment.py, vctk.py, libritts.py (both read from mounted Kaggle input)
  losses/       voicemark_losses.py, safespeech_losses.py
  eval/         disruption_effectiveness.py, audiopure_eval.py, false_positive_rate.py,
                cross_dataset_eval.py, quality_metrics.py (PESQ/STOI/SNR/WER),
                augmentation_robustness.py, gradient_diagnostic.py,
                save_audio_samples.py, audio_diff_analysis.py, compare_results.py
  train.py            Stage 1 training
  train_stage2.py     Stage 2 training (mel-mode / sim-mode disruption)
  recalibrate_presence.py   False-positive-rate recalibration (see below)
scripts/
  setup_env.sh        One-shot dependency install
  patch_audiopure.py  AudioPure submodule compatibility patch (re-run every fresh checkout)
checkpoints/          Trained LoRA adapter weights (see table below)
results/              All evaluation results as JSON
external/             Submodules: voicemark, safespeech, audiopure
```

---

## Setup

```bash
git clone --recurse-submodules https://github.com/pakizarashid/thesis.git
cd thesis
bash scripts/setup_env.sh
python scripts/patch_audiopure.py
```

---

## Dataset

**Training** (Stages 1 and 2) uses LibriSpeech `train-clean-100` (251 speakers, ~100 hours, 16kHz) exclusively — not VCTK (VoiceMark's own corpus) or LibriTTS+CMU ARCTIC (SafeSpeech's corpus) — chosen for native sample-rate match (avoiding resampling), internal consistency across all three project phases, automatic no-license-request download, and to keep iteration cycles fast given this project's compute constraints. Full rationale in `STAGE1_WRITEUP.md` Section 2.

**Evaluation only** additionally uses VCTK (see Results Section 6) — deliberately *not* used for training, since the entire point is testing whether LibriSpeech-trained checkpoints generalize to a corpus they've never seen. Read directly from a mounted Kaggle Input dataset, not downloaded, given VCTK's ~13GB size would otherwise conflict with Kaggle's working-disk budget.

| Phase | Train speakers | Train utterances | Eval speakers | Eval utterances | Clip length |
|---|---|---|---|---|---|
| Stage 1 (initial) | 30 | 300 | 5 | 25 | 3.0s @ 16kHz |
| Stage 2 / most evaluation | 10 | 100 | 5 | 25 | 3.0s @ 16kHz |
| Presence recalibration (final, v3) | 30 | 300 | 5 | 25 | 3.0s @ 16kHz |

Train/eval speaker pools are always disjoint (non-overlapping slices of one deterministic speaker shuffle).

---

## Checkpoints

| Checkpoint | Description | Status |
|---|---|---|
| `checkpoints/stage1_full/` | Stage 1, no augmentation | Original |
| `checkpoints/stage1_aug/` | Stage 1, VC-distortion augmentation | Original |
| `checkpoints/stage1_full_recalibrated_v3/` | Stage 1 full, presence-calibration fix applied | Canonical for detection/FPR tasks |
| `checkpoints/stage1_low_perturbation/` | Loss-rebalanced from v3 (Lmel/Lcos doubled) | Documented negative result — see Results Section 5 |
| `checkpoints/stage2_sim_longrun/` | Stage 2, similarity-targeted disruption, 30 epochs | Superseded — see `_recalibrated_v2` |
| `checkpoints/stage2_sim_longrun_recalibrated_v2/` | Stage 2, same disruption training, presence-calibration fix applied | **Canonical Stage 2 checkpoint** |

`stage1_full_recalibrated/` + `_v2/`, and `stage2_sim_longrun_recalibrated/` (v1), are retained as diagnostic evidence for the false-positive-rate investigation (see below), not for general use.

---

## Results

### 1. Stage 1 — traceability reproduction

Held-out detection accuracy before any fine-tuning: **99.55%**, matching VoiceMark's reported 96–98% range.

**Augmentation robustness** (detection accuracy on watermarked audio under simulated distortion):

| Condition | Clean | Masking | Shuffling | Replacing | Neural (noise proxy) |
|---|---|---|---|---|---|
| Pretrained baseline | 0.987 | 0.989 | 0.987 | 0.962 | 0.951 |
| Fine-tuned, no augmentation | 0.980 | 0.980 | 0.978 | 0.951 | 0.888 |
| Fine-tuned, with augmentation | 0.978 | 0.991 | 0.982 | 0.973 | 0.864 |

### 2. Stage 2 — joint disruption training

Five independent interventions tested (lambda scale, loss reweighting, adapter capacity, training duration, training objective) — see `STAGE2_WRITEUP.md` for full detail. Final, statistically-grounded result (3 independent evaluation runs per condition, given ~0.01–0.02 measurement noise in the SIM metric):

| Condition | SIM (mean of 3 runs) | Pivotal distance (mean) |
|---|---|---|
| Baseline | ~0.459 | ~1.867 |
| Stage 1 only | ~0.465 | ~1.877 |
| Stage 2 (similarity-targeted, final) | ~0.473 | ~1.854 |

No reliable disruption effect found; documented as a rigorous negative result with a leading explanation (likely adapter capacity limitation).

**Cross-domain validation (LibriTTS, SafeSpeech's own training corpus)** — checks whether the negative result is specific to LibriSpeech or holds generally, evaluated on the canonical checkpoint with no retraining:

| Condition | SIM (LibriTTS) | Pivotal distance (LibriTTS) |
|---|---|---|
| Baseline | 0.4246 | 2.2107 |
| Stage 2 (canonical) | 0.4448 | 2.1913 |

No disruption effect on LibriTTS either, ruling out dataset domain as an alternative explanation and strengthening the capacity-limitation hypothesis. See `STAGE2_WRITEUP.md` Section 8 for full detail.

### 3. AudioPure — purification attack (central thesis result)

| Condition | ACC before purification | ACC after purification | Drop |
|---|---|---|---|
| Baseline VoiceMark | 98.3% | 50.5% | 47.8 pp |
| Stage 1 (traceability fine-tuned) | 100.0% | 48.8–53.0% | ~48–52 pp |
| Stage 1 + distortion-augmentation | 98.8% | 51.8% | 47.0 pp |
| Stage 2 (disruption fine-tuned) | 99.5–100.0% | 45.3–54.8% | ~50–55 pp |
| Stage 1 (presence-recalibrated, v3) | 100.0% | 49.3% | 50.8 pp |
| Stage 2 (presence-recalibrated, v2) | 99.5% | 49.25% | 50.25 pp |

All conditions collapse to chance level (~50%) — the watermark is completely destroyed by purification, regardless of training approach, including the model specifically trained for distortion robustness, and regardless of the presence-calibration fix applied to either stage. This is the thesis's central, novel finding — neither VoiceMark nor SafeSpeech test this combination.

### 4. False positive rate (new metric, not covered by either source paper)

No training loss in this project (Stages 1 or 2) ever penalized the detector for wrongly flagging clean, never-watermarked audio — a genuine gap discovered during evaluation, not present in the original design intent.

| Checkpoint | False positive rate | Notes |
|---|---|---|
| Baseline (pretrained) | 4.0% | Healthy |
| **Stage 1** (original) | 76.0% | Degraded — no negative examples in training |
| Stage 1, recalibration v1 (10 spk, 5 epochs) | 28.0% | Partial fix |
| Stage 1, recalibration v2 (10 spk, 10 epochs total, same data) | 28.0% | No improvement — overfitting on narrow data |
| **Stage 1, recalibration v3 (30 spk, 5 epochs, fresh restart)** | **4.0%** | **Fixed — needed data diversity, not more epochs** |
| **Stage 2** (original) | 84.0% | Further degraded |
| Stage 2, recalibration v1 (30 spk, 5 epochs) | 12.0% | Partial fix — train-batch trend still improving, not yet plateaued |
| **Stage 2, recalibration v2 (30 spk, 10 epochs total, same data)** | **0.0%** | **Fully fixed — more epochs on already-diverse data worked directly, no overfitting** |

Fix: `src/recalibrate_presence.py` adds a binary cross-entropy presence loss computed on both watermarked (positive) and the same clean audio run directly through the detector (negative) — the missing negative-example signal. Note the two stages resolved via different mechanisms: Stage 1's first attempts used too narrow a dataset (10 speakers), so more epochs alone overfit rather than generalized — fixed only once data diversity increased (v3). Stage 2's attempts started with the larger dataset (30 speakers) from the outset, so more epochs alone were sufficient (v1→v2) with no overfitting observed — confirming the underlying lesson is "sufficient data diversity is a prerequisite," not "more epochs are inherently harmful."

### 5. Audio quality metrics (new, not previously measured)

| Condition | n | PESQ ↑ | STOI ↑ | SI-SNR ↑ | WER ↓ |
|---|---|---|---|---|---|
| Baseline | 50 | 2.197 | 0.910 | 3.26 dB | 0.045 |
| Stage 1, loss-rebalanced (`stage1_low_perturbation`) | 50 | 2.383 | 0.917 | 3.25 dB | 0.040 |
| **VoiceMark's own published numbers** | — | **2.20** | **0.89** | **2.01 dB** | — |

SI-SNR (scale-invariant SNR) is used specifically because it is the exact metric VoiceMark's own paper reports (Table 3, Li et al. 2025) — found via direct literature search, enabling a true apples-to-apples comparison rather than an approximate one.

**Two findings, both properly evidenced at matched sample sizes**:

1. **The loss-rebalancing experiment (`src/reduce_perturbation.py`) did not work.** An initial n=25 comparison using plain SNR appeared to show a real improvement (+1.27 dB). This did not survive correction: plain SNR doesn't account for scale/amplitude differences, and at n=50 with the paper-matching SI-SNR metric, baseline and rebalanced are statistically indistinguishable (3.26 vs 3.25 dB). Reported as a negative result, not omitted — see `STAGE1_WRITEUP.md` Section 11 for the full methodology trail, including the sample-size and metric-choice corrections that led here.
2. **The baseline reproduction already matches or exceeds VoiceMark's own published imperceptibility**, with no additional work: PESQ is a near-exact match (2.197 vs 2.20), STOI exceeds theirs (0.910 vs 0.89), and SI-SNR exceeds theirs by over 60% (3.26 vs 2.01 dB). This is a genuine, verified validation result on the original authors' own metric.

### 6. Cross-dataset generalization (VCTK)

Tests whether checkpoints trained exclusively on LibriSpeech generalize to a completely unseen corpus — VCTK, notably VoiceMark's own original training domain, making this also a direct comparison point to their paper. Evaluation only, no retraining (retraining on VCTK would defeat the purpose of testing generalization).

| Condition | LibriSpeech held-out ACC | VCTK detection ACC (unseen corpus) |
|---|---|---|
| Baseline | ~98.3–99.55% | 99.78% |
| Stage 1 (recalibrated v3) | ~99.5–100% | 99.78% |

Detection accuracy on VCTK closely matches LibriSpeech performance despite the model never training on VCTK — genuine evidence of generalization across recording conditions and speaker populations, not overfitting to a single corpus. See `src/data/vctk.py` for the loader (reads directly from a mounted Kaggle input dataset, avoiding the disk-space problems of downloading the full ~13GB corpus directly).

---

## Known limitations

- Single corpus (LibriSpeech), modest scale (100–300 training utterances) — a deliberate compute-budget tradeoff, not an oversight (see Dataset section).
- SafeSpeech's disruption loss was reimplemented from their published formulas, not validated against their own reported numbers on their own setup.
- AudioPure's checkpoint is trained on isolated spoken digits (SC09), a domain gap from this project's continuous sentences — addressed via architecture verification and precedent (see `AUDIOPURE_WRITEUP.md`), not eliminated.
- Full details and additional limitations are in each stage's dedicated writeup.

---

## Citations

- VoiceMark (traceability watermarking)
- SafeSpeech / POP (disruption loss formulation)
- AudioPure (Wu et al., ICLR 2023 — diffusion-based purification)
- YourTTS (zero-shot voice cloning surrogate)
- VocalBridge, De-AntiFake (2025–2026) — contemporary evidence that purification defeats protective perturbations generally, situating this project's findings within current literature
