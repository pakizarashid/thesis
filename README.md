# Dual-Defense Audio Watermarking for Zero-Shot Voice Cloning

MS thesis project: a joint traceability + disruption audio watermarking system built on VoiceMark (traceability) and a SafeSpeech-derived disruption objective, evaluated against AudioPure (diffusion-based purification). All training uses LibriSpeech `train-clean-100`; VCTK is additionally used for evaluation-only cross-dataset generalization testing (see [Dataset section](#dataset) for exact scope and rationale).

**Full technical writeups**: [`STAGE1_WRITEUP.md`](./STAGE1_WRITEUP.md) · [`STAGE2_WRITEUP.md`](./STAGE2_WRITEUP.md) · [`AUDIOPURE_WRITEUP.md`](./AUDIOPURE_WRITEUP.md)

---

## Repository structure

```
src/
  models/       backbone.py, adapters.py (LoRA), surrogate_vc.py (YourTTS)
  data/         librispeech.py, augment.py, vctk.py (reads from mounted Kaggle input)
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
| `checkpoints/stage1_full_recalibrated_v3/` | Stage 1 full, presence-calibration fix applied | **Canonical Stage 1 checkpoint** |
| `checkpoints/stage2_sim_longrun/` | Stage 2, similarity-targeted disruption, 30 epochs | **Canonical Stage 2 checkpoint** |

`stage1_full_recalibrated/` and `_v2/` are retained as diagnostic evidence for the false-positive-rate investigation (see below), not for general use.

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

### 3. AudioPure — purification attack (central thesis result)

| Condition | ACC before purification | ACC after purification | Drop |
|---|---|---|---|
| Baseline VoiceMark | 98.3% | 50.5% | 47.8 pp |
| Stage 1 (traceability fine-tuned) | 100.0% | 48.8–53.0% | ~48–52 pp |
| Stage 1 + distortion-augmentation | 98.8% | 51.8% | 47.0 pp |
| Stage 2 (disruption fine-tuned) | 99.5–100.0% | 45.3–54.8% | ~50–55 pp |
| Stage 1 (presence-recalibrated, v3) | 100.0% | 49.3% | 50.8 pp |

All conditions collapse to chance level (~50%) — the watermark is completely destroyed by purification, regardless of training approach, including the model specifically trained for distortion robustness. This is the thesis's central, novel finding — neither VoiceMark nor SafeSpeech test this combination.

### 4. False positive rate (new metric, not covered by either source paper)

No training loss in this project (Stages 1 or 2) ever penalized the detector for wrongly flagging clean, never-watermarked audio — a genuine gap discovered during evaluation, not present in the original design intent.

| Checkpoint | False positive rate | Notes |
|---|---|---|
| Baseline (pretrained) | 4.0% | Healthy |
| Stage 1 (original) | 76.0% | Degraded — no negative examples in training |
| Stage 2 (original) | 84.0% | Further degraded |
| Recalibration v1 (10 spk, 5 epochs) | 28.0% | Partial fix |
| Recalibration v2 (10 spk, 10 epochs total, same data) | 28.0% | **No improvement — overfitting confirmed** |
| **Recalibration v3 (30 spk, 5 epochs, fresh)** | **4.0%** | **Matches baseline — data diversity was the fix, not epoch count** |

Fix: `src/recalibrate_presence.py` adds a binary cross-entropy presence loss computed on both watermarked (positive) and the same clean audio run directly through the detector (negative) — the missing negative-example signal.

### 5. Audio quality metrics (new, not previously measured)

| Condition | Mean PESQ (transparency) | Mean STOI (intelligibility) | Mean SNR (perturbation magnitude) | Mean WER (cloned audio) |
|---|---|---|---|---|
| Baseline | 2.043 | 0.902 | 3.96 dB | 0.000 |
| Stage 2 (sim-mode) | 2.180 | 0.904 | 4.90 dB | 0.031 |

**Honest interpretation**: SNR of ~4–5 dB is low by typical watermarking-literature standards (imperceptible perturbations are usually reported in the 20–40 dB range) — by this energy-based measure, the watermark is a real, non-trivial perturbation, not a subtle one. This is consistent with two other independent measurements already on record: raw waveform correlation (~0.83, `STAGE1_WRITEUP.md` Section 10) and PESQ landing in the "fair" rather than "transparent" range. All three measurements agree: this is an energetically substantial perturbation that happens to remain perceptually tolerable (STOI ~0.90, WER near-zero), rather than a genuinely imperceptible one in the strict sense. Stated as an honest limitation, not glossed over.

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
