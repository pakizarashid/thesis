# Dual-Defense Audio Watermarking for Zero-Shot Voice Cloning

MSc thesis project: a joint traceability + disruption audio watermarking system built on VoiceMark (traceability) and a SafeSpeech-derived disruption objective, evaluated against AudioPure (diffusion-based purification). All training uses LibriSpeech `train-clean-100`; VCTK and LibriTTS are additionally used for evaluation-only cross-dataset generalization testing (see [Dataset section](#dataset) for exact scope and rationale).

**Headline results:** (1) six independent attempts to train the disruption objective into shared embedder weights all converged to a negative result — diagnosed as a mechanism mismatch, not a tuning failure; (2) switching to direct waveform-space PGD perturbation (`disruption_pgd.py`), matching SafeSpeech's own real mechanism, fixes this decisively and is confirmed at n=100 across three corpora (LibriSpeech, VCTK, LibriTTS) at one fixed, imperceptibility-calibrated operating point — this is the thesis's central positive contribution; (3) a separate, dedicated fine-tuning effort to make the detector survive AudioPure purification (`train_stage3_audiopure_robust.py`) was designed, run across three training configurations, and independently verified twice — concluded as a genuine negative result that localizes the vulnerability to the watermark's underlying representation, not the detector head.

**Full technical writeups**: [`STAGE1_WRITEUP.md`](./STAGE1_WRITEUP.md) · [`STAGE2_WRITEUP.md`](./STAGE2_WRITEUP.md) · [`AUDIOPURE_WRITEUP.md`](./AUDIOPURE_WRITEUP.md)

> **Note on the writeups above:** they predate the PGD hybrid (Step 8) and the Stage 3 AudioPure-robust fine-tuning effort (Step 9) described below. For those two, the Results sections in *this* README (3 and 5) are the authoritative, up-to-date account until the standalone writeups are refreshed to match.

---

## Pipeline overview

![Data flow diagram](./pipeline_diagram.png)

Green boxes are the only trainable components; yellow boxes are large frozen models used as fixed tools; pink ellipses are where measurements come out.

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

**Training** (Stages 1, 2, and 3) uses LibriSpeech `train-clean-100` (251 speakers, ~100 hours, 16kHz) exclusively — not VCTK (VoiceMark's own corpus) or LibriTTS+CMU ARCTIC (SafeSpeech's corpus) — chosen for native sample-rate match (avoiding resampling), internal consistency across all three project phases, automatic no-license-request download, and to keep iteration cycles fast given this project's compute constraints. Full rationale in `STAGE1_WRITEUP.md` Section 2.

**Evaluation only** additionally uses VCTK and LibriTTS (see Results Section 3 and Section 8) — deliberately *not* used for training, since the entire point is testing whether LibriSpeech-trained/calibrated results generalize to corpora they've never seen. Both are read directly from a mounted Kaggle Input dataset, not downloaded, given their size (VCTK ~13GB) would otherwise conflict with Kaggle's working-disk budget.

| Phase | Train speakers | Train utterances | Eval speakers | Eval utterances | Clip length |
|---|---|---|---|---|---|
| Stage 1 (initial) | 30 | 300 | 5 | 25 | 3.0s @ 16kHz |
| Stage 2 / most evaluation | 10 | 100 | 5 | 25 | 3.0s @ 16kHz |
| Presence recalibration (final, v3) | 30 | 300 | 5 | 25 | 3.0s @ 16kHz |
| PGD hybrid (Step 8), final operating point | — (no training) | — | 10 | 100 (LibriSpeech, VCTK), 98 (LibriTTS) | 3.0s @ 16kHz |
| Stage 3 AudioPure-robust fine-tuning (Step 9) | 10 | 100 | 5 | 25 (independent verification) | 3.0s @ 16kHz |

Train/eval speaker pools are always disjoint (non-overlapping slices of one deterministic speaker shuffle).

---

## Checkpoints

| Checkpoint | Description | Status |
|---|---|---|
| `checkpoints/stage1_full/` | Stage 1, no augmentation | Original |
| `checkpoints/stage1_aug/` | Stage 1, VC-distortion augmentation | Original |
| `checkpoints/stage1_full_recalibrated_v3/` | Stage 1 full, presence-calibration fix applied | **Canonical — everything below is built on this** |
| `checkpoints/stage1_low_perturbation/` | Loss-rebalanced from v3 (Lmel/Lcos doubled) | Documented negative result — see Results Section 6 |
| `checkpoints/stage2_sim_longrun/` | Stage 2, similarity-targeted disruption, 30 epochs | Superseded — see `_recalibrated_v2` |
| `checkpoints/stage2_sim_longrun_recalibrated_v2/` | Stage 2, same disruption training, presence-calibration fix applied | Negative result, superseded by the PGD hybrid — see Results Section 2 |
| `checkpoints/stage2_capacity_ffn/` | Stage 2, 4x LoRA capacity onto `msg_processor` FFN layers | **Lost** (Kaggle disk reset, never committed) — sixth negative result already fully confirmed before the loss (SIM 0.4602, statistically indistinguishable from baseline); not worth retraining, superseded by the PGD hybrid |
| `checkpoints/stage3_audiopure_robust/` | Stage 3, Run 1: 5 epochs, lr=5e-5 | Negative result, independently verified (ACC after = 0.4775) — see Results Section 5 |
| `checkpoints/stage3_audiopure_robust_v2/` | Stage 3, Run 2: 15-epoch attempt, lr=2e-4 — lost mid-training at epoch 13/step 700 (session disconnect); epochs 0-6 survived | Superseded by `_v2_cont` |
| `checkpoints/stage3_audiopure_robust_v2_cont/` | Stage 3, Run 3: resumed from Run 2's `stage3_epoch6.pt`, completed the remaining 8 epochs | Negative result, independently verified (ACC after = 0.5000 exactly) — see Results Section 5 |

**Final recommended inference pipeline (as of this writing):** `stage1_full_recalibrated_v3` (traceability + presence-calibration) with `disruption_pgd.py` applied at inference time at `epsilon=0.002, lambda_wm=1.0` (no additional checkpoint required — PGD needs no trained weights). None of the Stage 2 or Stage 3 checkpoints are part of this final pipeline; they are retained as evidence for the negative results they document.

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

### 2. Stage 2 — joint disruption training via shared embedder weights (six converging negative results)

Six independent interventions tested (lambda scale, loss reweighting, LoRA capacity at 4x attention, training duration, training objective mel-vs-sim, and 4x FFN capacity) — see `STAGE2_WRITEUP.md` for full detail on the first five. Final, statistically-grounded result for the sim-mode variant (3 independent evaluation runs, given ~0.01–0.02 measurement noise in the SIM metric):

| Condition | SIM (mean of 3 runs) | Pivotal distance (mean) |
|---|---|---|
| Baseline | ~0.459 | ~1.867 |
| Stage 1 only | ~0.465 | ~1.877 |
| Stage 2 (similarity-targeted, final) | ~0.473 | ~1.854 |

The sixth and final intervention, 4x LoRA capacity onto `msg_processor`'s feedforward layers (786,432 params, vs. the original 294,912 attention-only), was tested specifically to rule out capacity as the bottleneck: a gradient diagnostic confirmed the new capacity receives 51.2% of the disruption gradient (a live, non-dead branch) before committing to a full 30-epoch run. Result across 3 evaluation runs: SIM mean 0.4602 (spread 0.0171) — statistically indistinguishable from every other embedder-weight-training lever, which all plateaued at ~0.45–0.47.

No reliable disruption effect was found from any of the six interventions; documented as a rigorous negative result. The unifying explanation, confirmed directly by Step 8 below: this is a genuine **mechanism mismatch**, not a tuning artifact. All six levers train a small number of *shared* weights via Adam to produce one embedder that must generalize across every utterance at once — categorically less optimization freedom than SafeSpeech's own real mechanism (per-utterance PGD directly in waveform space, no shared weights, no generalization requirement).

**Cross-domain validation (LibriTTS, SafeSpeech's own training corpus)** — checks whether the negative result is specific to LibriSpeech or holds generally, evaluated on the canonical Stage 2 checkpoint with no retraining:

| Condition | SIM (LibriTTS) | Pivotal distance (LibriTTS) |
|---|---|---|
| Baseline | 0.4246 | 2.2107 |
| Stage 2 (canonical) | 0.4448 | 2.1913 |

No disruption effect on LibriTTS either, ruling out dataset domain as an alternative explanation and strengthening the mechanism-mismatch diagnosis. See `STAGE2_WRITEUP.md` Section 8 for full detail.

### 3. The PGD hybrid — direct waveform-space perturbation (headline positive result)

`disruption_pgd.py` tests the mechanism-mismatch diagnosis directly: after the frozen, LoRA-trained embedder produces `recon_wm`, it adds a small epsilon-ball-bounded additive perturbation `delta`, solved fresh per utterance via iterative sign-gradient PGD against the disruption objective. No weights are trained or saved — the backbone stays fully frozen (0 trainable params); the entire disruption signal lives in `delta`.

**Mechanism confirmed** (sim-only, `lambda_wm=0`, epsilon=0.01): SIM 0.4578 → 0.1963 — categorically larger than any of the six embedder-training levers above. Stacking PGD on top of the best embedder-trained checkpoint gave statistically identical numbers to PGD alone on the plain Stage 1 checkpoint — embedder training is fully subsumed and contributes nothing once PGD is applied.

**Watermark-preservation fix.** Unconstrained sim-only PGD dragged detection ACC from ~99.5% down to ~59% (near chance) — there was no incentive not to disrupt the exact signal the detector reads. Fixed by adding `compute_ldec` (the real watermark cross-entropy loss) on the perturbed audio, weighted by `--lambda_wm`, into the PGD objective. `lambda_wm=1.0` reaches the ACC ceiling (98.6%) with disruption strength within noise of the unconstrained case, and was adopted as final.

**Epsilon calibration — full trade-off curve (LibriSpeech, all at `lambda_wm=1.0`):**

| epsilon | n | SIM before → after | Δ SIM | ACC drop | PESQ | STOI |
|---|---|---|---|---|---|---|
| 0.001 | 25 | 0.4661 → 0.2884 | +0.1777 | +0.0048 (improved) | 1.888 | 0.896 |
| 0.002 | 25 | 0.4539 → 0.2507 | +0.2032 | 0.0000 | 1.747 | 0.887 |
| 0.003 | 25 | 0.4485 → 0.2308 | +0.2176 | −0.0024 | 1.645 | 0.883 |
| 0.005 | 25 | 0.4610 → 0.2204 | +0.2405 | −0.0024 | 1.502 | 0.872 |
| 0.007 | 25 | 0.4567 → 0.2199 | +0.2368 | −0.0048 | 1.411 | 0.855 |
| 0.010 | 25 | 0.4675 → 0.2214 | +0.2461 | −0.0096 | 1.322 | 0.838 |
| 0.010 | 100 | 0.4569 → 0.1984 | +0.2584 | −0.0094 | — | — |
| **0.002 (FINAL)** | **100** | **0.4544 → 0.2234** | **+0.2309** | **+0.0019 (improved)** | 1.747 (n=4 spot-check) | 0.887 (n=4 spot-check) |

Disruption strength is essentially flat from epsilon=0.003 to 0.01 (Δ SIM 0.218–0.246, within the ~0.02–0.03 run-to-run noise floor from YourTTS's own unseeded decoder stochasticity — see caveat below), while PESQ/STOI degrade steadily as epsilon grows. Below epsilon=0.003, disruption genuinely weakens (Δ SIM drops to 0.178 at epsilon=0.001) rather than quality continuing to improve for free — a real floor.

**FINAL OPERATING POINT — confirmed at n=100 (LibriSpeech): epsilon=0.002, lambda_wm=1.0.** Δ SIM=+0.2309 (matches the n=25 estimate of +0.2032 within run-to-run noise), detection ACC actually improved slightly (0.9969 → 0.9988), PESQ 1.747 / STOI 0.887 (n=4 spot-check — STOI essentially matches the watermark-only baseline). This is a real, honestly-quantified trade-off — PESQ 1.75 is not full transparency versus VoiceMark's own 2.20 — reported as such, not as a free win. Traceability, anti-cloning disruption, and imperceptibility are all measured together at n=100, in one fixed configuration — something none of the six embedder-training attempts achieved even one part of simultaneously.

**Cross-corpus generalization (confirmed at n=100/n=98) — same operating point, no per-dataset retuning:**

| Corpus | n | SIM before → after | Δ SIM | ACC before → after |
|---|---|---|---|---|
| LibriSpeech | 100 | 0.4544 → 0.2234 | +0.2309 | 0.9969 → 0.9988 |
| VCTK | 100 | 0.4688 → 0.2099 | **+0.2590** | 0.9944 → 1.0000 |
| LibriTTS | 98† | 0.4435 → 0.2420 | +0.2015 | 0.9943 → 0.9981 |

† LibriTTS returned 98 utterances, not 100, at `--n_speakers 30 --n_eval_speakers 10 --eval_utterances_per_speaker 10` — at least one sampled eval speaker had fewer than 10 usable utterances. Reported as n=98 rather than rounded up.

The operating point transfers cleanly with **no retuning** — VCTK's disruption is the strongest of the three (+0.2590), and both VCTK and LibriTTS show a small ACC *improvement*, not a drop, matching the LibriSpeech pattern. This rules out the concern that epsilon=0.002/lambda_wm=1.0 was an artifact of LibriSpeech's specific acoustic conditions — it holds across VCTK (studio-recorded, multi-accent) and LibriTTS (24kHz-native audiobook speech, downsampled).

**Caveat on `sim_before` varying across runs:** YourTTS's own VITS decoder injects unseeded Gaussian noise at inference, so absolute SIM values have run-to-run variance (~0.02–0.03) even for identical inputs — read within-run before→after deltas, not cross-run absolute comparisons.

### 4. AudioPure — purification attack

| Condition | ACC before purification | ACC after purification | Drop |
|---|---|---|---|
| Baseline VoiceMark | 98.3% | 50.5% | 47.8 pp |
| Stage 1 (traceability fine-tuned) | 100.0% | 48.8–53.0% | ~48–52 pp |
| Stage 1 + distortion-augmentation | 98.8% | 51.8% | 47.0 pp |
| Stage 2 (disruption fine-tuned) | 99.5–100.0% | 45.3–54.8% | ~50–55 pp |
| Stage 1 (presence-recalibrated, v3) | 100.0% | 49.3% | 50.8 pp |
| Stage 2 (presence-recalibrated, v2) | 99.5% | 49.25% | 50.25 pp |

All conditions collapse to chance level (~50%) — the watermark is completely destroyed by purification, regardless of training approach, including the model specifically trained for distortion robustness, and regardless of the presence-calibration fix applied to either stage.

**Accuracy note:** the PGD-hybrid-protected audio (Section 3) was never separately re-run through `audiopure_eval.py`. The expectation that it collapses the same way is a reasoned inference from how the two mechanisms interact — PGD's `delta` is a small additive waveform perturbation layered on top of the same `recon_wm` signal chain every row in the table above shares, and AudioPure's diffusion-based purification operates on the received waveform regardless of how it was produced — not a confirmed additional experiment. This is listed explicitly under Known Limitations below rather than stated as measured fact.

### 5. Stage 3 — AudioPure-robust detector fine-tuning (dedicated intervention, concluded negative)

Built and run as the direct, executed response to the concern that this project only combines two existing methods — a targeted attempt to close the exact gap Section 4 documents, not just a proposal.

**Mechanism.** `train_stage3_audiopure_robust.py` freezes `msg_processor` completely — including its already-trained LoRA delta, so `recon_wm` generation stays byte-identical to what every PGD/PESQ/STOI number in Section 3 was calibrated against — and continues fine-tuning ONLY `detector`'s existing LoRA adapter (196,608 trainable params). AudioPure's DiffWave denoiser is applied as a forward-only training-time augmentation (`--purify_prob`, default 0.7); the remaining batches stay clean so clean-condition accuracy is tracked, not assumed to hold.

**Diagnostic (1 batch, 1 step) passed:** gradient reached the detector's LoRA params on both the purified path (grad_norm=16.4) and the clean path (grad_norm=7.5); purified-path acc=0.4688 (chance, reproducing the Section 4 collapse exactly) before any training.

Run across three training configurations:

| Run | Config | Outcome |
|---|---|---|
| Run 1 | 5 epochs, lr=5e-5 (Stage 1/2's default) | Completed. Training-log purified acc across 5 checkpoints: mean 0.5238 (range 0.494–0.544). Clean acc held 0.98–1.00. |
| Run 2 | 15 epochs, lr=2e-4 (4x higher — to rule out "just needs a harder push" before concluding a floor) | Reached epoch 13/step 700 with mean purified acc 0.4875 across 14 checkpoints, then the Kaggle session disconnected mid-training and the checkpoint directory was lost before independent verification. |
| Run 3 | Resumed from Run 2's surviving `stage3_epoch6.pt`, completed the remaining 8 epochs to reach the full originally-planned 15-epoch, lr=2e-4 target | Completed. Checkpoints pushed to git after every epoch this time, so the final checkpoint survived. |

**Independently verified twice**, via the project's own `audiopure_eval.py` script (not just internal training logs):

| Run | Config | n | ACC before | ACC after |
|---|---|---|---|---|
| Run 1 | 5 epochs, lr=5e-5 | 25 | 1.0000 | 0.4775 |
| Run 3 | 15 epochs, lr=2e-4 | 25 | 1.0000 | **0.5000** |

Run 3's result is exactly the theoretical chance value for this bitwise metric. Combined across both independently-verified evaluations (800 bits total): mean ACC after purification = 0.4888 — statistically indistinguishable from chance. Clean-condition accuracy stayed at 1.0000 in both `audiopure_eval.py` runs, ruling out "the detector broke" as an explanation.

**Conclusion.** Detector-only LoRA fine-tuning, tested across three training runs spanning a 4x learning-rate range and the full originally-planned 15-epoch budget at the higher rate, does not recover watermark-bit detection beyond chance after AudioPure purification. This is a genuine, executed, twice-independently-verified negative result. It sharpens the thesis's central finding: the vulnerability to AudioPure purification is intrinsic to the SpeechTokenizer-based embedding representation `recon_wm` carries, not a fixable deficiency in the detector head that more training or a stronger learning rate can patch. A real fix would need to touch the representation itself (e.g. `msg_processor`'s embedding strategy, or an architecturally different detection mechanism), not just adapt the existing detector to harder inputs.

A mid-training data loss during Run 2 (Kaggle session disconnect) and the weight-only resume that produced Run 3 are documented as a process note in the methodology, not hidden — both independently-verified configurations (Run 1 and Run 3) agree with each other and with chance regardless.

### 6. False positive rate (new metric, not covered by either source paper)

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

Fix: `src/recalibrate_presence.py` adds a binary cross-entropy presence loss computed on both watermarked (positive) and the same clean audio run directly through the detector (negative) — the missing negative-example signal. Note the two stages resolved via different mechanisms: Stage 1's first attempts used too narrow a dataset (10 speakers), so more epochs alone overfit rather than generalized — fixed only once data diversity increased (v3). Stage 2's attempts started with the larger dataset (30 speakers) from the outset, so more epochs alone were sufficient (v1→v2) with no overfitting observed.

### 7. Audio quality metrics (watermark-only, not previously measured)

| Condition | n | PESQ ↑ | STOI ↑ | SI-SNR ↑ | WER ↓ |
|---|---|---|---|---|---|
| Baseline | 50 | 2.197 | 0.910 | 3.26 dB | 0.045 |
| Stage 1, loss-rebalanced (`stage1_low_perturbation`) | 50 | 2.383 | 0.917 | 3.25 dB | 0.040 |
| **VoiceMark's own published numbers** | — | **2.20** | **0.89** | **2.01 dB** | — |

SI-SNR is used specifically because it is the exact metric VoiceMark's own paper reports (Table 3, Li et al. 2025), enabling a true apples-to-apples comparison.

**Two findings, both properly evidenced at matched sample sizes**:

1. **The loss-rebalancing experiment (`src/reduce_perturbation.py`) did not work.** An initial n=25 comparison using plain SNR appeared to show a real improvement (+1.27 dB). This did not survive correction: plain SNR doesn't account for scale/amplitude differences, and at n=50 with the paper-matching SI-SNR metric, baseline and rebalanced are statistically indistinguishable (3.26 vs 3.25 dB). Reported as a negative result — see `STAGE1_WRITEUP.md` Section 11.
2. **The baseline reproduction already matches or exceeds VoiceMark's own published imperceptibility**, with no additional work: PESQ is a near-exact match (2.197 vs 2.20), STOI exceeds theirs (0.910 vs 0.89), and SI-SNR exceeds theirs by over 60% (3.26 vs 2.01 dB).

**PGD-protected condition (n=4 spot-check, at the final operating point epsilon=0.002/lambda_wm=1.0):** PESQ 1.747, STOI 0.887 — see Section 3 for the full epsilon-vs-quality trade-off curve. This spot-check is at a much smaller n than the watermark-only rows above and is not directly comparable to them; a full n≥50 PGD quality pass is listed under Known Limitations.

### 8. Cross-dataset generalization (VCTK) — detection accuracy

Tests whether checkpoints trained exclusively on LibriSpeech generalize to a completely unseen corpus — VCTK, notably VoiceMark's own original training domain. Evaluation only, no retraining.

| Condition | LibriSpeech held-out ACC | VCTK detection ACC (unseen corpus) |
|---|---|---|
| Baseline | ~98.3–99.55% | 99.78% |
| Stage 1 (recalibrated v3) | ~99.5–100% | 99.78% |

Detection accuracy on VCTK closely matches LibriSpeech performance despite the model never training on VCTK. See `src/data/vctk.py` for the loader. (See Section 3 for the separate, later cross-corpus generalization test of the PGD disruption effect itself, on both VCTK and LibriTTS.)

---

## Known limitations

- Single training corpus (LibriSpeech), modest scale (100–300 training utterances) — a deliberate compute-budget tradeoff, not an oversight (see Dataset section).
- SafeSpeech's disruption loss was reimplemented from their published formulas, not validated against their own reported numbers on their own setup.
- AudioPure's checkpoint is trained on isolated spoken digits (SC09), a domain gap from this project's continuous sentences — addressed via architecture verification and precedent (see `AUDIOPURE_WRITEUP.md`), not eliminated.
- The PGD-hybrid-protected audio (Section 3) has not been separately re-run through `audiopure_eval.py`; the claim that it also collapses under AudioPure purification (Section 4) is a reasoned inference from the shared signal chain, not a directly confirmed measurement.
- The PGD hybrid's audio-quality numbers at the final operating point (PESQ 1.747 / STOI 0.887) are a n=4 spot-check, not the n=50 sample size used for the watermark-only baseline in Section 7.
- Stage 3's Run 3 (the final, chance-exact independently-verified result) is a weight-only resume from Run 2's surviving checkpoint — Adam's optimizer momentum was not preserved across the resume, only the LoRA weights, so Run 3 is not a bit-perfect continuation of Run 2, though the training objective and total epoch count match what was originally planned.
- Stage 3's independent verification uses n=25 per run (n=50 combined across the two verified runs); this size was chosen to match Section 1/6's established evaluation protocol and is corroborated by internal training-log consistency across many more validation checks per run, but is smaller than Section 3's n=100 PGD evaluations.
- Full details and additional limitations are in each stage's dedicated writeup.

---

## Citations

- VoiceMark (traceability watermarking)
- SafeSpeech / POP (disruption loss formulation)
- AudioPure (Wu et al., ICLR 2023 — diffusion-based purification)
- YourTTS (zero-shot voice cloning surrogate)
- VocalBridge, De-AntiFake (2025–2026) — contemporary evidence that purification defeats protective perturbations generally, situating this project's findings within current literature
