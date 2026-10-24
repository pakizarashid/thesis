# Pre-registration: can VoiceMark's quality losses remove the clone-aware training's audio-quality cost without undoing its watermark-recovery gain?

Written 2026-10-24, before any Route 2 arm below is trained and before the Stage 1 evaluation (notebook 24, section 10) has been read. Committed to git before the runs; the commit timestamp is the record.

## Background

- Clone-aware training (Route 2: `msg_processor` + detector LoRA trained with a YourTTS clone inside the loss) raised watermark ACC after cloning on every cloner tested, but cost PESQ: 2.387 (detector-only) -> 2.171 (rank 8), n = 100 paired.
- Code review (2026-10-24) found the Route 2 loss had no quality term at all (`Ldec(clean) + Ldec(clone)` only). The three levers tried earlier (epochs, clone weight, rank) contained no quality term either.
- Code review also found that Stage 1 training never fed augmented audio to the detector (`src/train.py` discarded it). Stage 1 is retrained with VoiceMark-faithful augmentation first (`checkpoints/stage1_vmaug`), and every Route 2 arm here starts from that checkpoint (epoch 29).

## Arms (all: init `stage1_vmaug/stage1_epoch29.pt`, LoRA rank 8, `--train_msg_processor`, 20 epochs, 2 warm-up epochs, batch 2, lr 5e-5, the same 900 training utterances)

| Arm | Clone text in training | Seed | Extra loss terms |
|---|---|---|---|
| R0 | sampled per step from the 900 training transcripts (6-14 words) | 0 | none (old objective) |
| R0s1 | sampled | 1 | none (training-noise estimate) |
| R0f | fixed sentence | 0 | none (text-protocol control) |
| RQlo | sampled | 0 | mel 0.5, cos 0.5, adv 1 |
| **RQ** (primary) | sampled | 0 | mel 2, cos 2, adv 1 (VoiceMark's weights) |
| RQhi | sampled | 0 | mel 8, cos 8, adv 1 |
| RQA | sampled | 0 | mel 2, cos 2, adv 1, aug 1, vad 1 |

The seed fixes data order (own generator), messages, clone-text order and augmentation draws. GPU kernels are not deterministic, so two runs with the same seed are not bit-identical; R0s1 measures run-to-run variation.

Known in advance: the adversarial loss was flat at 0.0199 in every diagnostic step, so `adv 1` is expected to contribute little; the quality terms are effectively mel + cos.

## Evaluation (fixed now)

- Clips: LibriSpeech eval split of the 60/15/20/5 subset, 100 clips, 20 held-out speakers, message for clip i = seed 123 + i. Secondary: VCTK, 100 clips (10/10/20/5 draw).
- Quality: PESQ (wideband, 16 kHz), STOI, SI-SNR of the watermarked audio against the clean clip, 3 s, same clips for every checkpoint (`src/eval/quality_paired_ckpts.py`).
- Survival: `src/eval/clone_eval_v2.py`, 3 s prompts. Texts:
  - **unseen** (decisive): per clip, a transcript of a different held-out speaker (6-20 words), checked to be absent from every training text pool;
  - **fixed**: "This is a test sentence for voice cloning." (continuity with earlier results, not decisive).
- Cloners: YourTTS (the training cloner, reported, never decisive), XTTS-v2 (notebook 24), F5-TTS, MaskGCT, CosyVoice (notebook 25).
- Measures per cloner: mean bitwise ACC, traced rate (>= 13/16 bits), FAR (VoiceMark's definition, 99 random candidates, ties counted as false), ECAPA SIM, WER.
- **Decisive cloners** = those whose R0 clones pass the cloner-works gate (mean ECAPA SIM > 0.30 and mean WER < 20%, unseen text, 3 s). Expected: F5-TTS, MaskGCT. A cloner that fails the gate is descriptive only. XTTS-v2 gives an **interim** verdict in notebook 24 whatever its gate result, labelled interim.

## Quantities

- dPESQ = PESQ(arm) - PESQ(R0), paired over the 100 clips.
- dACC_c = ACC_c(arm) - ACC_c(R0), dTraced_c likewise, unseen text, per cloner c.
- Noise tolerance per cloner: tau_c = max(0.02, |ACC_c(R0s1) - ACC_c(R0)|) for ACC, and max(5 points, |Traced_c(R0s1) - Traced_c(R0)|) for the traced rate.
- Recovery gain G_c = ACC_c(R0) - ACC_c(stage1_vmaug); retention = (ACC_c(arm) - ACC_c(stage1_vmaug)) / G_c (descriptive).

## Decision rule (primary arm RQ; RQA is the second registered test; Holm correction over these two for the PESQ test)

- **QUALITY GAIN**: dPESQ >= +0.10 and one-sided paired Wilcoxon p < 0.05 (Holm-adjusted).
- **SURVIVAL KEPT** on decisive cloner c: dACC_c >= -tau_c and dTraced_c >= -(traced tolerance).
- Verdict:
  - **PRESERVES**: QUALITY GAIN and SURVIVAL KEPT on every decisive cloner.
  - **QUALITY AT A SURVIVAL COST**: QUALITY GAIN, SURVIVAL KEPT fails on at least one decisive cloner (name it).
  - **NO QUALITY GAIN**: otherwise (survival still reported).
- Gate G1 (retrain sanity): R0 clean ACC >= 0.95; otherwise the run is reported as failed and no verdict is given.
- Gate G2 (something to preserve): if G_c <= tau_c on a decisive cloner, R0 shows no recovery gain there; the verdict for that cloner is written as "no recovery gain to preserve" and only the quality half is read.
- RQlo and RQhi are a sweep: reported with the same quantities, no verdict, no "best arm" claim without a confirmation run on new speakers.
- R0f vs R0 (text protocol): descriptive. Reported on unseen and fixed text.

## Stage 1 reading rules (section 10 of notebook 24; also fixed before reading)

Comparisons of `stage1_vmaug` epoch 29 with `stage1_scaleup_aug` epoch 29 (old) and pretrained VoiceMark, same clips: clean ACC, ACC under each augmentation type, PESQ/STOI/SI-SNR, YourTTS and XTTS-v2 survival (unseen and fixed text). "Improves" is used only for a paired difference larger than 0.02 ACC (XTTS sampling noise) with p < 0.05; otherwise "no detectable difference". Pretrained VoiceMark's ACC under each of our augmentation types is the check that our augmentations resemble the ones it was trained with (low ACC on one type = that type differs from theirs).

## Prompt-length check (descriptive)

Pretrained VoiceMark, stage1_vmaug, R0, RQ, RQA at 3 s, 6 s and the full utterance (capped at 20 s), fixed and unseen text, on every cloner.

## Predictions (written now)

1. RQ reaches QUALITY GAIN: about 55%.
2. RQ keeps survival on F5-TTS and MaskGCT (both near ceiling): about 70%; on XTTS-v2: about 50%.
3. Verdict for RQ: PRESERVES about 35%, QUALITY AT A SURVIVAL COST about 25%, NO QUALITY GAIN about 40%.
4. RQA has higher ACC than RQ under the augmentation types on the quality script: about 70%.
5. |ACC(R0s1) - ACC(R0)| on XTTS-v2 is at most 0.02: about 60%.
6. R0f beats R0 on the fixed sentence but not on unseen text: about 55%.
7. ACC rises from 3 s to 6 s to full on F5-TTS and MaskGCT for every checkpoint: about 70%; on XTTS-v2: about 50%.
8. stage1_vmaug beats the old Stage 1 on XTTS-v2 (unseen text) by more than 0.02: about 65%; it stays within 0.01 of pretrained VoiceMark under augmentation: about 50%.

## Scope and limits

Zero-shot cloning only; 100 clips from 20 speakers per dataset (speaker-level intervals are reported and will be wide); one training corpus (900 LibriSpeech utterances); one seed replicate (R0s1) only; GPU non-determinism; VoiceMark's own training code is not public, so the augmentation follows the paper's text. No perceptual claim is made from PESQ.
