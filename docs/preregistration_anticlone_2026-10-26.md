# Pre-registration: anti-cloning perturbation on the clone-aware watermark (notebook 26)

Written 2026-10-26, before any run. Code: `src/eval/protect_prompts_v2.py` (protected prompts), `src/eval/clone_eval_v2.py --prompt_dir --resemblyzer` (cloning and scoring), `src/eval/summarise_anticlone.py` (every number and verdict below). Nothing is retuned after results are seen.

## Question
Does the registered perturbation Q2 still protect when it sits on the new clone-aware watermark, does that watermark still trace the clones of protected audio, and is the clone-aware watermark better than official VoiceMark when both carry the same protection?

## Design
- **Clips.** The clone_eval_v2 LibriSpeech split: 100 held-out clips, 20 speakers x 5, 3 s centre-crop prompts, message 123 + i. Same clips, crops and messages as every result in the combined summary.
- **Watermarks (3).** Official VoiceMark (`pretrained`), R0 and RQA (`route2_vmaug_*/route2_epoch19.pt`).
- **Arms (3 per watermark).**
  - `wm`: watermark only.
  - `rs`: random-sign noise at Q2's per-sample bound (Q2's own PGD start, signed). Same bound and about the same energy as Q2, no optimisation.
  - `q2`: Q2 exactly as registered (white-box F5-TTS rollout, three verifiers ECAPA / WavLM-SV / x-vector in the loss, peak 0.015, envelope alpha 1, floor 0.003, 40 steps, step 0.0015, dis:wm ratio 1.732, lambda_wm 1, F5 K = 6, CFG 2, PGD sentence = the fixed sentence, prompt transcript = faster-whisper of the clean prompt). The watermark term uses each checkpoint's own detector.
  - All arms are saved as 16-bit PCM FLAC. Source ACC and quality are measured on the saved files.
- **Cloners.** F5-TTS (the white-box target), MaskGCT and CosyVoice (paper cloners, black-box), XTTS-v2 and YourTTS (black-box, descriptive). One clone per clip and text, with fresh sampling seeds.
- **Texts.** `unseen` (another speaker's transcript, as in the combined summary) is decisive. `fixed` (the PGD sentence) is run on F5-TTS only, as an upper bound.
- **Scores.**
  - Held-out speaker similarity: Resemblyzer GE2E, clone vs clean full utterance. It is in no loss.
  - ECAPA similarity is reported but is in Q2's loss, so it is not evidence of protection.
  - Also reported: watermark ACC, traced (>= 13/16), FAR, WER (faster-whisper), and prompt PESQ / STOI / SI-SNR against the clean prompt (bars 1.70 / 0.85 / 0).
- **Statistics.**
  - Paired per clip, checked by utterance ID.
  - Speaker-level bootstrap CIs (10,000 resamples, seed 20261026).
  - Wilcoxon tests: one-sided for protection, two-sided for comparisons.
  - Exact McNemar test on traced.
  - Holm correction over the decisive cloners for H3.

## Gate
A cloner is decisive if it is F5-TTS, MaskGCT or CosyVoice and its RQA `wm` clones pass the cloner-works gate (ECAPA > 0.30 and WER < 20%, unseen text). The other rows are descriptive.

## Hypotheses and decision rules (per decisive cloner, unseen text)
- **H1 protection.** dHELD = HELD(RQA-wm) - HELD(RQA-q2).
  - **PROTECTS:** one-sided p < 0.05 and speaker-CI lower end > 0. Otherwise **NOT SHOWN**.
  - The secondary label "beyond random noise" uses the same test on HELD(RQA-rs) - HELD(RQA-q2).
- **H2 traceability under protection.**
  - **SURVIVES:** RQA-q2 traced >= 0.80 and ACC >= 0.90. Otherwise **DOES NOT SURVIVE**.
  - The ACC drop and the traced change vs RQA-wm are reported with their tests.
- **H3 clone-aware vs official under protection.** ACC(RQA-q2) - ACC(official-q2), two-sided Wilcoxon with Holm correction.
  - **BETTER / WORSE** if Holm p < 0.05. Otherwise **NO DIFFERENCE**.
  - Traced is compared with McNemar. The same test is reported for R0.
- **H4 quality (reported, no verdict).** Prompt PESQ / STOI / SI-SNR per arm against the bars, and PESQ(RQA) - PESQ(official) per arm.
- **H5 (descriptive).** dHELD(RQA) - dHELD(official), two-sided. This asks whether a clone-aware watermark makes the perturbation weaker or stronger.
- **Sensitivity.** H1 on clips 40-99 only. Clips 0-39 were Q2's development clips in earlier work.

## Predictions (before the run)
1. H1 PROTECTS on F5-TTS: about 85%; dHELD on unseen text +0.02 to +0.05. MaskGCT: about 70% (+0.015 to +0.035). CosyVoice: about 50% (no earlier Q2 data).
2. q2 beyond random noise on F5-TTS: about 75%. Earlier work found the white-box loss beats flat noise at equal budget.
3. H2 SURVIVES on F5-TTS about 85% (traced 0.85-0.95), MaskGCT about 65%, CosyVoice about 60%.
4. H3 BETTER on at least one decisive cloner: about 60%. Under protection the ACC gap should be similar to or larger than the 3 s gap without protection (+0.03 to +0.04).
5. Prompt PESQ of RQA-q2: 1.65-1.85. RQA minus official PESQ under q2: +0.03 to +0.12.
6. XTTS-v2 and YourTTS: the watermark is mostly lost (traced < 0.40), with a small similarity drop. Descriptive only.

## Consequences
- **PROTECTS + SURVIVES on the decisive cloners.** The thesis can state that the two defences compose on the clone-aware watermark at this budget. The supervisor's priority order (similarity drop with acceptable quality first, traceability second) is reported against the quality bars, which Q2 met only marginally in earlier runs.
- **NOT SHOWN.** The clone-aware watermark changes the prompt enough that the registered Q2 no longer transfers. This is reported as a negative result. No retuning in this experiment.
- **DOES NOT SURVIVE.** Composition costs traceability on that cloner. The table states the size of the cost.
- H3 decides whether "clone-aware beats official VoiceMark" also holds under protection.

## Deviations from earlier Q2 runs (stated in advance)
- New watermark checkpoints (official / R0 / RQA instead of the old rank-8 Route 2).
- The control is random-sign noise at Q2's bound, not A0 (the flat 0.002 YourTTS-surrogate PGD). Numbers are therefore not comparable with earlier dHELD values against A0.
- 100 clips including Q2's development clips (sensitivity row on 40-99).
- Held-out similarity is computed against the clean full utterance with Resemblyzer's standard preprocessing.
- CosyVoice runs with transformers 4.51.3 and MaskGCT with transformers 4.57.6 and compatibility patches, as in the combined summary.

## Limits
- One perturbation setting.
- One clone per clip and text, so sampling noise is about 0.015 on a mean held-out drop.
- 20 speakers.
- LibriSpeech only.
- Resemblyzer is a relative measure: values near 0.6 do not mean "same person".
- No perceptual claim.
- Q2's PGD sentence is fixed, so unseen-text protection was 60-70% of fixed-text protection in earlier work.
