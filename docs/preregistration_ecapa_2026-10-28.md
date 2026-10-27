# Pre-registration — ECAPA decision threshold and accept rates for both verifiers

Written 2026-10-28, before any ECAPA calibration score was computed.
Notebook: `29_ecapa_threshold_accept_rates.ipynb`. Code: `src/eval/sv_threshold_calibration.py --verifier ecapa`,
`src/eval/accept_rates.py`.

## Motivation

The evaluation reports ECAPA similarity (0.475 → 0.105 on the protected F5-TTS clone) next to Resemblyzer
similarity (0.800 → 0.735). Those two numbers are cosine similarities in different embedding spaces produced by
models trained with different objectives on different data. They are not comparable with each other, and no
statement should rest on comparing them.

What is comparable across verifiers is the **decision**: the share of clones at or above that verifier's own
threshold, where both thresholds are obtained by the identical protocol on the identical speakers. This
pre-registration fixes that protocol for ECAPA before any score is seen.

## What is run

Step 1. ECAPA calibration, identical in every respect to the Resemblyzer calibration pre-registered in
`docs/preregistration_q2t_2026-10-26.md` part A, with the encoder as the only difference:

- scorer: `clone_eval_v2.Ecapa` — the same `speechbrain/spkrec-ecapa-voxceleb` model and the same
  `encode_batch` call that produced every `sim_ecapa` value already stored in `results/ce_ac_*.json`, so the
  threshold lands on the scale of the data it will be applied to;
- enrolment: the whole trimmed utterance (`trim_padding`, 20 s crop), which is what a clone is compared with;
- probes: a different real recording of the same or another speaker, cut to a length drawn from the observed
  clone lengths of f5tts / maskgct / cosyvoice (RQA, watermark only, unseen text), K = 3 crops per utterance,
  seed 20261027;
- development speakers: 40 LibriSpeech test-clean speakers × 5 utterances — these appear nowhere else in this work;
- test speakers: the 20 held-out train-clean-100 speakers × 5 utterances used throughout.

Step 2. Accept rates read off the existing result files at each verifier's own threshold. No cloning, no new
audio, no re-scoring: `sim_ecapa` and `sim_resemblyzer` are already stored per clip.

## Thresholds, fixed now

- **primary**: EER on the development pairs (the observed score where |FAR − FRR| is smallest), then frozen.
- **secondary**: the lowest threshold with development FAR ≤ 1 %.
- **external reference**: 0.25. SafeSpeech (USENIX Security 2025) counts an ECAPA similarity above 0.25 as the
  same speaker, and `clone_eval_v2` already reports `asr_sim_gt_025` on that basis. It is a published criterion,
  not a threshold calibrated on this data, and SafeSpeech applies it to the output of a fine-tuned model rather
  than a zero-shot clone — so it is reported alongside our EER threshold for comparison, exactly as De-AntiFake's
  0.697 is reported alongside our Resemblyzer threshold, and it is not the rule any result is decided by.
  (De-AntiFake separately reports 0.25 for **x-vector** and 0.697 for Resemblyzer; the coincidence of the two
  0.25 values is not evidence that the scales agree.)

Reported on the test speakers at each threshold: FAR, FRR, test EER, speaker-bootstrap 95 % CIs
(10,000 resamples).

## Hypotheses

**HE1 (primary).** At its own calibrated threshold, the in-loop verifier (ECAPA) reports a **lower** accept rate
for the protected arm `q2` than the held-out verifier (Resemblyzer) does, on all three strong cloners.

- Pass: the ECAPA accept rate for `q2` is lower than the Resemblyzer accept rate for `q2` on f5tts, maskgct and
  cosyvoice.
- Interpretation if it passes: the gap is a measurement of how much a self-graded evaluation inflates a reported
  protection result. It is a statement about evaluation methodology, not about the strength of the protection.
- Interpretation if it fails (the gap is absent or reversed): the in-loop/held-out distinction is weaker than
  claimed on this data, and the claim on slide 10 is softened to the direction only.

**HE2.** The ECAPA accept rate for the watermark-only arm `wm` is high (the clone is accepted), confirming that
the low ECAPA similarity of `q2` reflects the optimisation and not a verifier that rejects everything.

- Pass: ECAPA accept rate for `wm` ≥ 0.70 on all three strong cloners.
- If it fails, the ECAPA threshold is not usable as a decision rule and only the Resemblyzer row is reported.

## What this cannot show

- It cannot make the two similarity **scores** comparable. Only the decisions are compared, and only ever at each
  verifier's own calibrated threshold.
- It does not change any protection claim. Every protection number reported in the thesis remains the held-out
  (Resemblyzer) number. The ECAPA row is reported as a diagnostic and is never evidence that the voice changed.
- ECAPA is one of three in-loop verifiers; the result describes ECAPA, not the ensemble.
- The calibration is LibriSpeech-only, so the thresholds are specific to this corpus and recording condition.

## Deliverables

- `results/sv_calibration_ecapa.json`
- `results/accept_rates.json` — every (cloner, watermark, arm) cell, both verifiers, both thresholds, with
  speaker-bootstrap CIs and exact McNemar tests against the watermark-only arm.
