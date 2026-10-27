# Pre-registration: verifier threshold, and Q2 + a prompt-transcription attack (notebook 28)

Written 2026-10-26, before any run of this experiment. Code:
- `src/eval/sv_threshold_calibration.py`: part A.
- `src/eval/protect_prompts_q2t.py`: the arms t and q2t.
- `src/eval/prompt_asr_eval.py`: what recognisers hear in each prompt.
- `src/eval/f5_whitebox_pgd.py`: adds `prompt_fn` / `prompt_weight`; Q2 is unchanged when they are unset.
- `src/eval/clone_eval_v2.py`: adds `--prompt_whisper_size` and `--only_indices`; defaults leave behaviour unchanged.
- `src/eval/summarise_q2t.py`: every number and verdict below.

## Why
Notebook 26 found that Q2 lowers held-out speaker similarity on every cloner, but two things are missing.
- **No absolute answer.** It never said whether a clone is still accepted as the speaker. With De-AntiFake's Resemblyzer threshold (0.697), 70–84% of Q2 clones from the three strong cloners would still be accepted.
- **No WER gain.** The intelligibility term on the F5-TTS rollout (q2c) did not raise clone WER.

E2E-VGuard attacks a different point instead: the ASR that transcribes the prompt. A cloner that needs a prompt transcript (MaskGCT, CosyVoice, and F5-TTS when no text is given) is then given text that does not match the prompt audio.

## Part A: decision threshold for Resemblyzer
- **Scoring:** the evaluation's own `clone_eval_v2.Resem`. Enrolment is the whole trimmed utterance, as for clones.
- **Pairs:**
  - Genuine: another recording of the same speaker.
  - Impostor: a recording of another speaker.
  - No recording is ever compared with itself.
- **Probes:** cut to clone-like lengths. Lengths are drawn from the clone lengths of the strong cloners (RQA, watermark only, unseen text). There are 3 crops per probe utterance, seed 20261027.
- **Development speakers:** LibriSpeech test-clean, 40 speakers × 5 utterances.
- **Test speakers:** the 20 held-out train-clean-100 speakers × 5 utterances, the same clips as every cloning result. The sets are checked to be disjoint.
- **Primary threshold:** EER on development, the observed score where |FAR − FRR| is smallest.
- **Secondary threshold:** the lowest threshold with FAR ≤ 1% on development. A strict threshold rejects more clones, so it is the optimistic bound for a protection, and it is reported as such.
- **Reference threshold:** 0.697 (De-AntiFake).
- **Reported on the test speakers at each threshold:** FAR and FRR with speaker-bootstrap 95% CIs, the score histograms, and acceptance curves from 0.50 to 0.95.
- **Acceptance:** "accepted as the speaker" means Resemblyzer(clone, enrolment) ≥ the threshold. Every acceptance rate is reported with its Wilson CI.
- **Scope:** the threshold applies to this verifier and these conditions. It is not a universal speaker-identity boundary.

## Part B: arms (RQA watermark, the 100 clips of notebook 26)
- **wm, rs, q2:** the notebook 26 prompts, unchanged.
- **q2t:** Q2 exactly as registered, plus a targeted ASR term on the prompt.
  - Surrogate recogniser: `openai/whisper-base.en`, with a differentiable log-mel checked against `WhisperFeatureExtractor`.
  - Loss: teacher-forced cross-entropy of a target sentence, minimised.
  - Target for clip i: the first N words of the LibriSpeech transcript of clip (i + 37) mod 100, which is always another speaker. N is the number of words base.en hears in the clean prompt, at least 3.
  - Gradient weight: 1.732 × ‖grad_wm‖, the same as the speaker term.
  - Same bound (shaped, peak 0.015), 40 steps, step 0.0015 and seeds as Q2.
- **t:** the ASR term and the watermark term only. There is no speaker term and no F5-TTS rollout; bound, steps and start seed are the same. This arm is for attribution.

Which recogniser is matched and which is held out:
- **MaskGCT and CosyVoice:** their prompt text comes from faster-whisper base.en, the same weights as the surrogate, so this is the matched case. Sensitivity runs transcribe with large-v3-turbo (held out) for q2 and q2t.
- **F5-TTS:** transcribes its prompt itself with Whisper large-v3-turbo, so this is held out.

Prompt level: `prompt_asr_eval.py` reports the WER of base.en, large-v3-turbo and wav2vec2 on every arm's prompt, against the large-v3-turbo transcript of the clean prompt.

Cloning:
- **Main runs:** F5-TTS, MaskGCT and CosyVoice, 3 s prompt, unseen text, arms t and q2t. Clone WER is measured with faster-whisper base.en against the target text. The clone itself carries no perturbation, so the attacked model is not scoring its own adversarial input.
- **Listening set:** clips 23, 1, 87, 31, 12 and 54 are re-cloned for wm, rs, q2, t and q2t with audio saved. Their scores are reported from that run, not mixed with the main results.

## Hypotheses and decision rules (unseen text, paired per clip, speaker-bootstrap CIs)
- **HT1, primary (intelligibility).** Clone WER q2t − q2, one-sided Wilcoxon, Holm over the three cloners.
  - RAISES WER: Holm p < 0.05 and mean difference ≥ 0.05.
  - NO WER GAIN SHOWN: otherwise.
  - Also reported: the share of clones with WER ≥ 0.5, with exact McNemar.
- **HT2 (identity).** Acceptance at the primary threshold, q2t vs q2, exact McNemar, Holm over the three cloners.
  - FEWER ACCEPTED or MORE ACCEPTED: Holm p < 0.05, in the direction observed.
  - NO DIFFERENCE: otherwise.
- **HT3.** Resemblyzer q2 − q2t, two-sided. Descriptive.
- **HT4 (watermark).** Under q2t, SURVIVES if traced ≥ 0.80 and ACC ≥ 0.90, the same rule as notebook 26's H2.
- **HT5 (attribution).** t vs wm on WER and on similarity. Descriptive.
- **Calibrated notebook 26 results.** Acceptance with McNemar for q2 vs wm, q2 vs rs and rs vs wm. Descriptive, no new claims.
- **Quality.** Prompt PESQ, STOI and SI-SNR of t and q2t, and PESQ q2t − q2. Reported, no rule.

## Predictions (before the run)
1. HT1: RAISES WER on MaskGCT and CosyVoice (about 60%), because their transcriber has the same weights as the surrogate. On F5-TTS, NO WER GAIN SHOWN (about 60%), because its transcriber is held out.
2. HT2: NO DIFFERENCE on all three (about 65%). A wrong transcript changes content more than timbre.
3. The prompt-level base.en WER of q2t is at least 0.5 (about 70%); large-v3-turbo's is lower than base.en's (about 80%).
4. HT4: q2t does not survive on MaskGCT or CosyVoice, the same as q2 (about 70%).
5. Calibration: development EER between 0.02 and 0.10, with the EER threshold between 0.65 and 0.80 (about 70%).

## Consequences
- **HT1 RAISES WER only in the matched case:** the thesis reports the transcription attack as effective when the defender knows the attacker's ASR, and not shown otherwise.
- **HT1 RAISES WER on held-out recognisers too:** q2t replaces Q2 as the protection, and the E2E-VGuard comparison is run on it.
- **Either way,** Q2 and q2t are described with the calibrated acceptance rate, not with the similarity drop alone.

## Deviations allowed
None for the rules. Platform (Colab or Kaggle), GPU and resume points are recorded in the result files.

## Limits
- One dataset, 100 clips and 20 speakers.
- The target sentence rule is one choice among several in E2E-VGuard.
- 40 PGD steps (E2E-VGuard uses 500), so the ASR attack is weaker than E2E-VGuard's own. This is fixed here so that q2t costs the same as Q2.
