# Pre-registration: F5-TTS instead of YourTTS in the clone-aware training loop (notebook 27)

Written 2026-10-26, before any run. Code:
- `src/models/f5_surrogate.py` (F5-TTS in-loop cloner).
- `src/train_route2_clone_aware.py --cloner f5tts`. The options `--grad_accum` and `--resume` were added and default to unchanged behaviour.
- `src/eval/summarise_f5loop.py` (every number and verdict below).

## Question
R0 was trained with YourTTS as the differentiable cloner in the loop. It improved watermark survival on every cloner, and most on the weak ones (YourTTS 0.53 -> 0.63, XTTS-v2 0.64 -> 0.69). Does replacing YourTTS with F5-TTS change that? Is the clone-aware gain tied to which cloner is in the loop?

## Design
- **Arms.**
  - R0F5: R0's exact settings (init `stage1_vmaug/stage1_epoch29.pt`, `--train_msg_processor --epochs 20 --clone_warmup_epochs 2 --batch_size 2 --clone_text sampled --seed 0`, no quality or augmentation terms) with `--cloner f5tts`.
  - R0F5s1: the same with seed 1, for training noise.
  - Nothing else changes.
- **F5-TTS in the loop.**
  - The same differentiable sampler as the anti-cloning perturbation Q2: K = 6 Euler steps, CFG 2, gradient checkpointing.
  - Prompt transcript: faster-whisper base.en of the watermarked prompt.
  - Output: the generated region only, Vocode at 24 kHz resampled to 16 kHz.
  - Fresh seeded noise per call.
  - If batch 2 does not fit in memory: `--batch_size 1 --grad_accum 2`, which keeps the same samples per optimiser step. This is recorded in the checkpoint's train_args.
- **Evaluation.**
  - The combined-summary protocol (`clone_eval_v2.py`): the same 100 LibriSpeech clips (20 held-out speakers), 3 s prompt, fixed and unseen text, messages 123 + i, five cloners.
  - Quality and per-augmentation ACC with `quality_paired_ckpts.py` on the same clips (control R0).
- **Comparisons.**
  - R0F5 vs R0, and R0F5 vs official VoiceMark, paired per clip, unseen text.
  - Two-sided Wilcoxon on ACC with Holm correction within each group, exact McNemar on traced, speaker-level bootstrap CIs.
  - Tolerance = max(0.02, |R0 - R0s1|, |R0F5 - R0F5s1|) per cloner.

## Decision rules (per cloner)
- **F5 IN THE LOOP IS BETTER:** R0F5 - R0 > tolerance and Holm p < 0.05.
- **F5 IN THE LOOP IS WORSE:** R0F5 - R0 < -tolerance and Holm p < 0.05.
- **NO DIFFERENCE:** otherwise.
- **Groups.** Weak cloners are XTTS-v2 and YourTTS; strong cloners are F5-TTS, MaskGCT and CosyVoice. Holm correction is applied within each group.

## Predictions (before the run)
1. **Weak cloners:** R0F5 is WORSE than R0 on YourTTS (about 80%) and on XTTS-v2 (about 60%). R0F5 is within ±0.03 of official VoiceMark on both (about 65%). Reason: the weak-cloner gain needs a cloner that keeps only a speaker embedding in the loop. F5-TTS copies the prompt.
2. **Strong cloners:** NO DIFFERENCE on MaskGCT and CosyVoice (about 75%). On F5-TTS, NO DIFFERENCE or BETTER, with R0F5 >= 0.98 (about 80%).
3. **Quality:** PESQ(R0F5) - PESQ(R0) >= 0 (about 70%), because surviving F5-TTS is easy, so there is less pressure on the waveform. Clean ACC >= 0.95 (about 90%).
4. **Cost:** 2.5-5 s per training step on a T4, so 6-11 h per arm.

## Consequences
- **WORSE on the weak cloners and NO DIFFERENCE on the strong ones:** the clone-aware gain depends on which cloner is in the loop. The thesis keeps YourTTS for R0 and RQA and states this as evidence that survival is set by the cloner's conditioning bottleneck.
- **BETTER on the weak cloners:** unexpected. An F5-TTS-loop version of RQA would be the next registered arm.
- **Either way,** no further arms are added from this experiment without a new pre-registration.

## Limits
- R0 settings only (no quality terms), so this is a comparison of in-loop cloners, not a new operating point.
- One training corpus, 100 evaluation clips and 20 speakers.
- A Kaggle session limit may force a resume. The loader is replayed so data order and crops match, but the F5 noise counter and the text sampler restart. Any resume is recorded.
