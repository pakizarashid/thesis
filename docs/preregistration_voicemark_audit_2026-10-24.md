# Pre-registration: VoiceMark reproduction audit (Experiments A and B)

Written 2026-10-24, before any run below. Companion to `docs/preregistration_quality_cost_2026-10-24.md`. The aim is to separate two questions that earlier comparisons mixed: (A) does the official VoiceMark checkpoint reproduce the paper's numbers under our evaluation, and which evaluation choices move the result; (B) does our training implementation, before and after the 2026-10-24 fix, behave as the paper describes. A difference must not be attributed to augmentation when it could come from the checkpoint, prompt duration, generated text or test-set composition.

## Experiment A — checkpoint and evaluation audit (pretrained VoiceMark only, no training)

- Checkpoint: `external/voicemark/voicemark.pth` and `SpeechTokenizer.pt`, SHA-256 compared with the LFS hashes published by the authors' Space `haiyunli/VoiceMark` (`provenance_manifest.py --check_hf_space`). A mismatch stops the reading.
- Protocol: `src/eval/clone_eval_v2.py`; messages 123 + clip; ACC = bits correct / 16 (paper's definition); FAR with 99 random candidates (paper's definition; candidates are drawn randomly, ties counted as false); also traced (>= 13/16), ECAPA SIM, WER, gate.
- **Baseline configuration B0**: our clips (LibriSpeech train-clean-100, 20 held-out speakers x 5 = 100 clips), 3 s centre crop as prompt, fixed sentence "This is a test sentence for voice cloning.".
- **One factor changed at a time** from B0:
  1. text: unseen held-out transcript (same clips, same prompts);
  2. prompt: 6 s centre crop; 3. prompt: full utterance (capped at 20 s) (same clips);
  4. test set: LibriSpeech **test-clean** (all 40 speakers x 5 = 200 clips); 5. test set: VCTK (100 clips, 20 speakers);
  plus one combined point closest to the paper: test-clean, full utterance, unseen text.
- Cloners: CosyVoice, F5-TTS, MaskGCT (the paper's three), YourTTS and XTTS-v2 for context.
- Reading rules:
  - A factor "changes the result" for a cloner if the difference from B0 exceeds 0.02 ACC and, for the same-clip factors (text, prompt), the paired Wilcoxon test gives p < 0.05; for the test-set factors (different clips), if the speaker-level 95% interval of the new condition excludes the B0 mean. Otherwise "no detectable change".
  - **Reproduced** for a cloner if the paper-like point is within 0.02 of the paper's ACC (CosyVoice 0.964, F5-TTS 0.979, MaskGCT 0.957) or the paper's value lies inside its speaker-level 95% interval. Otherwise "not reproduced", with the factor table as the account of the gap. A cloner that fails the cloner-works gate is reported but not scored.
- Known limits (stated now): the paper does not give prompt length, cloner versions or inference settings; VoiceMark was trained on the VCTK train split and its held-out speakers are not published, so our VCTK clips may include its training speakers (the VCTK row can be optimistic); test-clean is safely unseen (VoiceMark did not train on LibriSpeech). Matching ACC does not prove the implementation matches; it is one metric under one protocol.

## Experiment B — training implementation audit (Stage 1)

- Same data (900 train-clean-100 utterances, 60 speakers), 30 epochs, batch 4, lr 5e-5, LoRA rank 8, start from pretrained VoiceMark, checkpoint of epoch 29 evaluated:

| Arm | Training path | Seed |
|---|---|---|
| S1-legacy | `--use_augmentation --legacy_augmentation` (the original code path: augmented audio discarded, only VAD labels changed; the "neural" step falls back to 8-bit rounding exactly as before) | 0 |
| S1-corrected | `--use_augmentation` (VoiceMark-faithful augmentation fed to the detector) | none (the run started 2026-10-24 before `--seed` existed) |
| S1-corrected-rep | `--use_augmentation` | 1 (training-noise estimate) |

  The old `stage1_scaleup_aug` checkpoint (training command not recorded) and pretrained VoiceMark are reported for reference only; they are not the comparison.
- Evaluation, same clips for all arms: `quality_paired_ckpts.py` (PESQ, STOI, SI-SNR, clean ACC, ACC under each augmentation type separately) and `clone_eval_v2.py` (3 s, fixed and unseen text) on every cloner.
- Reading rules:
  - Implementation check: under each augmentation type, S1-corrected should exceed S1-legacy; "confirmed" for a type if the paired Wilcoxon p < 0.05. Pretrained VoiceMark's per-type ACC shows how close our augmentations are to the ones it was trained with.
  - Cloning: corrected vs legacy per cloner, unseen text: "improves" / "worsens" if paired p < 0.05 and the difference exceeds max(0.02, |S1-corrected - S1-corrected-rep|); otherwise "no detectable difference".
  - Quality: PESQ/STOI/SI-SNR differences reported with paired tests; no threshold.

## Predictions (written now)

1. The checkpoint hashes match the Space: about 95%.
2. Paper-like point within 0.02 of the paper: F5-TTS about 50%, MaskGCT about 45%; CosyVoice passes the gate at all: about 35%.
3. Largest single factor (F5-TTS / MaskGCT): prompt length about 50%, test set about 25%, text about 15%, none above 0.02 about 10%.
4. S1-corrected beats S1-legacy under the replace and EnCodec augmentations: about 80%; under all five types: about 55%.
5. S1-corrected improves on S1-legacy on XTTS-v2 (unseen text): about 55%; on F5-TTS and MaskGCT (near ceiling): about 35%.
6. |S1-corrected - S1-corrected-rep| on XTTS-v2 ACC <= 0.02: about 60%.
7. S1-corrected stays within 0.01 of pretrained VoiceMark under augmentation (per-type mean): about 45% (the run log at epoch 13 shows 0.92-0.93 against pretrained 0.943).
