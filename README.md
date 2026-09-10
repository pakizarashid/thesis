# Dual-Defense Audio Protection: Watermarking + Anti-Cloning Perturbation

Combines VoiceMark-style traceable watermarking with SafeSpeech-style adversarial
perturbation, and evaluates whether both survive an attacker who denoises the protected
audio before cloning it.

**Core question:** attribution needs speaker information to *survive* cloning; anti-cloning
protection needs it *destroyed*. Can one system do both?

---

## Headline result

Speech-enhancement DEMUCS denoising, then zero-shot cloning. n=100, ECAPA-TDNN, LibriSpeech.

| metric | clean | protected | after DEMUCS |
|---|---|---|---|
| Speaker similarity (ECAPA-TDNN) | 0.4400 | **0.1468** | 0.1953 |
| Attack success rate (SIM > 0.25) | 95.0% | **17.0%** | 34.0% |
| Clone WER (Whisper) | 4.12% | 6.00% | 6.50% |
| Watermark accuracy | 0.9931 | 1.0000 | **0.9844–0.9950** |

**Protection cuts attacker success 95% → 17%. Denoising recovers it only to 34%.
Attribution stays at ~99% throughout.**

Against SafeSpeech's published numbers (same encoder, same 0.25 threshold):

| | protection strength | lost to DEMUCS | mean clone succeeds after attack? | attribution |
|---|---|---|---|---|
| SafeSpeech | 66.2% | 20.0% | **yes** (0.284 > 0.25) | none |
| This work | 66.6% | 16.5% | **no** (0.195 < 0.25) | ~99% |

Equivalent protection strength; comparable degradation under attack; **plus attribution the
baseline structurally lacks.**

---

## Key findings

**Both protections compose.** Applying disruption costs nothing in watermark survival
(p = 0.75, null result). They are not antagonistic — an earlier hypothesis that they compete
for the speaker channel was tested and falsified.

**Disruption transfers black-box.** PGD optimised against YourTTS degrades XTTS-v2 clones
(different architecture, non-differentiable, unseen by the optimiser): SIM 0.4930 → 0.3747,
p = 1.9 × 10⁻⁸.

**VoiceMark's cloning robustness does not generalise.** Using the authors' own released
weights: 0.6119 on XTTS-v2 and 0.5337 on YourTTS, against their published 0.957–0.979 on
CosyVoice/F5-TTS/MaskGCT. Reproduction verified identical to their model (p = 0.97);
reference duration ruled out (p = 0.20).

**The attack that best removes protection best preserves the watermark.** A speech-enhancement
denoiser strips the adversarial perturbation but preserves speech structure — including the
speaker latent carrying the watermark. Attribution *rose* under the stronger attack
(0.938 → 0.99).

**Four ways to strengthen attribution, all eliminated:** relocating the perturbation to the
speaker latent, increasing its magnitude, concentrating the payload in a single RVQ layer
(no single layer carries a readable payload — the code is distributed), and clone-aware
detector training (works in-loop, +0.128; does not transfer, p = 0.117).

---

## Reproduce

```bash
bash scripts/setup_env.sh

# Protection + attribution under DEMUCS (writes WAVs for re-scoring)
python src/eval/demucs_fallback_eval.py \
  --checkpoint ./checkpoints/stage1_final_scaleup_recalibrated/recalibrated_final.pt \
  --backend denoiser --n_utterances 100 \
  --save_clones_dir ./audio_samples/demucs_speech \
  --output results/results_demucs_speech_n100.json

# Speaker similarity, comparable to SafeSpeech (separate process — no pipeline imports)
python src/eval/ecapa_sim_eval.py --sample_dir ./audio_samples/demucs_speech \
  --output results/results_ecapa_sim_speech.json

# Clone intelligibility
python src/eval/clone_wer_eval.py --sample_dir ./audio_samples/demucs_speech \
  --output results/results_clone_wer_speech.json
```

Every eval script has `--diagnostic` — run it first. It checks assumptions in seconds
rather than after hours of GPU time.

---

## Layout

```
src/models/      backbone (VoiceMark + LoRA), differentiable YourTTS surrogate
src/losses/      VoiceMark's five losses, SafeSpeech-derived disruption losses
src/data/        LibriSpeech/VCTK loaders, VC-simulated augmentations
src/train.py     Stage 1 watermark training (--use_augmentation)
src/eval/        evaluation suite (see below)
results/         all result JSONs, per-utterance values included
docs/            full experimental write-up
```

Key eval scripts: `demucs_fallback_eval.py` (dual defense under denoising),
`ecapa_sim_eval.py` (SafeSpeech-comparable SIM), `clone_wer_eval.py` (intelligibility),
`xtts_transfer_eval.py` (black-box cloning transfer), `disruption_pgd.py` (protection),
`layer_subspace_probe.py` (where the watermark lives).

---

## Limitations

- **Zero-shot threat model only.** SafeSpeech's headline numbers come from fine-tuning
  attacks; that model is not evaluated here. This also explains the flat WER — in zero-shot
  cloning, content comes from the text encoder, so perturbing the reference affects voice,
  not intelligibility.
- **No coverage of CosyVoice, F5-TTS or MaskGCT** — the three models VoiceMark evaluates.
  F5-TTS is blocked by a `transformers` version conflict with `coqui-tts`.
- **Augmentation is a reconstruction.** VoiceMark's training code is not public; segment
  lengths, probabilities and the codec for "neural" distortion were inferred. `sox_effects`
  was unavailable at runtime, so that augmentation ran as a quantisation approximation.
- **Different corpus and cloner from SafeSpeech** (LibriSpeech/YourTTS vs LibriTTS/BERT-VITS2),
  so absolute SIM is not comparable — normalised comparisons are used throughout.
- **No subjective listening test (SMOS).** All quality evidence is objective.

---

## Reproducibility notes

- **Report ASR, not mean SIM.** The protected-clone distribution is right-skewed; the mean
  can cross the threshold while the median sits well below it.
- **Name the denoiser.** A music separator (torchaudio HDemucs) removes almost no
  perturbation; a speech enhancer removes it at SafeSpeech's own rate. Any DEMUCS number
  without the exact model is uninterpretable.
- **Noise floors:** ~0.02 for XTTS evaluations (stochastic decoding), ~3 points across
  identical `denoiser` runs. Report means of repeated runs.
- **Paired tests on complete sets.** A 34-utterance subsample once gave p = 0.038 where the
  full 100 gave p = 0.117.

---

## References

VoiceMark ([Interspeech 2025](https://www.isca-archive.org/interspeech_2025/li25g_interspeech.pdf)) ·
SafeSpeech ([USENIX Security 2025](https://www.usenix.org/system/files/usenixsecurity25-zhang-zhisheng.pdf)) ·
Dual Defense ([IEEE TIFS](https://arxiv.org/abs/2310.16540)) ·
AudioPure · ECAPA-TDNN (speechbrain)

Full experimental record, including negative results and withdrawn claims:
[`docs/experimental_writeup.md`](docs/experimental_writeup.md)
