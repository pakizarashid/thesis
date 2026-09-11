# Dual-Defense Audio Protection: Watermarking + Anti-Cloning Perturbation

Combines VoiceMark-style traceable watermarking with SafeSpeech-style adversarial
perturbation, and evaluates whether both survive an attacker who denoises the protected
audio before cloning it.

**Core question:** attribution needs speaker information to *survive* cloning; anti-cloning
protection needs it *destroyed*. Can one system do both?

**Short answer:** yes on the architecture the protection was built against (YourTTS) —
both halves hold together. On the architecture attribution works best on (F5-TTS),
attribution transfers but the protection does not. Both results are reported below.

---

## Headline result — dual defense on YourTTS

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

**Both protections compose on YourTTS.** Applying disruption costs nothing in watermark
survival (p = 0.75, null result) — not antagonistic, tested and falsified as a hypothesis.

**Watermark survival through cloning is architecture-dependent, and predictable — across
all five architectures VoiceMark itself evaluates.** Same detector, same 16-bit payload,
same harness throughout:

| cloner | speaker conditioning | ACC in clone |
|---|---|---|
| YourTTS | fixed d-vector | 0.5337 |
| XTTS-v2 | GPT audio-prompt tokens (discretised) | 0.6119 |
| CosyVoice 2 | flow-matching decoder, regenerated from prompt mel | 0.7669 |
| MaskGCT | masked infilling, quantised RVQ tokens retained in-context | 0.9137 |
| **F5-TTS** | mel infilling, reference mel retained | **0.9300** |

Monotone across all five, zero ties, zero inversions. Survival tracks two things: how
much raw reference detail the architecture is exposed to (separates the YourTTS/XTTS-v2
cluster from the rest), and — sharper — whether that detail reaches the output as
**retained** samples/tokens (F5-TTS, MaskGCT: 0.91–0.93) or as **regenerated**
conditioning through a decoder (CosyVoice: 0.77). MaskGCT's tokens are quantised and still
land within 2 points of F5-TTS's continuous mel — quantisation itself does not break the
carrier; retained-vs-regenerated does. This explains VoiceMark's published 0.957–0.979
(CosyVoice/MaskGCT/F5-TTS): their evaluation set is entirely high-bandwidth architectures.

**Protection does not transfer to the architecture attribution works best on.** The PGD
perturbation above is optimised against a differentiable YourTTS surrogate. Measured on
F5-TTS clones of protected audio (n=100): attribution still holds well above chance
(ACC 0.8381 protected, 0.8187 after DEMUCS — a real ~0.09 cost from protection itself, not
from denoising), but speaker similarity barely moves (SIM 0.41–0.43, attack success
91–92%, vs. this project's own 17% on YourTTS and SafeSpeech's own 0.204/~17% at the same
threshold). **The two halves of dual defense decouple by architecture**: both hold on
YourTTS; only attribution transfers to F5-TTS. An F5-TTS-aware surrogate is future work,
not attempted here.

**The attack that best removes protection best preserves the watermark (on YourTTS).** A
speech-enhancement denoiser strips the adversarial perturbation but preserves speech
structure — including the speaker latent carrying the watermark. Attribution *rose* under
the stronger attack (0.938 → 0.99).

**Four ways to strengthen attribution, all eliminated:** relocating the perturbation to
the speaker latent, increasing its magnitude, concentrating the payload in a single RVQ
layer (no single layer carries a readable payload — the code is distributed), and
clone-aware detector training (works in-loop, +0.128; does not transfer, p = 0.117).

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

# Speaker similarity, comparable to SafeSpeech (separate process, speechbrain only)
python src/eval/ecapa_sim_eval.py --sample_dir ./audio_samples/demucs_speech \
  --output results/results_ecapa_sim_speech.json
```

**F5-TTS, CosyVoice, and MaskGCT each run in their own clean environment** (own GPU job,
own notebook — see the docstrings in `cloner_watermark_eval.py` and
`f5tts_watermark_only.py` for exact install steps; do not co-install, their `transformers`
pins conflict). All three reuse WAVs the DEMUCS run above already wrote via
`--audio_only --save_clones_dir`:

```bash
# CosyVoice / MaskGCT -- one script, --cloner picks the backend
python src/eval/cloner_watermark_eval.py --cloner cosyvoice \
  --cosyvoice_root ./CosyVoice --auto_transcribe --n_utterances 100 \
  --output results/results_cosyvoice_n100.json

python src/eval/cloner_watermark_eval.py --cloner maskgct \
  --amphion_root ./Amphion --auto_transcribe --n_utterances 100 \
  --output results/results_maskgct_n100.json

# F5-TTS: watermark-only, then the protected / post-DEMUCS arms (reusing the DEMUCS WAVs)
python src/eval/f5tts_watermark_only.py --n_utterances 100 \
  --output results/results_f5tts_wmonly_n100.json

CKPT=./checkpoints/stage1_final_scaleup_recalibrated/recalibrated_final.pt
for arm in watermarked denoised; do
  python src/eval/f5tts_watermark_only.py --checkpoint $CKPT \
    --input_wav_dir ./audio_samples/demucs_speech_v2/audio --wav_suffix $arm \
    --n_utterances 100 --save_clones_dir ./audio_samples/f5tts_$arm \
    --output results/results_f5tts_$arm.json
done
```

`--checkpoint` on the protected/denoised arms must match whatever `demucs_fallback_eval.py`
used to embed the mark — a mismatch degrades detection silently rather than erroring.
`--message_seed_base` (default 123) must match the generating script; the script aborts on
utterance 0 if source ACC < 0.85 rather than let a silent mismatch read as "protection
destroys attribution." Every eval script has `--diagnostic` — run it first.

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

Key eval scripts: `demucs_fallback_eval.py` (dual defense under denoising, on YourTTS),
`ecapa_sim_eval.py` (SafeSpeech-comparable SIM, speechbrain-only), `cloner_watermark_eval.py`
(watermark survival through cloning — CosyVoice/MaskGCT, one protocol), `f5tts_watermark_only.py`
(same protocol for F5-TTS, plus the protected/post-DEMUCS arms), `xtts_transfer_eval.py`
(black-box cloning transfer), `disruption_pgd.py` (protection), `layer_subspace_probe.py`
(where the watermark lives).

`audio_samples/` is gitignored — it holds regenerable clone/reference WAVs, not
source-of-truth. `checkpoints/` and `results/` are tracked; they're small.

---

## Limitations

- **Zero-shot threat model only.** SafeSpeech's headline numbers come from fine-tuning
  attacks; that model is not evaluated here.
- **Protection is YourTTS-surrogate-specific, not architecture-general.** It transfers
  black-box to XTTS-v2 (SIM 0.4930 → 0.3747, p = 1.9×10⁻⁸) but not to F5-TTS (SIM barely
  moves). Attribution transfers to all five architectures measured; protection has only
  been shown to transfer to one of the other four.
- **Augmentation is a reconstruction.** VoiceMark's training code is not public; segment
  lengths, probabilities and the codec for "neural" distortion were inferred.
- **Different corpus and cloner from SafeSpeech** (LibriSpeech/YourTTS vs
  LibriTTS/BERT-VITS2), so absolute SIM is not comparable — normalised comparisons are
  used throughout.
- **No subjective listening test (SMOS).** All quality evidence is objective.

---

## Reproducibility notes

- **Report ASR, not mean SIM** — the protected-clone distribution is right-skewed.
- **Name the denoiser** — a music separator (torchaudio HDemucs) removes almost no
  perturbation; a speech enhancer removes it at SafeSpeech's own rate.
- **Paired tests on complete sets** — a 34-utterance subsample once gave p = 0.038 where
  the full 100 gave p = 0.117.

---

## References

VoiceMark ([Interspeech 2025](https://www.isca-archive.org/interspeech_2025/li25g_interspeech.pdf)) ·
SafeSpeech ([USENIX Security 2025](https://www.usenix.org/system/files/usenixsecurity25-zhang-zhisheng.pdf)) ·
Dual Defense ([IEEE TIFS](https://arxiv.org/abs/2310.16540)) ·
AudioPure · ECAPA-TDNN (speechbrain)

Full experimental record, including negative results and withdrawn claims:
[`docs/experimental_writeup.md`](docs/experimental_writeup.md)
