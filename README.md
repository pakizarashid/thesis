# Dual-Defense Audio Protection: Watermarking + Anti-Cloning Perturbation

Combines VoiceMark-style traceable watermarking with SafeSpeech-style adversarial
perturbation, and asks whether both survive contact with real zero-shot voice cloning.

**Core question:** attribution needs speaker information to *survive* cloning; anti-cloning
protection needs it *destroyed*. Can one system do both — and if not everywhere, exactly
where does it break, and can that be fixed?

The work below is a five-stage progression. Stages 1–3 are complete and their findings are
final. Stage 4 is the open decision — three candidate directions are sized below, none
started yet. Stage 5 depends on which direction Stage 4 takes.

---

## Stage 1 — Baseline: does the watermark survive being cloned?

VoiceMark's own released watermark, tested against all five zero-shot TTS architectures its
own paper evaluates, using the same detector, the same 16-bit payload, and the same harness
throughout.

| cloner | speaker conditioning | watermark ACC in the clone |
|---|---|---|
| YourTTS | fixed d-vector (single speaker embedding) | 0.5337 |
| XTTS-v2 | GPT audio-prompt tokens (discretised) | 0.6119 |
| CosyVoice 2 | flow-matching decoder, regenerated from prompt mel | 0.7669 |
| MaskGCT | masked infilling, quantised RVQ tokens retained in-context | 0.9137 |
| F5-TTS | mel infilling, reference mel retained | 0.9300 |

**Finding: watermark survival is architecture/reference-pathway dependent, not a fixed
property of the watermark.** The ordering is monotone across all five points, zero ties, zero
inversions. It tracks two things: how much raw reference acoustic detail an architecture is
exposed to, and — more decisive — whether that detail reaches the output **retained**
(F5-TTS, MaskGCT) or **regenerated** through a decoder (CosyVoice). This also explains
VoiceMark's own published numbers (0.957–0.979): their evaluation set is entirely
high-bandwidth, retained-conditioning architectures.

**Status: done.**

---

## Stage 2 — Anti-cloning: does the protection transfer to the same five architectures?

A PGD-style adversarial perturbation, optimised against a differentiable YourTTS surrogate,
layered on top of the watermark.

On the architecture it was built for (YourTTS), denoising-attack scenario, n=100:

| metric | clean | protected | after DEMUCS |
|---|---|---|---|
| Speaker similarity (ECAPA-TDNN) | 0.4400 | 0.1468 | 0.1953 |
| Attack success rate (SIM > 0.25) | 95.0% | 17.0% | 34.0% |
| Watermark ACC | 0.9931 | 1.0000 | 0.9844–0.9950 |

Both objectives compose at zero cost here (adding disruption costs nothing in watermark
survival, p = 0.75), and this is comparable to or better than SafeSpeech's own published
protection strength at the same threshold.

Measured on the other architectures:

| cloner | does the YourTTS-trained perturbation transfer? |
|---|---|
| XTTS-v2 | yes — SIM 0.4930 → 0.3747, p = 1.9 × 10⁻⁸ |
| F5-TTS | no — SIM stays at 0.41–0.43, attack success 91–92%, barely below the *unprotected* baseline |

**Finding: protection transfer is also architecture-dependent — and it fails specifically on
the architecture where attribution is strongest.** Attribution survives cloning on F5-TTS
(ACC 0.8381 protected, 0.8187 after DEMUCS — a real but modest ~0.09 cost from protection
itself), but the perturbation that works on YourTTS and transfers to XTTS-v2 essentially does
nothing to F5-TTS's cloning success. The two halves of the defense decouple by architecture.

**Status: done.**

---

## Stage 3 — Epsilon sweep: does a bigger perturbation budget buy back protection on F5-TTS?

Same YourTTS-trained PGD objective, perturbation budget raised well past the original
operating point (ε = 0.002), re-measured on F5-TTS.

| ε | attribution ACC (F5-TTS clone) | SIM mean | SIM median | attack success rate |
|---|---|---|---|---|
| 0.002 (original operating point) | 0.8381 | 0.4140 | 0.4154 | 91.0% |
| 0.01 | 0.7000 | 0.3250 | 0.3096 | 75.0% |
| 0.02 | 0.6844 | 0.2713 | 0.2708 | 55.0% |
| 0.04 | 0.6062 | 0.1605 | 0.1573 | 15.0% |
| 0.08 | 0.5875 | 0.0969 | 0.1010 | 5.0% |

Audio quality at the same budgets, measured independently:

| ε | PESQ | STOI | SI-SNR |
|---|---|---|---|
| 0.002 (original operating point) | 1.919 | 0.885 | 0.43 dB |
| 0.01 | 1.337 | 0.834 | 0.24 dB |
| 0.02 | 1.155 | 0.784 | −0.28 dB |
| 0.04 | 1.069 | 0.708 | −1.77 dB |
| 0.08 | 1.042 | 0.610 | −4.95 dB |

(YourTTS-side sanity, same runs, own architecture rather than the F5-TTS transfer question
above: protected ACC stays 0.94–0.99 and protected-after-DEMUCS ACC declines from 0.91 to
0.62 across the same ε range — the surrogate's own architecture keeps working throughout;
it's specifically the transfer to F5-TTS that's in question.)

**Finding: increasing ε eventually does buy back protection on F5-TTS — SIM falls from 0.41
to 0.10, attack success from 91% to 5% — but not for free.** Attribution degrades toward
chance in parallel (0.84 → 0.59), and, independently, audio quality collapses: PESQ is near
its floor by ε = 0.04, and by ε = 0.08 the perturbation is on average louder than the speech
itself (negative SI-SNR). There is no ε in this range where protection, attribution, and
usable audio all hold at once — it's a three-way trade-off across the full swept range, not
a clean case of the protection simply not working on F5-TTS.

**Status: done** (trend pass, n=20 per point above ε = 0.002; the ε = 0.002 row is the
earlier n=100 result).

---

## Stage 4 — The actual contribution: a defense that knows the watermark exists

Stage 3 diagnosed *why* the trade-off exists, not just *that* it exists: the PGD objective
optimises purely for reduced speaker similarity — it has no term that requires the watermark
to keep decoding, and no exposure to F5-TTS's conditioning behaviour at all. Pushing ε further
is not really "more protection" so much as "a blunter perturbation that damages everything in
the same acoustic budget the watermark and the audio quality also depend on."

The goal for this stage: a defense that reaches low speaker similarity, high watermark ACC,
and acceptable audio quality *together*, at a smaller perturbation budget than the naive
sweep needs. Three candidate directions, sized honestly:

| direction | what it requires | rough cost | risk to the December deadline |
|---|---|---|---|
| watermark-aware perturbation objective (add a term that keeps the watermark decodable through the same cloning process the perturbation is optimised against, alongside the existing anti-cloning and quality terms) | extends the existing optimisation loop, no new models | days | low |
| ensemble / F5-TTS-aware surrogate (optimise the perturbation against more than one cloning architecture, or one built to resemble F5-TTS's conditioning) | a new differentiable cloning model to optimise against | weeks, uncertain convergence | high |
| end-to-end watermark retraining with the cloning operation inside the training loop (the mechanism that solved this exact antagonism in the face-swap domain) | retraining the watermark itself against F5-TTS's actual cloning behaviour | weeks or more | very high |

**Status: direction not yet selected.** This is the open decision this stage is waiting on.

---

## Stage 5 — Final evaluation

Once a direction is chosen and implemented, this stage re-measures it against the same
protocol as Stage 3 — attribution ACC, SIM/attack-success-rate, and audio quality together —
and checks whether it shifts the trade-off curve Stage 3 established, or merely moves along
it. A threshold for what counts as success is meant to be fixed before that run, not after.

**Status: pending Stage 4.** Its key finding and the hypothesis it tests will depend on which
direction is chosen above.

---

## Before Stage 4 begins

The Stage 3 numbers above are a trend pass (n=20 per point), not the full n=100 with paired
per-utterance statistics this project otherwise uses throughout. A clean re-run to get that —
now that the environment split between the two evaluation stages is a known, working
procedure — is cheap. That re-run is worth doing, but it makes more sense to fold it together
with whichever Stage 4 direction gets chosen than to run it twice. Holding off on scheduling
it until that decision is made.

---

## Limitations

- **Zero-shot threat model only.** No fine-tuning-based cloning attack is evaluated.
- **Protection is YourTTS-surrogate-specific below Stage 4.** It transfers black-box to
  XTTS-v2 but not, at the original operating point, to F5-TTS — the exact gap Stage 4
  targets.
- **No real SafeSpeech baseline in this harness** — all "vs SafeSpeech" comparisons use
  their published numbers, same encoder and threshold, corpus/model differences stated as a
  caveat throughout.
- **No compression-robustness arm** (MP3/Opus) alongside the denoising attack already
  characterised.
- **No subjective listening test (SMOS).** All quality evidence is objective (PESQ/STOI/SI-SNR).
- **Single corpus** — LibriSpeech only; VoiceMark itself trained on VCTK.

---

## References

VoiceMark ([Interspeech 2025](https://www.isca-archive.org/interspeech_2025/li25g_interspeech.pdf)) ·
SafeSpeech ([USENIX Security 2025](https://www.usenix.org/system/files/usenixsecurity25-zhang-zhisheng.pdf)) ·
Dual Defense ([IEEE TIFS](https://arxiv.org/abs/2310.16540)) ·
AudioPure · ECAPA-TDNN (speechbrain)

Full experimental record, including negative results and withdrawn claims: `docs/experimental_writeup.md`
