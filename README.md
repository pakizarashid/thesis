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

VoiceMark's own released watermark, evaluated across five zero-shot TTS architectures: the three reported in the VoiceMark paper (CosyVoice, F5-TTS, MaskGCT) plus YourTTS and XTTS-v2, using the same detector, the same 16-bit payload, and the same harness
throughout.

| Cloner | Speaker conditioning | ↑ Watermark ACC |
|---|---|---|
| YourTTS | fixed d-vector (single speaker embedding) | 0.5337 |
| XTTS-v2 | GPT audio-prompt tokens (discretised) | 0.6119 |
| CosyVoice 2 | flow-matching decoder, regenerated from prompt mel | 0.7669 |
| MaskGCT | masked infilling, quantised RVQ tokens retained in-context | 0.9137 |
| F5-TTS | mel infilling, reference mel retained | 0.9300 |

**Finding: watermark survival is architecture/reference-pathway dependent, not a fixed property of the watermark.**
> The same VoiceMark watermark produced substantially different attribution accuracy across the five zero-shot TTS architectures, with lower ACC on YourTTS/XTTS-v2 and higher ACC on F5-TTS/MaskGCT.

The result is consistent with the architectures' different reference-conditioning pathways: systems that preserve more reference acoustic structure tend to retain more watermark information than systems that compress or regenerate the speaker information more aggressively. This interpretation is further supported by CARRIER-PROBE, which found the same cross-architecture ordering in the raw watermark-bearing latent representation across the reliably measured architectures..

**Status: done**

---

## Stage 2 — Anti-cloning: does the protection transfer to the same five architectures?

A PGD-style adversarial perturbation, optimised against a differentiable YourTTS surrogate,
layered on top of the watermark.

On the architecture it was built for (YourTTS), denoising-attack scenario, n=100:

| Metric | Clean | Protected | after DEMUCS |
|---|---|---|---|
| Speaker similarity (ECAPA-TDNN) ↓ | 0.4400 | 0.1468 | 0.1953 |
| Attack success rate (SIM > 0.25) ↓| 95.0% | 17.0% | 34.0% |
| Watermark ACC ↑ | 0.9931 | 1.0000 | 0.9844–0.9950 |

Both objectives compose at zero cost here (adding disruption costs nothing in watermark
survival, p = 0.75), and using the project's ECAPA-TDNN-based evaluation protocol and SafeSpeech's SIM > 0.25 threshold, the observed protection strength is broadly comparable to the published SafeSpeech result, although direct numerical comparison is limited by corpus and cloner differences.

Measured on the other architectures:

<table>
  <thead>
    <tr>
      <th rowspan="2">Cloner</th>
      <th colspan="3">Does the YourTTS-trained perturbation transfer?</th>
    </tr>
    <tr>
      <th>Protection effect</th>
      <th>Significance / interpretation</th>
      <th>Speaker SIM ↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>XTTS-v2</strong></td>
      <td><strong>Yes</strong></td>
      <td>p = 1.9 × 10⁻⁸</td>
      <td>0.4930 → <strong>0.3747</strong></td>
    </tr>
    <tr>
      <td><strong>F5-TTS</strong></td>
      <td><strong>No</strong></td>
      <td>Attack success 91–92%; barely below the <em>unprotected</em> baseline</td>
      <td>0.41–0.43</td>
    </tr>
  </tbody>
</table>

**Finding: protection transfer is also architecture-dependent — and it fails specifically on
the architecture where attribution is strongest.** 

Attribution survives cloning on F5-TTS
(ACC 0.8381 protected, 0.8187 after DEMUCS — a real but modest ~0.09 cost from protection
itself), but the perturbation that works on YourTTS and transfers to XTTS-v2 essentially does
nothing to F5-TTS's cloning success. 

>The two halves of the defense decouple by architecture.

**Status: done**

---

## Stage 3 — Epsilon sweep: does a bigger perturbation budget buy back protection on F5-TTS?

Same YourTTS-trained PGD objective, perturbation budget raised well past the original
operating point (ε = 0.002), re-measured on F5-TTS.

| ε | attribution ACC (F5-TTS clone) ↑ | SIM mean ↓ | SIM median ↓ | attack success rate ↓ |
|---|---|---|---|---|
| 0.002 (original operating point) | 0.8381 | 0.4140 | 0.4154 | 91.0% |
| 0.01 | 0.7000 | 0.3250 | 0.3096 | 75.0% |
| 0.02 | 0.6844 | 0.2713 | 0.2708 | 55.0% |
| 0.04 | 0.6062 | 0.1605 | 0.1573 | 15.0% |
| 0.08 | 0.5875 | 0.0969 | 0.1010 | 5.0% |

Audio quality at the same budgets, measured independently:

| ε | PESQ ↑ | STOI ↑ | SI-SNR ↑ |
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
to 0.10, attack success from 91% to 5% — but not for free.** 
> Attribution degrades toward
chance in parallel (0.84 → 0.59), and, independently, audio quality collapses: PESQ is near
its floor by ε = 0.04, and by ε = 0.08 the perturbation is on average louder than the speech
itself (negative SI-SNR).

There is no ε in this range where protection, attribution, and
usable audio all hold at once — it's a three-way trade-off across the full swept range, not a clean case of the protection simply not working on F5-TTS.
 
**Status: done** (trend pass, n=20 per point above ε = 0.002; the ε = 0.002 row is the
earlier n=100 result).

---

## Stage 4 — Route 2: training the watermark encoder jointly with the detector, cloning inside the loop

The direction chosen from Stage 4's three candidates was the third: retraining the watermark
itself with the cloning operation inside the training loop, rather than reshaping the
perturbation objective or building a second surrogate cloner. LoRA adapters on top of the
frozen pretrained backbone are trained against a joint loss (clean-audio detection + detection
on a differentiable YourTTS clone of the watermarked audio), so the watermark itself learns to
leave a signal that survives being cloned — not just a perturbation reacting to a fixed
watermark.

**Checkpoint provenance note:** the original Stage-1 checkpoint this work was meant to start
from (`checkpoints/stage1_aug/`) was never committed to git and was lost when its Kaggle
session ended. All Route 2 results below start instead from `checkpoints/stage1_scaleup_aug/`
(augmentation + the full 900-utterance scaled-up data — a different training run, git-tracked).
Its own baseline XTTS numbers (measured fresh, both trained and untrained) are used as the
Route 2 control throughout, rather than mixing in Stage 1–3's numbers, which used a different
Stage-1 lineage.

### Detector-only baseline (control)

Training only the detector's LoRA adapters (msg_processor frozen), 20 epochs, same
`stage1_scaleup_aug` base checkpoint used throughout this stage. Held-out XTTS, n=100:

| | ACC | SIM |
|---|---|---|
| untrained (`stage1_scaleup_aug`, zero LoRA training) | 0.5537 | 0.4900 |
| detector-only trained | 0.6031 | 0.4908 |

Paired significance (same 100 utterances, matched ordering): ACC p = 0.0031 (t), p = 0.0050
(Wilcoxon) — training the detector alone gives a real, if modest, attribution gain.

### Adding msg_processor to the trainable set

Same setup, but msg_processor's LoRA adapters are unfrozen too (`--train_msg_processor`),
20 epochs, λ_clone = 1.0:

| | ACC | SIM | PESQ | STOI | SI-SNR |
|---|---|---|---|---|---|
| detector-only | 0.6031 | 0.4908 | — | — | — |
| + msg_processor (20 epochs) | 0.6913 | 0.3939 | 1.963 | 0.888 | 3.10 dB |

Training the watermark encoder jointly with the detector gives a substantially larger
attribution gain than detector-only training — but at a real cost: SIM against the
detector-only checkpoint drops by ~0.10 (paired, highly significant), and PESQ is lower than
this project's own untouched-embedder checkpoints. The gain and the cost are the same
phenomenon: touching msg_processor's LoRA changes what the watermarked audio sounds like,
which is exactly what both the attribution improvement and the quality/SIM cost trace back to.

### Chasing the quality/SIM cost: three independent nulls

Three separate levers were tried against the PESQ/SIM cost, each testing a different
hypothesis for what was driving it:

| lever tested | result vs. the untouched run |
|---|---|
| `--epochs` (early-stopped at epoch 9 vs. the full 20) | not significant (ACC p = 0.235, SIM p = 0.183) |
| `--lambda_clone` (0.5 vs. 1.0) | not significant (ACC p = 0.907, SIM p = 0.607) |
| `--msgproc_lora_r` (rank 2 vs. the default rank 8) | not significant (ACC p = 0.205, SIM p = 0.438, vs. the closest rank-8 match) |

None of training duration, clone-loss weighting, or LoRA capacity itself moves the quality/SIM
cost. Taken together, this is a well-powered negative result: the cost looks structural to
training msg_processor's adapters at all, not tunable via any of these three independent
levers.

**Bonus finding from the rank sweep:** `--msgproc_lora_r 2` — trained from LoRA zero-init
(no Stage-1 warm start, since a rank-2 adapter can't reuse rank-8 weights) — reached the best
ACC of any Route 2 variant (0.7288, n=100 XTTS), a real improvement over the original rank-8
checkpoint (paired p = 0.011), with ~25% fewer trainable parameters (221K vs. 295K). It is
statistically indistinguishable from the rank-8 checkpoint on ACC (p = 0.205) and SIM
(p = 0.438) — the two are practical equivalents, rank 2 just gets there cheaper and without
needing the Stage-1 warm start.

### Generalization: dataset and cloner architecture

**VCTK (fully speaker-disjoint from the LibriSpeech training data), n=100, unprotected:**

| checkpoint | ACC | SIM |
|---|---|---|
| detector-only | 0.6169 | 0.5317 |
| + msg_processor (rank 8, epoch 9) | 0.6913 | 0.4475 |
| + msg_processor (rank 2) | 0.7006 | 0.4457 |

Both the ACC gain and the SIM cost from training msg_processor replicate on a second,
independent dataset (rank-8 vs. detector-only: p < 0.00001 both metrics; rank-2 vs.
detector-only: p < 0.00001 both metrics; rank-2 vs. rank-8: not significant, p = 0.49/0.71 —
consistent with the rank-8/rank-2 equivalence found above).

**Cross-cloner (F5-TTS, n=100), the msg_processor checkpoints only:**

| checkpoint | ACC on F5-TTS clone |
|---|---|
| rank 8 (epoch 9) | 0.9875 |
| rank 2 | 0.9881 |
| VoiceMark published | 0.979 |

Both checkpoints land at or slightly above VoiceMark's own published number and do not
regress relative to this project's own earlier F5-TTS baseline (0.9300) — unsurprising, since
F5-TTS was already near-ceiling before Route 2, but confirms training msg_processor doesn't
cost anything on the architecture where attribution was already easiest.

**Cross-cloner (CosyVoice, n=98/100 — 2 utterances skipped when CosyVoice's own text
normalizer crashed on an unusual whisper transcript, an unrelated cloner-side issue handled by
skipping and continuing), the rank-2 checkpoint:**

| checkpoint | ACC on CosyVoice clone |
|---|---|
| pretrained VoiceMark (zero-init LoRA) | 0.7669 |
| rank 2 (Route 2) | 0.8801 |
| VoiceMark published | 0.964 |

Unlike F5-TTS, CosyVoice was not near-ceiling before Route 2: training msg_processor moves ACC
from 0.767 to 0.880 on this architecture, a real generalization of the gain already seen on
LibriSpeech/XTTS and VCTK to a third, independent cloner. Still below VoiceMark's own published
0.964, landing in the "gradient, not two clusters" middle of the conditioning-bandwidth ladder
rather than at the high-bandwidth end with F5-TTS.

**MaskGCT cross-cloner validation for the Route 2 checkpoints: not yet run** — next step, same
harness, same checkpoints already pushed.

### Composability: does Stage 2/3's anti-cloning PGD still transfer, on a Route 2 checkpoint?

PGD optimised against the differentiable YourTTS surrogate (ε = 0.002, λ_wm = 1.0, matching
Stage 2's original operating point), then the protected audio cloned through the real,
non-differentiable XTTS-v2 — the same held-out transfer test as Stage 2. Route 2 rank-2
checkpoint, n=100:

| | ACC | SIM |
|---|---|---|
| unprotected clone | 0.7244 | 0.3996 |
| protected clone | 0.6631 | 0.3032 |
| paired significance | p = 0.0016 | p < 0.00001 |

The disruption transfers to XTTS-v2 (SIM drop is real and large), same as Stage 2's original
finding for this architecture. Unlike Stage 2's DEMUCS result, though, watermark ACC here
takes a real, statistically significant hit under the combined attack (−0.061) rather than
holding flat — reported honestly as a real cost, not rounded up to "free." ACC remains well
above chance (0.66 vs. 0.5), so the watermark survives meaningfully, just not without cost.
F5-TTS/CosyVoice/MaskGCT composability (PGD + Route 2 checkpoint, cloned through each
architecture) has not yet been measured.

**Status: Route 2 (msg_processor training, either rank) is the current leading candidate for
Stage 4's contribution.** Detector-only vs. +msg_processor is a settled, well-replicated
finding (two datasets). The quality/SIM cost is a settled negative result (three levers ruled
out). Rank-2 is the current best checkpoint (best ACC, fewest parameters) and is statistically
equivalent to rank-8 everywhere it's been tested. CosyVoice cross-cloner validation is now done
(0.767 -> 0.880, a real generalization of the Route 2 gain to a third cloner). Remaining before
this stage can be called complete: MaskGCT cross-cloner validation, and composability against
CosyVoice/MaskGCT/F5-TTS.

---

## Stage 5 — Final evaluation

**Status: in progress, not pending.** Route 2 (Stage 4's chosen direction) already has
real head-to-head numbers against the detector-only control across two datasets (LibriSpeech/
XTTS, VCTK) and one additional cloner architecture (F5-TTS); what remains is finishing that
same protocol against CosyVoice and MaskGCT, and re-measuring composability (PGD + Route 2
checkpoint, held-out cloning) across the full five-architecture set rather than XTTS alone. A
success threshold for the overall dual-defense claim (attribution ACC, SIM/attack-success-rate,
and audio quality together) is still to be fixed explicitly once that full table exists.

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
