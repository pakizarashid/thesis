# Dual-Defense Audio Protection: Watermarking (Traceability) + Perturbation (Anti-Cloning)

Combines **VoiceMark-style traceable watermarking** with **SafeSpeech-style adversarial
perturbation--, and study whether a zero-shot voice clone can be made harder to impersonate while the source remains traceable.

**Core question:** attribution needs speaker information to *survive* cloning; anti-cloning
protection needs it *destroyed*. Can one system do both — and if not everywhere, exactly
where does it break, and can that be fixed?

### Main research goal:
**Can a watermark-bearing adversarial protection system reduce usable speaker information and clone usability, while preserving watermark-based attribution and acceptable quality of the protected source audio, across heterogeneous zero-shot TTS architectures?**

The system has two complementary jobs:

```text
Protected speech
      │
      ├── Watermark  →  trace / attribute the source
      │
      └── Perturbation →  disrupt cloning and reduce usable speaker information
                               │
                               ↓
                         Zero-shot TTS
                               │
                    ┌──────────┴──────────┐
                    ↓                     ↓
              lower speaker SIM       watermark survives
                    ↓                     ↓
             harder impersonation      traceability
```

A further objective is to reduce **clone usability/intelligibility** where measurable (e.g. higher WER), while keeping the original protected speech usable to a human listener.

---

### Notes

This project have:
- **Reproduction:** same published setup where feasible.
- **Independent evaluation:** controlled experiments using this project's datasets, samples, or additional TTS architectures.
- **Our contribution:** comparisons made under the same conditions between our baseline and our proposed modification.

> The work below is a five-stage progression. Stages 1–3 are complete and their findings are
final. Stage 4 is the open decision — three candidate directions are sized below, none
started yet. Stage 5 depends on which direction Stage 4 takes.

---
# Research progression

## Stage 1 — Watermark 
> **Does the watermark survive being cloned?**

VoiceMark's released watermark was evaluated on five zero-shot TTS architectures: the three used in the VoiceMark paper (CosyVoice, F5-TTS, MaskGCT) plus YourTTS and XTTS-v2, using the same detector, the same 16-bit payload, and the same harness throughout.

| Cloner      | Main reference-conditioning path                           | Watermark ACC ↑            |
| ----------- | ---------------------------------------------------------- | -------------------------- |
| YourTTS     | fixed d-vector (single speaker embedding)                  | 0.5337                     |
| XTTS-v2     | GPT audio-prompt tokens (discretised)                      | 0.6119                     |
| CosyVoice 2 | flow-matching decoder, regenerated from prompt mel         | 0.7669                     |
| MaskGCT     | masked infilling, quantised RVQ tokens retained in-context | 0.9137                     |
| F5-TTS      | mel infilling, reference mel retained                      | 0.9300                     |

### Finding
> watermark survival is architecture/reference-pathway dependent, not a fixed property of the watermark.

The same watermark produced substantially different attribution across cloning architectures. This motivates the hypothesis that watermark survival depends on the **reference transformation/conditioning pathway**, rather than being a fixed property of the watermark alone.

Across all five architectures (monotone, no ties/inversions), survival is higher when the cloner retains reference audio (F5-TTS, MaskGCT) rather than regenerating it through a decoder (CosyVoice) — consistent with VoiceMark's own published numbers, whose eval set is entirely retained-conditioning.

**CARRIER-PROBE** confirms this at the latent level: re-encoded clone audio vs. the original watermark carrier shows the same ordering on 4/5 architectures (CosyVoice excluded — clone/reference durations didn't align closely enough to compare), across all seven RVQ layers (n=15/architecture). Layer 2 is the most fragile; an inference-time attempt to reweight/remove it didn't improve attribution, so this stays a diagnostic, not a fix.

**Status: completed**

---

## Stage 2 — Anti-cloning protection
> **Does the protection transfer to the same five architectures?**

### Mechanism
The main anti-cloning is **waveform-domain PGD**. The protected, decoded waveform is perturbed directly:
```text
watermarked waveform + bounded δ → protected waveform → zero-shot TTS → clone
```
optimized against a differentiable **YourTTS surrogate** — the only differentiable target available; every other cloner is evaluated black-box.

On YourTTS with ECAPA-TDNN similarity scoring, denoising-attack scenario, n=100:

| Metric                             | Clean  | Protected | After DEMUCS  |
| ---------------------------------- | ------ | --------- | ------------- |
| Speaker similarity (ECAPA-TDNN) ↓  | 0.4400 | 0.1468    | 0.1953        |
| Attack success rate (SIM > 0.25) ↓ | 95%    | 17%       | 34%           |
| Watermark ACC ↑                    | 0.9931 | 1.0000    | 0.9844–0.9950 |

Both disruption objectives composes with attribution at zero cost here (adding disruption costs nothing in watermark
survival, p = 0.75), and this is comparable to or better than SafeSpeech's own published protection strength at the same threshold.


### Transfer
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

### Finding
> protection transfer is also architecture-dependent, and it fails specifically on the architecture where attribution is strongest.

Attribution survives cloning on F5-TTS (ACC 0.8381 protected, 0.8187 after DEMUCS — a real but modest ~0.09 cost from protection itself), but the perturbation that works on YourTTS and transfers to XTTS-v2 essentially does nothing to F5-TTS's cloning success. 

> The two halves of the defense decouple by architecture.

**Status: completed**

---

## Stage 3 — Epsilon sweep
> **Protection/attribution/quality trade-off**

Same YourTTS-trained PGD objective, an epsilon sweep showed that stronger waveform perturbation eventually reduces speaker similarity, but simultaneously damages watermark attribution and source-audio quality.

| ε     | F5 clone WM ACC ↑ | SIM ↓ | ASR ↓ | PESQ ↑ | STOI ↑ | SI-SNR ↑ |
|---:   |---:    |---:    |---:  |---:  |---:   |---:      |
| 0.002 | 0.8381 | 0.4140 | 91% | 1.919 | 0.885 | 0.43 dB  |
| 0.01  | 0.7000 | 0.3250 | 75% | 1.337 | 0.834 | 0.24 dB  |
| 0.02  | 0.6844 | 0.2713 | 55% | 1.155 | 0.784 | -0.28 dB |
| 0.04  | 0.6062 | 0.1605 | 15% | 1.069 | 0.708 | -1.77 dB |
| 0.08  | 0.5875 | 0.0969 | 5%  | 1.042 | 0.610 | -4.95 dB |

The surrogate's own architecture keeps working throughout; it's specifically the transfer to F5-TTS that's in question.

| ε     | ourTTS WM ACC ↑ | After DEMUCS | 
|---:   |---:    |---:    |
| 0.002 | 0.94   | 0.91 | 
| 0.08  | 0.99   | 0.62 | 

| Protected ACC	stays 0.94–0.99
| Protected → DEMUCS ACC	0.91 → 0.62

### Finding
> increasing ε does buy back protection on F5-TTS — SIM falls from 0.41
to 0.10, attack success from 91% to 5% — but not for free

Attribution degrades toward chance in parallel (ACC 0.84 → 0.59), and audio quality collapses independently (PESQ near floor by ε = 0.04; SI-SNR negative by ε = 0.08 — perturbation louder than the speech itself). There is no ε in this range where protection, attribution, and usable audio all hold at once — a genuine three-way **trade-off**.

Two later attempts to fix this by reformulating the PGD objective (H-SPEC, H-DIRECT) came
back null/adverse and are closed.

**Status: completed** 

---

## Stage 4 — Clone-aware watermark adaptation 
> **route2 - training the watermark encoder jointly with the detector, cloning inside the loop**

The direction chosen from Stage 4's three candidates was the third: retraining the watermark
itself with the cloning operation inside the training loop, rather than reshaping the
perturbation objective or building a second surrogate cloner. LoRA adapters on top of the
frozen pretrained backbone are trained against a joint loss (clean-audio detection + detection
on a differentiable YourTTS clone of the watermarked audio), so the watermark itself learns to
leave a signal that survives being cloned — not just a perturbation reacting to a fixed
watermark.

**Checkpoint provenance note:** the original Stage-1 checkpoint this work was meant to start
from (`checkpoints/stage1_aug/`) was never committed to git and was lost when its Kaggle
session ended. All Route 2 results below start instead from `checkpoints/stage1_scaleup_aug/` (augmentation + the full 900-utterance scaled-up data — a different training run, git-tracked).
Its own baseline XTTS numbers (measured fresh, both trained and untrained) are used as the
Route 2 control throughout, rather than mixing in Stage 1–3's numbers, which used a different
Stage-1 lineage.

### Detector-only baseline (control)

Training only the detector's LoRA adapters (msg_processor frozen), 20 epochs, same `stage1_scaleup_aug` base checkpoint used throughout this stage. Held-out XTTS, n=100:

|                                                      | ACC    | SIM    |
| ---------------------------------------------------- | ------ | ------ |
| untrained (`stage1_scaleup_aug`, zero LoRA training) | 0.5537 | 0.4900 |
| detector-only trained                                | 0.6031 | 0.4908 |

Paired significance (same 100 utterances, matched ordering): ACC p = 0.0031 (t), p = 0.0050
(Wilcoxon) — training the detector alone gives a real, if modest, attribution gain.

### Adding msg_processor to the trainable set

Same setup, but msg_processor's LoRA adapters are unfrozen too (`--train_msg_processor`),
20 epochs, λ_clone = 1.0:

|                              | ACC    | SIM    | PESQ  | STOI  | SI-SNR  |
| ---------------------------- | ------ | ------ | ----- | ----- | ------- |
| detector-only                | 0.6031 | 0.4908 | —     | —     | —       |
| + msg\_processor (20 epochs) | 0.6913 | 0.3939 | 1.963 | 0.888 | 3.10 dB |

Training the watermark encoder jointly with the detector gives a substantially larger
attribution gain than detector-only training — but at a real cost: SIM against the
detector-only checkpoint drops by ~0.10 (paired, highly significant), and PESQ is lower than
this project's own untouched-embedder checkpoints. The gain and the cost are the same
phenomenon: touching msg_processor's LoRA changes what the watermarked audio sounds like,
which is exactly what both the attribution improvement and the quality/SIM cost trace back to.

### Chasing the quality/SIM cost: three independent nulls

Three separate levers were tried against the PESQ/SIM cost, each testing a different
hypothesis for what was driving it:

| lever tested                                          | result vs. the untouched run                                                 |
| ------------------------------------------------------ | ------------------------------------------------------------------------------ |
| `--epochs` (early-stopped at epoch 9 vs. the full 20) | not significant (ACC p = 0.235, SIM p = 0.183)                               |
| `--lambda_clone` (0.5 vs. 1.0)                          | not significant (ACC p = 0.907, SIM p = 0.607)                               |
| `--msgproc_lora_r` (rank 2 vs. the default rank 8)     | not significant (ACC p = 0.205, SIM p = 0.438, vs. the closest rank-8 match) |

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

| checkpoint                         | ACC    | SIM    |
| ------------------------------------ | ------ | ------ |
| detector-only                      | 0.6169 | 0.5317 |
| + msg\_processor (rank 8, epoch 9) | 0.6913 | 0.4475 |
| + msg\_processor (rank 2)          | 0.7006 | 0.4457 |

Both the ACC gain and the SIM cost from training msg_processor replicate on a second,
independent dataset (rank-8 vs. detector-only: p < 0.00001 both metrics; rank-2 vs.
detector-only: p < 0.00001 both metrics; rank-2 vs. rank-8: not significant, p = 0.49/0.71 —
consistent with the rank-8/rank-2 equivalence found above).

**Cross-cloner (F5-TTS, n=100), the msg_processor checkpoints only:**

| checkpoint          | ACC on F5-TTS clone |
| -------------------- | --------------------- |
| rank 8 (epoch 9)    | 0.9875               |
| rank 2              | 0.9881               |
| VoiceMark published | 0.979                |

Both checkpoints land at or slightly above VoiceMark's own published number and do not
regress relative to this project's own earlier F5-TTS baseline (0.9300) — unsurprising, since
F5-TTS was already near-ceiling before Route 2, but confirms training msg_processor doesn't
cost anything on the architecture where attribution was already easiest.

**CosyVoice and MaskGCT cross-cloner validation for the Route 2 checkpoints (2026-09-15):**
measured as a byproduct of the composability run below (each cloner's *unprotected* arm is
exactly this measurement — watermark survival through real cloning, no PGD involved). MaskGCT:
ACC 0.9481, n=100, rank-2 checkpoint — a real, clean result. CosyVoice: ACC 0.8801, n=100 —
**not treated as reliable** (see the Composability section below for why: the underlying
clone audio shows signs of not being genuine cloned speech for a large fraction of samples,
so this ACC number is reported for completeness but should not be read as validated watermark
survival through real CosyVoice cloning).

### Composability: does Stage 2/3's anti-cloning PGD still transfer, on a Route 2 checkpoint?

PGD optimised against the differentiable YourTTS surrogate (ε = 0.002, λ_wm = 1.0, matching
Stage 2's original operating point), then the protected audio cloned through the real,
non-differentiable XTTS-v2 — the same held-out transfer test as Stage 2. Route 2 rank-2
checkpoint, n=100:

|                     | ACC        | SIM         |
| -------------------- | ------------ | ------------- |
| unprotected clone   | 0.7244     | 0.3996      |
| protected clone     | 0.6631     | 0.3032      |
| paired significance | p = 0.0016 | p < 0.00001 |

The disruption transfers to XTTS-v2 (SIM drop is real and large), same as Stage 2's original
finding for this architecture. Unlike Stage 2's DEMUCS result, though, watermark ACC here
takes a real, statistically significant hit under the combined attack (−0.061) rather than
holding flat — reported honestly as a real cost, not rounded up to "free." ACC remains well
above chance (0.66 vs. 0.5), so the watermark survives meaningfully, just not without cost.

**F5-TTS and MaskGCT composability (2026-09-15) — the same protocol extended to two more
architectures.** Same PGD operating point (ε=0.002, λ_wm=1.0), same Route 2 rank-2 checkpoint,
n=100 held-out LibriSpeech utterances drawn via a dedicated 3-arm generation script
(reference/unprotected/protected WAVs per utterance, shared identically across all three
cloners tested):

| cloner  | ACC unprotected → protected | paired significance                        | SIM ASR unprotected → protected (ECAPA-TDNN, threshold 0.25) |
| ------- | ---------------------------- | -------------------------------------------- | --------------------------------------------------------------- |
| F5-TTS  | 0.9844 → 0.9450             | p = 4.7×10⁻⁶ (t), p = 2.3×10⁻⁵ (Wilcoxon) | 87.0% → 72.0%                                                  |
| MaskGCT | 0.9481 → 0.8794             | p = 7.7×10⁻⁶ (t), p = 2.3×10⁻⁵ (Wilcoxon) | 71.0% → 51.0%                                                  |

(F5-TTS's unprotected ACC here, 0.9844, is a second, independent measurement from the 0.9881
in the cross-cloner table above — different utterance draw, same checkpoint — the two agree
closely.) Quality of the shared protected source at this budget: PESQ 1.787, STOI 0.864,
SI-SNR 0.35 dB, consistent with Stage 3's own ε=0.002 quality numbers measured on a different
checkpoint (PESQ 1.919, STOI 0.885, SI-SNR 0.43 dB) — Route 2 isn't buying this composability
result with a louder or quieter perturbation than already characterised.

**Finding: protection composes with attribution — both significantly reduced together — on
two more held-out architectures beyond XTTS-v2**, extending the composability result above
rather than repeating it. Effect size differs by cloner (MaskGCT's ACC drop is larger than
F5-TTS's, and both are smaller than the corresponding SIM-ASR reduction), but the direction
and significance hold on both.

**CosyVoice: attempted, excluded.** Its ACC-only number looks ordinary (0.8801 → 0.7191,
p = 3.7×10⁻¹³, the largest and most significant drop of the three cloners) — but its SIM
collapsed even in the *unprotected* arm (6.1% attack success, mean SIM 0.128, against
F5-TTS/MaskGCT's 87%/71% on this identical protocol and SafeSpeech's own ~60% unprotected
reference point). Three checks, each targeting a different candidate explanation, converged on
one root cause:

1. **Audio sanity** — reference/clone duration and sample rate were normal (3.00s reference,
   7.20s clone, 16kHz throughout); nothing malformed in our own pipeline.
2. **Control run** — ECAPA SIM between the true reference and the *watermarked-but-uncloned*
   audio (no cloning involved) came back normal (100%/94% attack success unprotected/
   protected, means 0.538/0.432) — ruling out a bug in our watermarking, PGD, or SIM-scoring
   code.
3. **Intelligibility check** — transcribing the actual CosyVoice clones against the known
   target text ("This is a test sentence for voice cloning.") gave a mean WER of 1.43–1.49
   (worse than chance), dominated by empty transcriptions, non-English tokens, and
   non-terminating repetition loops ("go, go, go, go...", "I don't know. I don't know. I
   don't know."), echoed in generation logs showing a suspiciously fixed ~7.2s output length
   regardless of content — the classic signature of an autoregressive decoder failing to hit
   its stop condition.

The reference transcripts themselves (obtained via the same faster-whisper pathway used
identically for all three cloners) were verified legitimate — real, on-topic, just naturally
truncated mid-utterance, since each is a 3-second crop of continuous speech rather than a
clean sentence boundary. F5-TTS and MaskGCT clone these same crops and transcripts without
issue; CosyVoice2's zero-shot decoder appears specifically brittle to a non-sentence-final
prompt in a way the other two architectures are not. This is reported as an excluded,
root-caused protocol limitation — both the composability numbers above and the cross-cloner
ACC in the table further up are built on audio that likely isn't genuine cloned speech for a
large share of samples — not folded into either the composability comparison or the
cross-cloner validation table as if it were a clean measurement.

**Status: Route 2 (msg_processor training, either rank) is the current leading candidate for
Stage 4's contribution.** Detector-only vs. +msg_processor is a settled, well-replicated
finding (two datasets). The quality/SIM cost is a settled negative result (three levers ruled
out). Rank-2 is the current best checkpoint (best ACC, fewest parameters) and is statistically
equivalent to rank-8 everywhere it's been tested. F5-TTS and MaskGCT cross-cloner validation
and composability are both now measured and positive. **Remaining before this stage can be
called complete: a reliable CosyVoice measurement** (the current attempt is excluded for a
stated, architecture-specific reason — see above — not a defense-side finding either way).

---

## Stage 5 — Final evaluation

**Status: in progress, not pending.** Route 2 (Stage 4's chosen direction) now has real
head-to-head numbers against the detector-only control across two datasets (LibriSpeech/XTTS,
VCTK), and both cross-cloner validation and PGD composability across three additional cloner
architectures (F5-TTS, MaskGCT, and an excluded, root-caused CosyVoice attempt), alongside the
original XTTS-v2 composability result. What remains is a reliable CosyVoice measurement
(current attempt excluded for a stated architecture-specific reason — a longer or
sentence-aligned reference crop might resolve it, not yet attempted) and fixing an explicit
success threshold for the overall dual-defense claim now that F5-TTS and MaskGCT composability
numbers exist alongside XTTS-v2's.

---

### Metrics: what each one means

| Goal | Metric | Desired direction |
|---|---|---|
| Preserve traceability | Watermark ACC | ↑ |
| Destroy speaker identity | ECAPA speaker SIM | ↓ |
| Reduce clone usability/intelligibility | WER / ASR | ↑ |
| Preserve protected-source quality | PESQ / STOI / SI-SNR | ↑ |

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
- **CosyVoice2 zero-shot cloning is unreliable under this protocol's 3-second, non-sentence-
  aligned reference crops** — confirmed via a control experiment and WER check, not fixed
  (see Stage 4's Composability section). Affects both the cross-cloner validation and
  composability measurements for this one architecture only.
  
- The strongest remaining limitation is that the anti-cloning perturbation is optimized through a **single differentiable YourTTS surrogate**, so transfer strength varies by target architecture. The project therefore does not claim universal protection against all future TTS systems.
  
---

## References

- VoiceMark ([Interspeech 2025](https://www.isca-archive.org/interspeech_2025/li25g_interspeech.pdf))  — speaker-specific latent watermarking for zero-shot voice-cloning resistance.
- SafeSpeech ([USENIX Security 2025](https://www.usenix.org/system/files/usenixsecurity25-zhang-zhisheng.pdf)) — proactive adversarial protection against voice cloning.
- Dual Defense ([IEEE TIFS](https://arxiv.org/abs/2310.16540))
- AudioPure — diffusion-based audio purification attack.
- ECAPA-TDNN ECAPA-TDNN ([speechbrain](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb))  — speaker-verification encoder used for SafeSpeech-comparable similarity evaluation.
  
Full experimental record, including negative results and withdrawn claims: `docs/experimental_writeup.md`

---

## Stage 4  

**At a glance — what's done, in one table:**

| Question | Result | Detail below |
|---|---|---|
| Does training msg_processor jointly beat detector-only training? | Yes — 0.603 → 0.691 ACC (held-out XTTS, n=100), real and significant | "Adding msg_processor" |
| Is there a cost? | Yes — SIM drops ~0.10, PESQ lower. Real, not tunable away (3 levers ruled out) | "Chasing the quality/SIM cost" |
| Which checkpoint is best? | Rank-2 LoRA on msg_processor — best ACC (0.729), 25% fewer params, no warm-start needed, statistically tied with rank-8 everywhere | "Bonus finding" |
| Does the gain replicate on a second dataset? | Yes — VCTK (speaker-disjoint), same gain and same cost pattern | "Generalization: dataset..." |
| Does the gain generalize to cloners never seen in training? | Yes, on all 4 tested — F5-TTS 0.988 ↑, MaskGCT 0.959 ↑, CosyVoice 0.880 ↑, all ≥ pretrained baseline | "Generalization: ...cloner architecture" |
| Does Stage 2/3's anti-cloning PGD still work on a Route 2 checkpoint? | Tested against XTTS-v2 only so far — disruption transfers (SIM drops), but watermark ACC takes a real hit (0.724 → 0.663), not free | "Composability" |
| What's left | Composability against CosyVoice, MaskGCT, F5-TTS (only XTTS-v2 measured) | "Composability" |



Held-out XTTS, n=100, 20 epochs, λ_clone = 1.0:

| Condition | ACC ↑ | SIM ↓ |
|---|---|---|
| untrained (`stage1_scaleup_aug`, zero LoRA training) | 0.5537 | 0.4900 |
| detector-only trained | 0.6031 | 0.4908 |

Paired significance (same 100 utterances, matched ordering): ACC p = 0.0031 (t), p = 0.0050
(Wilcoxon) — training the detector alone gives a real, if modest, attribution gain.




| Condition | ACC ↑ | SIM ↓ | PESQ ↑ | STOI ↑ | SI-SNR ↑ |
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

| Lever | ACC ↑	| SIM ↓ |	Conclusion |
|---|---|---|---|
| `--epochs` (early-stopped at epoch 9 vs. the full 20)	| p = 0.235	| p = 0.183 | No significant difference |
| `--lambda_clone` (0.5 vs. 1.0) |	p = 0.907 |	p = 0.607 |	No significant difference |
| `--msgproc_lora_r` (rank 2 vs. the default rank 8) |	p = 0.205 |	p = 0.438 |	No significant difference |

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

| checkpoint | ACC ↑ | SIM ↓ |
|---|---|---|
| detector-only | 0.6169 | 0.5317 |
| + msg_processor (rank 8, epoch 9) | 0.6913 | 0.4475 |
| + msg_processor (rank 2) | 0.7006 | 0.4457 |

Both the ACC gain and the SIM cost from training msg_processor replicate on a second,
independent dataset (rank-8 vs. detector-only: p < 0.00001 both metrics; rank-2 vs.
detector-only: p < 0.00001 both metrics; rank-2 vs. rank-8: not significant, p = 0.49/0.71 —
consistent with the rank-8/rank-2 equivalence found above).

| Checkpoint / rank |	XTTS ACC ↑	| Trainable parameters ↓ |
|---|---|---|
| Rank 8 (epoch 9) |	— |	295K |
| Rank 2	| 0.7288 |	221K|

**Cross-cloner (F5-TTS, n=100), the msg_processor checkpoints only:**

| checkpoint | F5-TTS clone ACC ↑ |
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

**Cross-cloner (MaskGCT, n=100, zero skipped), the rank-2 checkpoint:**

| checkpoint | ACC on MaskGCT clone ↑ |
|---|---|
| pretrained VoiceMark (zero-init LoRA) | 0.9138 |
| rank 2 (Route 2) | 0.9594 |
| VoiceMark published | 0.957 |

MaskGCT was already near-ceiling before Route 2, like F5-TTS: training msg_processor moves ACC
from 0.914 to 0.959, at or slightly above VoiceMark's own published number, and lands with
F5-TTS at the high-bandwidth end of the conditioning ladder. Cross-cloner validation for the
Route 2 rank-2 checkpoint is now complete across all four architectures (YourTTS/XTTS via the
LibriSpeech eval, F5-TTS, CosyVoice, MaskGCT).

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
architecture) has not yet been measured — the XTTS-v2 result above is the only composability
number so far.

**Status: Route 2 (msg_processor training, rank 2) is Stage 4's contribution.** Detector-only
vs. +msg_processor is a settled, well-replicated finding (two datasets). The quality/SIM cost
is a settled negative result (three levers ruled out). Rank-2 is the best checkpoint (best ACC,
fewest parameters) and statistically tied with rank-8 everywhere it's been tested. Cross-cloner
validation is complete across all four architectures (0.767 → 0.880 on CosyVoice, 0.914 → 0.959
on MaskGCT, both real generalizations of the Route 2 gain, not an F5-TTS-only effect). Only
thing left before this stage is fully closed out: composability (PGD + Route 2 checkpoint)
against CosyVoice, MaskGCT and F5-TTS — currently measured only against XTTS-v2.

---
