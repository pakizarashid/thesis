# Dual-Defense Audio Protection: Watermarking (Traceability) + Perturbation (Anti-Cloning)

Combines **VoiceMark-style traceable watermarking** with **SafeSpeech-style adversarial perturbation**, and study whether a zero-shot voice clone can be made harder to impersonate while the source remains traceable.

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

| Protected ACC	stays 0.94–0.99 |

| Protected → DEMUCS ACC	0.91 → 0.62|

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

The direction chosen from Stage 4's three candidates was the third: retraining the watermark itself with the cloning operation inside the training loop, rather than reshaping the perturbation objective or building a second surrogate cloner. 
LoRA adapters on top of the frozen pretrained backbone are trained against a joint loss (clean-audio detection + detection on a differentiable YourTTS clone of the watermarked audio), so the watermark itself learns to leave a signal that survives being cloned — not just a perturbation reacting to a fixed watermark.


This is the strongest watermarking contribution is **Route 2**: train the VoiceMark watermark encoder (`msg_processor`) and detector jointly while a differentiable YourTTS clone is inside the training loop.

LoRA is used only as the parameter-efficient implementation mechanism. **LoRA itself is not claimed as the contribution.** The contribution is the **clone-aware adaptation strategy**.

```text
   watermark encoder + detector
               │
      clone-aware training
               ↓
     differentiable YourTTS
               ↓
 learn a watermark that is easier
to recover after zero-shot cloning
```


### 1) Detector-only baseline (control)

> Held-out XTTS, n=100, 20 epochs, λ_clone = 1.0:

|                                                      | WM ACC ↑ | SIM ↓ |
| ---------------------------------------------------- | ------ | ------ |
| Untrained (zero LoRA training)                       | 0.5537 | 0.4900 |
| Detector-only trained                                | 0.6031 | 0.4908 |

Training only the detector's LoRA adapters (msg_processor frozen) alone gives a real, if modest, attribution gain @ 20 epochs
Paired significance (same 100 utterances, matched ordering): ACC p = 0.0031 (t), p = 0.0050 (Wilcoxon)

### 2) Joint encoder + detector training

> Same setup, but msg_processor's LoRA adapters are unfrozen too (`--train_msg_processor`), 20 epochs, λ_clone = 1.0:

| Condition         | WM ACC ↑   | SIM ↓  | PESQ ↑ | STOI ↑ | SI-SNR ↑ |
|-------------------|-----------:|-------:|------:|------:|--------:|
| Detector-only     | 0.6031     | 0.4908 | —     | —     | —       |
| + `msg_processor` | **0.6913** | 0.3939 | 1.963 | 0.888 | 3.10 dB |

**Improvement:** +0.0881 ACC on held-out XTTS (paired t p=0.000003; Wilcoxon p=0.000007).

The encoder itself therefore contributes additional clone-robust information; the improvement is not only a better detector reading a fixed carrier.

PESQ is lower than this project's own untouched-embedder checkpoints. The gain and the cost are the same phenomenon: touching msg_processor's LoRA changes what the watermarked audio sounds like, which is exactly what both the attribution improvement and the quality/SIM cost trace back to.

### Chasing the quality/SIM cost: three independent nulls

Three separate levers were tried against the PESQ/SIM cost, each testing a different hypothesis for what was driving it:

| Lever | ACC ↑	| SIM ↓ |	Conclusion |
|---|---|---|---|
| `--epochs` (early-stopped at epoch 9 vs. the full 20)	| p = 0.235	| p = 0.183 | No significant difference |
| `--lambda_clone` (0.5 vs. 1.0) |	p = 0.907 |	p = 0.607 |	No significant difference |
| `--msgproc_lora_r` (rank 2 vs. the default rank 8) |	p = 0.205 |	p = 0.438 |	No significant difference |

None of training duration, clone-loss weighting, or LoRA capacity itself moves the quality/SIM cost. Taken together, this is a well-powered negative result: the cost looks structural to training msg_processor's adapters at all, not tunable via any of these three independent levers.

**Bonus finding from the rank sweep:** `--msgproc_lora_r 2` — trained from LoRA zero-init
(no Stage-1 warm start, since a rank-2 adapter can't reuse rank-8 weights) — reached the best ACC of any Route 2 variant (0.7288, n=100 XTTS), a real improvement over the original rank-8
checkpoint (paired p = 0.011), with ~25% fewer trainable parameters (221K vs. 295K). It is
statistically indistinguishable from the rank-8 checkpoint on ACC (p = 0.205) and SIM
(p = 0.438) — the two are practical equivalents, rank 2 just gets there cheaper and without needing the Stage-1 warm start.

### 3) Generalization: 
> **Cross-Dataset and Cross-Cloner architecture**

n=100, rank= 2
| Evaluation| Detector-only | Joint Route 2 |       Change |
|-----------|-------:|----------------:|------------------:|
| VCTK      | 0.6169 | 0.6913 / 0.7006 | clear improvement |
| CosyVoice | 0.7669 | 0.8801          | +0.1132 |
| F5-TTS    | 0.9300 | 0.9881          | +0.0581 |
| MaskGCT   | 0.9138 | 0.9594 / 0.9481 | +0.0456 |

The Route 2 gain therefore transfers beyond the YourTTS training loop to multiple zero-shot TTS architectures.

>Detail

**VCTK (fully speaker-disjoint from the LibriSpeech training data), unprotected:**

| Checkpoint                         | ACC ↑  | SIM ↓ | 
| ---------------------------------- | ------ | ------ |
| detector-only                      | 0.6169 | 0.5317 |
| + msg\_processor (rank 8, epoch 9) | 0.6913 | 0.4475 |
| + msg\_processor (rank 2)          | 0.7006 | 0.4457 |

a) **Cross-cloner (XTTS):**
| Checkpoint        | ↑ ACC on XTTS clone | Trainable parameters ↓ |
|-------------------|---------------------|------------------------|
| Rank 8 (epoch 9)  |	                — |	              295K |
| Rank 2	        | 0.7288               |	               221K|

b) **Cross-cloner (F5-TTS), the msg_processor checkpoints only:**

| Checkpoint          |  ↑ ACC on F5-TTS clone |
| ------------------- | -------------------- |
| Rank 8 (epoch 9)    | 0.9875               |
| Rank 2              | 0.9881               |
| VoiceMark published | **0.979**            |

c) **Cross-cloner (MaskGCT):**

| Checkpoint            | ↑ ACC on MaskGCT clone |
|---------------------------------------|--------|
| pretrained VoiceMark (zero-init LoRA) | 0.9138 |
| rank 2 (Route 2)                      | 0.9594 |
| VoiceMark published                   | 0.957  |

> Both checkpoints land at or slightly above VoiceMark's own published number, and lands with F5-TTS at the high-bandwidth end of the conditioning ladder. 

d) **Cross-cloner (CosyVoice)**
> n=98/100 — 2 utterances skipped when CosyVoice's own text normalizer crashed on an unusual whisper transcript, the rank-2 checkpoint:

| Checkpoint          | ↑ ACC on CosyVoice clone |
|---------------------------------------|--------|
| pretrained VoiceMark (zero-init LoRA) | 0.7669 |
| Rank 2 (Route 2)                      | 0.8801 |
| VoiceMark published                   | 0.964  |

> Still below VoiceMark's own published 0.964, landing in the "gradient, not two clusters" middle of the conditioning-bandwidth ladder rather than at the high-bandwidth end with F5-TTS.

**CosyVoice cross-cloner validation is not treated as reliable:** 

(see the Composability section below for why: the underlying clone audio shows signs of not being genuine cloned speech for a large fraction of samples, so this ACC number is reported for completeness but should not be read as validated watermark survival through real CosyVoice cloning).

### 4) Composability
> Does Stage 2/3's anti-cloning PGD still transfer, on a Route 2 checkpoint?

PGD optimised against the differentiable YourTTS surrogate `(ε = 0.002, λ_wm = 1.0, matching Stage 2's original operating point)`, then the protected audio cloned through the real, non-differentiable XTTS-v2 — the same held-out transfer test as Stage 2. Route 2 rank-2 checkpoint, n=100:

|                     | WM ACC ↑   | SIM ↓       |
| ------------------- | ---------- | ----------- |
| unprotected clone   | 0.7244     | 0.3996      |
| protected clone     | 0.6631     | 0.3032      |
| paired significance | p = 0.0016 | p < 0.00001 |

The disruption transfers to XTTS-v2 (SIM drop is real and large), same as Stage 2's original finding for this architecture. 

Unlike Stage 2's DEMUCS result, though, watermark ACC here takes a real, statistically significant hit under the combined attack (−0.061) rather than holding flat — reported honestly as a real cost, not rounded up to "free." 

ACC remains well above chance (0.66 vs. 0.5), so the watermark survives meaningfully, just not without cost.

**F5-TTS and MaskGCT Composability**
> the same protocol extended to two more architectures.

Same PGD operating point `(ε=0.002, λ_wm=1.0)`, same Route 2 rank-2 checkpoint, n=100 held-out LibriSpeech utterances drawn via a dedicated 3-arm generation script `(reference/unprotected/protected WAVs per utterance, shared identically across all three cloners tested)`:

| Cloner  | ↑ ACC unprotected → protected | paired significance                       | ↓ SIM ASR unprotected → protected (ECAPA-TDNN, threshold 0.25) |
| ------- | ----------------------------- | ----------------------------------------- | -------------------------------------------------------------- |
| F5-TTS  | 0.9844 → 0.9450               | p = 4.7×10⁻⁶ (t), p = 2.3×10⁻⁵ (Wilcoxon) | 87.0% → 72.0%                                                  |
| MaskGCT | 0.9481 → 0.8794               | p = 7.7×10⁻⁶ (t), p = 2.3×10⁻⁵ (Wilcoxon) | 71.0% → 51.0%                                                  |

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
- **No subjective listening test (SMOS).** All quality evidence is objective (PESQ/STOI/SI-SNR).
- **Single corpus** — LibriSpeech only; VoiceMark itself trained on VCTK.

- Compression/resampling/amplitude/noise robustness now characterised (see section above) for detection accuracy and cloned-speech intelligibility on F5-TTS/Route 2; speaker similarity under the same attacks is still open.
- 
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


**Status: Route 2 (msg_processor training, rank 2) is Stage 4's contribution.** Detector-only
vs. +msg_processor is a settled, well-replicated finding (two datasets). The quality/SIM cost
is a settled negative result (three levers ruled out). Rank-2 is the best checkpoint (best ACC,
fewest parameters) and statistically tied with rank-8 everywhere it's been tested. Cross-cloner
validation is complete across all four architectures (0.767 → 0.880 on CosyVoice, 0.914 → 0.959
on MaskGCT, both real generalizations of the Route 2 gain, not an F5-TTS-only effect). Only
thing left before this stage is fully closed out: composability (PGD + Route 2 checkpoint)
against CosyVoice, MaskGCT and F5-TTS — currently measured only against XTTS-v2.

---



#******* Dual-Defense Audio Protection: Watermarking + Anti-Cloning Perturbation

---

## Robustness to Post-Processing Attacks (Route 2, F5-TTS)

Tests whether watermark detection on Route 2's dual-defense checkpoint
(`route2_scaleupaug_msgproc_r2`, msgproc_lora_r=2) survives common post-processing attacks
applied to the protected audio *before* cloning, and whether those attacks also destroy the
cloned speech's basic intelligibility — i.e., whether an attacker gains anything beyond
generic degradation of the recording.

Baselines (F5-TTS clone, n=100, same checkpoint, no post-processing):
detection ACC = 0.9431 (protected) / 0.9825 (unprotected).

Ten attack conditions (5 attack types × 2 severities — mp3/opus lossy re-encode via ffmpeg,
resample via a polyphase round-trip through an intermediate rate, amplitude a random ±dB
gain, noise additive white Gaussian at a target SNR), n=100 each:

| Attack | Severity | Detection ACC (clone) | Δ from protected baseline | Mean WER (cloned speech) |
|---|---|---|---|---|
| amplitude ±3dB | mild | 0.9406 | −0.003 | 0.021 |
| amplitude ±6dB | aggressive | 0.9400 | −0.003 | 0.026 |
| opus 64kbps | mild | 0.9400 | −0.003 | 0.016 |
| resample 16k→22.05k→16k | mild | 0.9256 | −0.017 | 0.021 |
| mp3 128kbps | mild | 0.9019 | −0.041 | 0.041 |
| mp3 32kbps | aggressive | 0.8931 | −0.050 | 0.016 |
| opus 16kbps | aggressive | 0.8725 | −0.071 | 0.013 |
| noise 20dB SNR | mild | 0.8275 | −0.116 | 0.024 |
| resample 16k→8k→16k | aggressive | 0.7488 | −0.194 | 0.022 |
| noise 10dB SNR | aggressive | 0.6569 | −0.286 | 0.018 |

**Finding: the watermark degrades gradually under compression/resampling and fails
specifically under additive noise, while the underlying speech stays almost perfectly
intelligible throughout (WER ≤ 0.041 in every condition, mostly near 0).** This decouples
watermark survival from audio quality — mp3/opus/resample/amplitude all leave clean,
transcribable speech while still meaningfully weakening or (aggressive noise) collapsing
detection, so an attacker is not "destroying the audio" to break the mark. Two secondary
observations: amplitude scaling is essentially free for the watermark at both severities
tested; mp3 shows almost no mild→aggressive gap (0.902 → 0.893, plausibly because its
psychoacoustic masking targets bands the watermark doesn't rely on), while noise and
resampling both show steep mild→aggressive drops.

*Method note: WER via Whisper (base) against the fixed synthesis prompt, computed on the
attacked→cloned audio only (no clean/watermarked reference pair exists for these
directories, so PESQ/STOI/SI-SNR were not computed here — this section measures
intelligibility, not perceptual quality).*

**Status: detection-accuracy and WER halves complete (n=100, all 10 conditions). Speaker
similarity / cloning-success-rate (ECAPA-TDNN SIM) under the same 10 conditions is the
remaining open half of this section — does an attack that survives detection loss also
restore the attacker's actual cloning success, or does protection hold even where detection
doesn't?**

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

**Status: direction not yet selected in this doc's earlier draft — NOTE (2026-09-22): the
live repo README has since been updated with Route 2 results per commit 855be70, which this
Project doc predates. The Route 2 checkpoint used in the Post-Processing Robustness section
above suggests the "watermark-aware perturbation objective" direction was chosen and
implemented; this doc should be reconciled against the live README's current Stage 4/5
text next time it's touched, rather than trusted as current.**

---

## Stage 5 — Final evaluation

Once a direction is chosen and implemented, this stage re-measures it against the same
protocol as Stage 3 — attribution ACC, SIM/attack-success-rate, and audio quality together —
and checks whether it shifts the trade-off curve Stage 3 established, or merely moves along
it. A threshold for what counts as success is meant to be fixed before that run, not after.

**Status: see note under Stage 4 above — likely superseded by the live README's current
state.**

---

## Before Stage 4 begins

The Stage 3 numbers above are a trend pass (n=20 per point), not the full n=100 with paired
per-utterance statistics this project otherwise uses throughout. A clean re-run to get that —
now that the environment split between the two evaluation stages is a known, working
procedure — is cheap. That re-run is worth doing, but it makes more sense to fold it together
with whichever Stage 4 direction gets chosen than to run it twice. Holding off on scheduling
it until that decision is made.

---
