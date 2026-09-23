# Clone-Aware Dual-Defense Audio Protection: Joint Watermarking (Traceability) + Perturbation (Anti-Cloning) Robustness Against Zero-Shot Speech Synthesis

Combines **VoiceMark-style traceable watermarking** with **SafeSpeech-style adversarial perturbation**, and studies whether both survive contact with real zero-shot voice cloning.

**Core question:** attribution needs speaker information to *survive* cloning; anti-cloning protection needs it *destroyed*. Can one system do both — and if not everywhere, exactly where does it break, and can that be fixed?

### Main research goal:
**Can a watermark-bearing adversarial protection system reduce usable speaker information and clone usability, while preserving watermark-based attribution and acceptable quality of the protected source audio, across heterogeneous zero-shot TTS architectures?**

### Two complementary defenses

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

| Objective | Main metric | Desired direction |
|---|---|---:|
| Destroy speaker identity | ECAPA-TDNN SIM | ↓ |
| Reduce clone usability | WER / ASR | WER ↑ / ASR ↓ |
| Preserve traceability | Watermark ACC | ↑ |
| Preserve protected-source quality | PESQ / STOI / SI-SNR | ↑ |

---

**Central hypothesis:**
A watermark trained with the cloning transformation inside the optimization loop can improve traceability across heterogeneous zero-shot cloning architectures, while a complementary adversarial perturbation can reduce speaker identity; however, the two objectives introduce architecture-dependent trade-offs that must be explicitly characterized.

```
                YOUR RESEARCH
                     │
        ┌────────────┴────────────┐
        ↓                         ↓
 VoiceMark limitation       SafeSpeech limitation
        │                         │
 watermark survival         perturbation transfer
 depends on cloner          depends on cloner
        │                         │
        └────────────┬────────────┘
                     ↓
             Architecture-aware
              dual protection
                     │
             ┌───────┴───────┐
             ↓               ↓
       Clone-aware       Anti-cloning
       watermarking      perturbation
             │               │
             └───────┬───────┘
                     ↓
              Joint evaluation
          WM × SIM × ASR × WER
                × quality
                     ↓
          characterize where the
             defense succeeds,
              fails, and why
```
---

### Notes

This project have:
- **Reproduction:** same published setup where feasible.
- **Independent evaluation:** controlled experiments using this project's datasets, samples, or additional TTS architectures.
- **Our contribution:** comparisons made under the same conditions between our baseline and our proposed modification.

> The work below is a five-stage progression.

# Research contributions

| Contribution | Evidence |
|---|---|
| **1. Architecture-dependent watermark survival** | VoiceMark attribution varies strongly across five zero-shot TTS systems; CARRIER-PROBE supports the same trend in the watermark-bearing latent. |
| **2. Clone-aware watermark adaptation** | Jointly training VoiceMark `msg_processor` + detector with a differentiable YourTTS clone improves attribution across multiple unseen cloners. |
| **3. Transferable anti-cloning perturbation** | Waveform PGD trained with YourTTS transfers to unseen XTTS-v2, F5-TTS and MaskGCT under tested conditions. |
| **4. Dual-defense composability** | Route 2 watermark + PGD simultaneously reduces speaker similarity while retaining meaningful watermark detection on multiple architectures. |
| **5. Trade-off characterization** | Epsilon sweep and robustness attacks quantify protection, attribution, intelligibility, and quality interactions. |

**LoRA is an implementation choice, not the claimed novelty.** The contribution is the **clone-aware adaptation strategy**.

---

---
# Research progression

## Stage 1 — Watermark 
> **Does the same watermark survive different zero-shot TTS architectures?**

VoiceMark's released watermark was evaluated on five zero-shot TTS architectures: the three used in the VoiceMark paper (CosyVoice, F5-TTS, MaskGCT) plus YourTTS and XTTS-v2, using the same detector, the same 16-bit payload, and the same harness throughout.

| Cloner      | Main reference-conditioning path                           | Watermark ACC ↑            |
| ----------- | ---------------------------------------------------------- | -------------------------- |
| YourTTS     | fixed d-vector (single speaker embedding)                  | 0.5337                     |
| XTTS-v2     | GPT audio-prompt tokens (discretised)                      | 0.6119                     |
| CosyVoice 2 | flow-matching decoder, regenerated from prompt mel         | 0.7669                     |
| MaskGCT     | masked infilling, quantised RVQ tokens retained in-context | 0.9137                     |
| F5-TTS      | mel infilling, reference mel retained                      | 0.9300                     |

### Finding
> watermark survival is architecture/reference-pathway dependent.

The same watermark produced substantially different attribution across cloning architectures. This motivates the hypothesis that watermark survival depends on the **reference transformation/conditioning pathway**, rather than being a fixed property of the watermark alone.

Across all five architectures (monotone, no ties/inversions), survival is higher when the cloner retains reference audio (F5-TTS, MaskGCT) rather than regenerating it through a decoder (CosyVoice) — consistent with VoiceMark's own published numbers, whose eval set is entirely retained-conditioning.

**CARRIER-PROBE**
Re-encoding clone audio with VoiceMark's own SpeechTokenizer measured raw carrier survival independent of the detector.

| Result | Finding |
|---|---|
| Pooled similarity | Higher survival tracks higher ACC on the reliably measured architectures |
| Per-layer result | All 7 watermark-bearing layers show the same architecture ordering |
| Most fragile layer | **Layer 2** shows the largest architecture sensitivity |
| Simple layer reweighting | **Null**: inference-time down-weighting/removal of Layer 2 did not improve ACC |
| CosyVoice | Excluded from the clean carrier-sim subset because of reference-duration mismatch |

**Interpretation:** carrier survival is a useful diagnostic, but simple inference-time carrier reweighting is not a fix.

**Status: completed**

---

## Stage 2 — Anti-cloning protection
> **Does the protection transfer to the same five architectures?**

### Mechanism
The main anti-cloning is **waveform-domain PGD**. 
```text
watermarked waveform + bounded δ → protected waveform → zero-shot TTS → clone
```
The perturbation is optimized through a differentiable **YourTTS surrogate**; the other TTS systems are black-box evaluation targets.

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

### Finding
> Increasing ε eventually makes F5-TTS cloning much harder, but stronger protection also reduces watermark attribution and protected-source quality.

Attribution degrades toward chance in parallel (ACC 0.84 → 0.59), and audio quality collapses independently (PESQ near floor by ε = 0.04; SI-SNR negative by ε = 0.08 — perturbation louder than the speech itself). There is no ε in this range where protection, attribution, and usable audio all hold at once — a genuine three-way **trade-off**.

Two later attempts to fix this by reformulating the PGD objective (H-SPEC, H-DIRECT) came back null/adverse and are closed.

### Reformulation attempts

| Attempt | Result |
|---|---|
| H-SPEC: SafeSpeech-style KL/L1 terms through surrogate clone | **Null / adverse** |
| H-DIRECT: KL/L1 directly on perturbed input mel | **Null / adverse** |
| Simple RVQ layer reweighting | **Null** |

These closed the simple **loss-reformulation / inference-time carrier-eweighting** paths.

**Status: completed** 
---

### Stage 4 result — Route 2 epsilon sweep

In route 2 was then evaluated across the same five ε values used in Stage 3 (F5-TTS, n=100).

| ε | ACC (orig → Route2) | SIM mean (orig → Route2) | ASR (orig → Route2) | PESQ (orig → Route2) | STOI (orig → Route2) | SI-SNR (orig → Route2) |
|---|---|---|---|---|---|---|
| 0.002 | 0.8381 → **0.9575** | 0.4140 → **0.3122** | 91% → **70.0%** | 1.919 → 1.786 | 0.885 → 0.863 | 0.43 → 0.35 dB |
| 0.01  | 0.7000 → **0.8575** | 0.3250 → **0.2260** | 75% → **44.0%** | 1.337 → 1.317 | 0.834 → 0.809 | 0.24 → 0.22 dB |
| 0.02  | 0.6844 → **0.7819** | 0.2713 → **0.1685** | 55% → **26.0%** | 1.155 → 1.152 | 0.784 → 0.763 | −0.28 → −0.18 dB |
| 0.04  | 0.6062 → **0.6900** | 0.1605 → **0.1269** | 15% → **12.0%** | 1.069 → 1.070 | 0.708 → 0.697 | −1.77 → −1.37 dB |
| 0.08  | 0.5875 → 0.5988 | 0.0969 → **0.0669** | 5% → 8.0% | 1.042 → 1.043 | 0.610 → 0.612 | −4.95 → −4.12 dB |

**SEE STAGE 4 for finding**

---

## Stage 4 — Clone-aware watermark adaptation 
> **route2 - training the watermark encoder jointly with the detector, cloning inside the loop**

The direction chosen from Stage 4's three candidates was the third: retraining the watermark itself with the cloning operation inside the training loop, rather than reshaping the perturbation objective or building a second surrogate cloner. 
LoRA adapters on top of the frozen pretrained backbone are trained against a joint loss (clean-audio detection + detection on a differentiable YourTTS clone of the watermarked audio), so the watermark itself learns to leave a signal that survives being cloned — not just a perturbation reacting to a fixed watermark.


This is the strongest watermarking contribution is **Route 2**: train the VoiceMark watermark encoder (`msg_processor`) and detector jointly while a differentiable YourTTS clone is inside the training loop.

### Idea
Instead of changing the anti-cloning perturbation, adapt the watermark itself to cloning:

```text
Watermark encoder (`msg_processor`) + detector
                    ↓
          clone-aware training
                    ↓
       differentiable YourTTS clone
                    ↓
        watermark learns to remain
       recoverable after zero-shot cloning
```

The LoRA adapters are trained on top of the frozen VoiceMark backbone. **YourTTS is the single differentiable training cloner; XTTS/F5-TTS/CosyVoice/MaskGCT are evaluation targets.**


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

**Improvement:** +8% ACC on held-out XTTS (paired t p=0.000003; Wilcoxon p=0.000007).

> The encoder itself therefore contributes additional clone-robust information; the improvement is not only a better detector reading a fixed carrier.

PESQ is lower than this project's own untouched-embedder checkpoints. The gain and the cost are the same phenomenon: touching msg_processor's LoRA changes what the watermarked audio sounds like, which is exactly what both the attribution improvement and the quality/SIM cost trace back to.

### Chasing the quality/SIM cost: three independent nulls

Three separate levers were tried against the PESQ/SIM cost, each testing a different hypothesis for what was driving it:

### Quality-cost analysis

Three controlled attempts did not remove the quality/SIM cost of adapting `msg_processor`:
| Lever | ACC ↑	| SIM ↓ |	Conclusion |
|---|---|---|---|
| `--epochs`, Fewer epochs (9 vs. 20) | p = 0.235	| p = 0.183 | No significant difference |
| `--lambda_clone`, Lower clone-loss weight (0.5 vs. 1.0) |	p = 0.907 |	p = 0.607 |	No significant difference |
| `--msgproc_lora_r`, Lower LoRA rank (2 vs. 8) |	p = 0.205 |	p = 0.438 |	No significant difference |

**Conclusion:** the observed quality cost appears associated with adapting `msg_processor` itself rather than training duration, clone-loss weight, or LoRA capacity among the tested settings.


### Rank-2 efficiency

> **Bonus finding from the rank sweep:** `--msgproc_lora_r 2` — trained from LoRA zero-init (no Stage-1 warm start, since a rank-2 adapter can't reuse rank-8 weights)

| Metric | Rank 8 | Rank 2 |
|---|---:|---:|
| XTTS ACC, n=100 | 0.69–0.71 | **0.7288** |
| Trainable parameters | 295K | **221K** |
| Relative parameter count | 100% | **~75%** |
| ACC vs rank-8 | — | not significant, p = 0.205 |
| SIM vs rank-8 | — | not significant, p = 0.438 |

Rank 2 is the preferred checkpoint for final evaluation because it is smaller and statistically equivalent to rank 8 in tested comparisons.


### 3) Generalization: 
> **Cross-Dataset and Cross-Cloner architecture**

n=100, rank= 2
| Evaluation| Detector-only | Joint Route 2 | Change |
|-----------|-------:|----------------:|------------:|
| VCTK      | 0.6169 | 0.6913 / 0.7006 | +0.0837 |
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
| Checkpoint        | ↑ ACC on XTTS clone |
|-------------------|---------------------|
| Rank 8 (epoch 9)  |	                — |	         
| Rank 2	        | 0.7288              |	               

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

PGD optimised against the differentiable YourTTS surrogate `(ε = 0.002, λ_wm = 1.0, matching Stage 2's original operating point)`, then the protected audio cloned through the real, non-differentiable XTTS-v2 — the same held-out transfer test as Stage 2. Route 2 rank-2 checkpoint, 

** XTTS-v2, n=100:**

|                     | WM ACC ↑   | SIM ↓       |
| ------------------- | ---------- | ----------- |
| unprotected clone   | 0.7244     | 0.3996      |
| protected clone     | 0.6631     | 0.3032      |
| paired significance | p = 0.0016 | p < 0.00001 |

This shows cross-architecture anti-cloning transfer while the watermark remains above chance. The ACC decrease is real and is reported as a trade-off

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


### Stage 4 result — Route 2 epsilon sweep

In route 2 was then evaluated across the same five ε values used in Stage 3 (F5-TTS, n=100).
>> see Stage 3 for table
>> 
### Finding
> **Route 2 shifts the protection/attribution curve outward at ε=0.002–0.04.**

At matched ε, Route 2 generally gives:
- higher watermark ACC,
- lower speaker SIM / ASR,
- similar protected-source quality.

**Caveat:** the original non-baseline sweep points were n=20 while Route 2 uses n=100; therefore the comparison is a strong direction/magnitude result, not a formal paired significance test across the two sweeps.

**Status: ✅ Complete**

---

# Post-processing robustness

Route 2 dual-defense audio was tested against 10 pre-cloning attacks: MP3, Opus, resampling, amplitude scaling, and additive noise; each at mild/aggressive severity, n=100.

### No-attack baseline

| Condition | WM ACC ↑ | SIM ↓ | ASR ↓ |
|---|---:|---:|---:|
| Unprotected | 0.9825 | 0.3680 | 88% |
| Protected (WM + PGD) | **0.9431** | **0.3106** | **70%** |

### 10 attack conditions

| Attack | Severity | WM ACC ↑ | WER ↑ | SIM ↓ | ASR ↓ |
|---|---|---:|---:|---:|---:|
| Amplitude ±3 dB | Mild | 0.9494 | 0.021 | 0.3018 | 63% |
| Amplitude ±6 dB | Aggressive | 0.9431 | 0.026 | 0.3068 | 68% |
| Opus 64 kbps | Mild | 0.9275 | 0.016 | 0.3030 | 70% |
| Resample 16k→22.05k→16k | Mild | 0.9175 | 0.021 | 0.3040 | 68% |
| MP3 128 kbps | Mild | 0.8988 | 0.041 | 0.3067 | 62% |
| MP3 32 kbps | Aggressive | 0.8875 | 0.016 | 0.2974 | 69% |
| Opus 16 kbps | Aggressive | 0.8719 | 0.013 | 0.3051 | 71% |
| Noise 20 dB SNR | Mild | 0.8206 | 0.024 | 0.2883 | 61% |
| Resample 16k→8k→16k | Aggressive | 0.7525 | 0.022 | 0.2745 | 63% |
| Noise 10 dB SNR | Aggressive | 0.6375 | 0.018 | 0.2137 | 35% |

### Finding

> **Post-processing did not provide a route back to successful cloning against genuinely protected audio.**

Nine of ten attack conditions have ASR at or below the 70% protected baseline; the 71% Opus-16 kbps result is only a 1-point difference.

The harshest noise condition simultaneously gives the lowest watermark ACC and the lowest cloning ASR, showing that stronger signal damage can hurt both sides rather than restoring cloning.

### Important metric distinction

| Metric | What it measures |
|---|---|
| PESQ / STOI / SI-SNR | Quality of the **protected source audio** compared with the clean original |
| Watermark ACC | Watermark/traceability in the **clone** |
| ECAPA SIM | Similarity of the **clone speaker identity** to the true source speaker |
| WER | Intelligibility of the **clone speech** |
| ASR (SIM > 0.25) | Fraction of clones above the chosen speaker-similarity threshold |

A separate human listening test (SMOS) has not been run.

**Status: ✅ Complete**


---


# Final dual-defense evidence

The current strongest evidence is that watermark attribution and anti-cloning protection can operate together rather than one automatically destroying the other.

| TTS | WM ACC before → after PGD | Speaker/clone result | Interpretation |
|---|---:|---:|---|
| XTTS-v2 | 0.7244 → **0.6631** | SIM 0.3996 → **0.3032** | Both mechanisms survive; ACC cost is significant |
| F5-TTS | 0.9844 → **0.9450** | ASR 87% → **72%** | Both mechanisms survive; protection transfers |
| MaskGCT | 0.9481 → **0.8794** | ASR 71% → **51%** | Both mechanisms survive; protection transfers |
| CosyVoice | Excluded | Unreliable clone control | Protocol limitation, not a defense finding |

### Final interpretation

> **The proposed dual-defense system can make zero-shot voice cloning harder while retaining watermark-based traceability across multiple heterogeneous TTS architectures.**

The evidence also shows a real trade-off: stronger perturbations and some combined-attack conditions can reduce watermark attribution. The system should therefore be evaluated using **all four dimensions together**: speaker identity, clone usability, watermark attribution, and protected-source quality.

---

# What is established vs. what is not

| Established by experiments | Not claimed |
|---|---|
| Clone-aware `msg_processor` + detector training improves attribution | Universal protection against every TTS |
| Waveform PGD can transfer to unseen TTS architectures | Perfect prevention of cloning |
| Route 2 improves the measured attribution/protection curve at ε=0.002–0.04 | Zero-cost interaction at every architecture/condition |
| Post-processing did not restore cloning success in the tested 10 conditions | That every possible post-processing attack will fail |
| Watermark and anti-cloning can coexist | That cloned audio is always unintelligible |
| WER provides a separate measure of clone usability | That low SIM alone means the clone is unusable |

---

## Limitations

- **Zero-shot threat model only.** No fine-tuning-based cloning attack is evaluated.
- **YourTTS is the differentiable training cloner.** Other TTS architectures are black-box evaluation targets.
- **No exact VoiceMark & SafeSpeech reproduction.** Published Voicemark & SafeSpeech comparisons differ in corpus/model setup and are reported only as contextual comparisons.
- **CosyVoice composability is excluded.** Its standardized reference-crop protocol produced unreliable unprotected clones, so it is not treated as evidence for or against the defense.
- **No subjective listening test (SMOS).** All quality evidence is objective (PESQ/STOI/SI-SNR/WER).
- **Single primary training corpus.** Main development uses LibriSpeech; VCTK is used as an independent evaluation set for Route 2.
- **Route 2 vs. original epsilon sweep is not a formal paired comparison** because the original non-baseline sweep used n=20 while Route 2 used n=100.

- **Future attack scope.** Fine-tuning attacks and additional compression codecs remain outside the current thesis evaluation.
  
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

---

## Robustness to Post-Processing Attacks (Route 2, F5-TTS)

Ten attack conditions (5 attack types × 2 severities — mp3/opus lossy re-encode via ffmpeg,
resample via a polyphase round-trip through an intermediate rate, amplitude a random ±dB
gain, noise additive white Gaussian at a target SNR), n=100 each:

| Attack | Severity | Detection ACC (clone) | Mean WER (cloned speech) | SIM mean | SIM ASR |
|---|---|---|---|---|---|
| amplitude ±3dB | mild | 0.9406 | 0.021 | 0.5298 | 97.0% |
| amplitude ±6dB | aggressive | 0.9400 | 0.026 | 0.5295 | 99.0% |
| opus 64kbps | mild | 0.9400 | 0.016 | 0.5392 | 98.0% |
| resample 16k→22.05k→16k | mild | 0.9256 | 0.021 | 0.5354 | 99.0% |
| mp3 128kbps | mild | 0.9019 | 0.041 | 0.5273 | 99.0% |
| mp3 32kbps | aggressive | 0.8931 | 0.016 | 0.5011 | 99.0% |
| opus 16kbps | aggressive | 0.8725 | 0.013 | 0.5502 | 99.0% |
| noise 20dB SNR | mild | 0.8275 | 0.024 | 0.4571 | 93.0% |
| resample 16k→8k→16k | aggressive | 0.7488 | 0.022 | 0.4214 | 96.0% |
| noise 10dB SNR | aggressive | 0.6569 | 0.018 | 0.3656 | 87.0% |

1) How much watermark survives these attacks — direct answer from the table already measured:

Detection ACC is watermark survival. Against the protected-no-attack baseline (0.9431), retention by condition:

| Attack | Severity | Detection ACC (clone) | % of basline retrained |
|---|---|---|---|
| amplitude ±3dB | mild | 0.9406 | 99.7% |
| amplitude ±6dB | aggressive | 0.9400 | 99.7% |
| opus 64kbps | mild | 0.9400 |  99.7% |
| resample 16k→22.05k→16k | mild | 0.9256 |  98.2% |
| mp3 128kbps | mild | 0.9019 | 95.6% |
| mp3 32kbps | aggressive | 0.8931 | 94.7% |
| opus 16kbps | aggressive | 0.8725 |  92.5% |
| noise 20dB SNR | mild | 0.8275 | 987.7% |
| resample 16k→8k→16k | aggressive | 0.7488 | 79.4% |
| noise 10dB SNR | aggressive | 0.6569 |  69.7% |

---
