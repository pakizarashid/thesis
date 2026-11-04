# Pre-registration N5 — PCA-based perturbation, adopted from Jahangir et al.

Written 2026-11-04, before any PCA clip exists. The implementation is patched into
`src/eval/nonadditive_transforms.py` and its self-test passes (exact reconstruction at scale 0), but no arm has
been generated and no clone has been run.

## What is being adopted, and from where

The perturbation construction from two papers by the same group:

- **Jahangir, Basit, Younis, Shafait — *PCA-Based Adversarial Attacks on Signature Verification Systems*.**
  Algorithm 1: `PC = PCA.transform(X)`; `N = SpatialTransformation.Rotation(PC)` (180°); `nc_i = PC_i + region ×
  N[i]`; `X_adv = PCA.inversetransform(NC)`. Black-box, gradient-free, model-free, data-free.
- **Masood, Jahangir, Athar, Malik, Shafait, Khan — *Assessing Vulnerabilities to Adversarial Perturbations in
  EEG-Based Pathology Detection Systems*.** The same construction on raw multichannel time series, with a second
  noise variant (Laplacian-PCA), reporting up to 72.7 % black-box success at SNR > 15 dB.

**Only the perturbation method is adopted.** Neither paper's threat model, framing or claims are carried over.

### The central claim being replicated

The signature paper's headline is not its 90 % success rate but the comparison at matched imperceptibility:
PCA-structured noise 90 % against **Gaussian 0.5 % and Speckle 0 %** on Cedar. Structure in the perturbation is
worth roughly 90 points over random noise of similar perceptual cost.

This project already holds the audio equivalent of that control. `rs` is same-bound random noise at PESQ 1.6218,
generated before N1 existed and never tuned against anything. **N5 is therefore a direct test of the papers'
central claim in the voice-cloning setting**, with the random-noise comparator fixed in advance.

### The extra stage, which is the scientific content

In both papers the perturbed sample is fed straight to the classifier. Here it is not. The perturbed audio is
published; an attacker clones from it; only then does a verifier compare the clone against the real speaker. **The
perturbation must survive a generative resynthesis**, and nothing in the PCA construction constrains it to.

That is the open question and the registered reason this may fail. Q2 survives because it is optimised through a
differentiable rollout of the cloner; every blind transform tried so far has survived only partially.

## Arms

Applied to the RQA-watermarked audio. Gradient-free, no surrogate, ~1 s/clip.

| arm | variant | role |
|---|---|---|
| `pcarot` | Rotation-PCA, `N = PC[::-1, ::-1]` | **primary** |
| `pcalap` | Laplacian-PCA, seeded | **primary** |

Two primaries, so **Holm correction across the two** for every inferential test. No other arms. Per supervisor
instruction, PCA is **not** combined with Q2 or N1 in this experiment, and the SpeechTokenizer-latent adaptation is
**not** registered here — it is a separate experiment that would need its own evidence that the latent is what a
given cloner reads.

Framing: 1024 samples, hop 512, Hann overlap-add divided by the window sum. At scale 0 this reconstructs the input
to 6×10⁻¹⁶, so any measured degradation is the perturbation and not the framing. Sample count preserved, output
RMS-matched to input.

**Component region: `mid` (the middle third), fixed in advance** on the source paper's evidence — components
100:200 gave 82 % success at SSIM 0.74 against 58 % at SSIM 0.70 for the top-variance 0:100. `low` and `high` are
reported descriptively with no decision rule.

## Calibration, before any clone

The scale is swept on clips 0–19 and interpolated between **adjacent** brackets to **PESQ 1.6218, the measured
PESQ of `rs`**, then verified on all 100 clips. Matching the random-noise control rather than a round number is
what makes the papers' structure-versus-noise comparison like-for-like.

Grid: scale ∈ {0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5}, extended once in the needed direction if the target is not
bracketed, then dropped as not calibratable. Frozen to `results/n5_calib.json` before any decisive clip. PESQ only;
no clone, verifier score or accept rate enters calibration.

Calibrating on clips 0–19 and judging on 100 has produced a subset-versus-full gap three times in this project
(T1E's epsilon, the K=12 PESQ claim, c2 at +0.1026). The gap is therefore **reported, and the calibration is not
retuned** if the full-set PESQ misses.

## Measures

Primary verifier is **Resemblyzer at 0.6868**, held out from every method. ECAPA at 0.2994 is reported beside it
and is **not** combined into a worst case: ECAPA sits inside Q2's optimisation loop, so a maximum across the two is
a descriptive figure and not a measure of independent generalisation. Per-verifier scores are always shown.

Reported for every arm beside `rs`, `n1notchK12`, `q2` and `wm`: accept@Resemblyzer, accept@ECAPA, PESQ, STOI,
SI-SNR, **SNR in dB** (both source papers report it; the EEG paper gates at > 15 dB), WER on three ASR systems,
`traced` (in clone), `traced_source` and `acc_source` (on the published file).

## Hypotheses

Conjunctive; the verdict cell must evaluate and print every condition and name each failure.

**HN14 (primary) — structure beats matched-cost noise after resynthesis.**
For each PCA arm on F5-TTS, all of:
1. accept@Resemblyzer ≤ accept(`rs`) − 0.05;
2. one-sided exact McNemar p < 0.05, Holm-corrected across the two arms;
3. speaker-bootstrap 95 % CI for the difference excludes 0;
4. PESQ within 0.05 of 1.6218, verified on all 100 clips;
5. STOI within 0.05 of `rs`'s and WER within 0.05 of `rs`'s;
6. **`traced_source` ≥ 0.95.**

Condition 6 is new and comes from N4: c2 reached `traced_source` 0.71, meaning 29 % of published clips could not be
traced from the file itself, before any cloning. An arm whose watermark cannot be read from its own published file
fails regardless of its accept rate, and that is now a condition rather than a surprise.

**HN15 — is it better than what we already have?**
accept@Resemblyzer ≤ accept(`n1notchK12`) − 0.05 with CI excluding 0. `n1notchK12` is noise-free at PESQ 1.9583, so
a PCA arm that only ties it at a worse PESQ is not an improvement.

**HN16 — generalisation.** If HN14 passes, confirm on MaskGCT with the same conditions. A pass on F5 alone is not a
pass.

**HN17 — traceability through cloning**, reported not gated: `traced` beside `traced(rs)` = 0.620. A repeated-run
discrepancy of 9 percentage points in `traced` was observed in N4 on identical audio and seeds, so differences of
this size are **not** treated as evidence until the repeated-run variance study is done.

## Pre-declared outcomes

| outcome | interpretation |
|---|---|
| HN14 and HN15 pass, HN16 confirms | PCA perturbation survives resynthesis and beats both random noise and the notch — the strongest result available |
| HN14 passes, HN15 fails | structure beats noise but not the notch; PCA is a valid method, not an improvement here |
| HN14 fails on 1–3 | the papers' structure advantage does **not** survive generative resynthesis — a clean negative, and the question neither paper asks |
| HN14 fails on 6 | PCA reconstruction destroys the watermark, as the RIR did |
| not calibratable | reported and dropped |

## Predictions, before any run

1. P(HN14 for `pcarot`) ≈ 35 %. 2. P(HN14 for `pcalap`) ≈ 30 %.
3. P(HN15 | HN14) ≈ 40 %. 4. P(HN16 | HN14) ≈ 55 %.
5. P(condition 6 fails for at least one arm) ≈ 40 % — PCA reconstruction touches the whole signal, as the RIR did.
6. P(not calibratable within the extended grid) ≈ 15 %.

## What this cannot show

- One corpus, 100 clips, 20 held-out speakers, one watermark, one cloner unless HN16 runs, two verifiers of which
  only one is held out from every method.
- A failure does **not** show PCA perturbation is ineffective in general — only that this construction, at `rs`'s
  perceptual cost, does not beat matched-cost noise under this protocol **after a cloning stage**. The source
  papers' own results stand; they are measured without that stage.
- No comparison with the source papers' numbers is licensed: different modality, different task, different threat
  model, no matched conditions.
- Nothing here licenses a comparison with SafeSpeech, CloneShield, AntiFake, E2E-VGuard or ClearMask.
- Human perception of speaker identity is **not** measured here. That it is unmeasured in this literature,
  including in the two source papers, is a separate open question.

## Order of operations, enforced by the cell

1. Patch the module, run its self-test, commit this document, verify it is tracked. Generation halts otherwise.
2. Sweep the scale on clips 0–19; freeze `results/n5_calib.json`.
3. Generate `pcarot` and `pcalap` on 100 clips; verify PESQ, length and per-clip RMS.
4. **Two-clip `--diagnostic` of every invocation before the 100-clip runs**, per the N4 lesson.
5. Clone with F5-TTS at the frozen thresholds, with `--ecapa --resemblyzer`.
6. Evaluate HN14, HN15 and HN17, printing every condition and every failure.
7. Rebuild the blinded listening file with both arms.
8. MaskGCT only if HN14 passes.

No expansion of scope mid-experiment. No combination with Q2 or N1. No latent-space variant.
