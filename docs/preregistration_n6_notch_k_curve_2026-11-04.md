# Pre-registration N6 — how far does the noise-free notch go?

Written 2026-11-04, before any K > 12 clip exists.

## Why

`n1notchK12` is the best noise-free method in the project: held-out Resemblyzer accept 0.690, PESQ 1.9583, traced
0.980, ~1 s/clip, cross-cloner spread 1.8–3.4 points. It is noise-free by the author's blinded listening check,
repeated twice.

But **K = 12 was never chosen for protection**. It came from calibrating the notch to Q2's PESQ in N1. Nothing has
tested whether notching more bins buys more protection while keeping the property that makes this arm worth
having — that it adds no audible noise.

N5 sharpened why this arm is the one to push: at matched perceptual cost, a PCA variance criterion was
indistinguishable from random noise (0.890/0.930 against rs's 0.900), while the F-ratio discriminative criterion
reaches 0.690. The selection criterion is doing the work, so the question is how much further it goes.

## Arms

Frozen F-ratio ranking (`results/n1_fratio_bins.json`), top-K bins zeroed, RMS-matched, sample count preserved.
Amendment A5 curve design: **K is passed explicitly and PESQ is reported, not matched to a target.**

| arm | K | role |
|---|---|---|
| `n1notchK12` | 12 | anchor, regenerated; must reproduce PESQ 1.9583 |
| `n1notchK24` | 24 | **PRIMARY — HN18 is tested on this arm alone** |
| `n1notchK48` | 48 | secondary, Holm-corrected |
| `n1notchK96` | 96 | secondary, Holm-corrected |

K = 24 is the primary because it is the single doubling from the established arm, fixed here before any clip.
"At least one K of three works" is not a result at this project's effect sizes; the other two are curve points.

## Hypotheses

Conjunctive. The verdict cell must evaluate and print every condition and name each failure.

**HN18 (primary) — more notching buys more protection, noise-free.** For `n1notchK24` on F5-TTS:
1. accept@Resemblyzer ≤ accept(`n1notchK12`) − 0.05;
2. one-sided exact McNemar p < 0.05;
3. speaker-bootstrap 95 % CI for the difference excludes 0;
4. **`traced_source` ≥ 0.95** — the watermark readable from the published file;
5. WER within 0.05 of `n1notchK12`'s on all three ASR systems;
6. **the author's blinded listening check reports no added noise.**

Condition 6 is the point of the arm. A notch that protects more but is audible is just another point on the curve
the project has already mapped seven times.

**HN19 (the boundary) — where does noise-freeness end?** Every arm gets the 8-clip blinded check. The smallest K
at which added noise is reported is itself the finding, and is reported whether or not HN18 passes.

**HN20 — intelligibility.** WER on the published audio at the primary K, within 0.05 of clean's. Zeroing bins in
the 80–7800 Hz band is the arm most at risk of damaging speech, and this is a condition, not a note.

**HN21 — generalisation.** If HN18 passes, confirm on MaskGCT with the same conditions. F5 alone is not a pass.

## Measures

Primary verifier **Resemblyzer at 0.6868**, held out from every method. **ECAPA at 0.2994 reported separately and
never combined into a worst case** — it sits inside Q2's loop, so a maximum over the two is descriptive only, and
per-verifier scores are always shown. Also reported for each arm: PESQ, STOI, SI-SNR, SNR, WER (three ASR systems),
`traced`, `traced_source`, `acc_source`.

`traced` differences smaller than about 9 points are **not** treated as evidence: a repeated-run discrepancy of
that size was observed in N4 on identical audio and seeds. `traced_source` is deterministic and is not subject to
this.

## Pre-declared outcomes

| outcome | interpretation |
|---|---|
| HN18 passes and HN19 finds no noise through K=96 | the noise-free method was under-used at K=12; report the curve and the new operating point |
| HN18 passes but noise appears at K=24 | protection and noise-freeness trade off within the notch itself; K=12 is near the boundary |
| HN18 fails on 1–3 | the F-ratio ranking saturates; the top bins carry what there is to take |
| HN18 fails on 4 | more notching destroys the watermark, as the RIR did |
| HN18 fails on 5 | intelligibility is the binding constraint, not audibility |

## Predictions, before any run

1. P(HN18 for K=24) ≈ 45 %. 2. P(noise is audible at K=24) ≈ 25 %, at K=96 ≈ 70 %.
3. P(condition 4 fails at K=24) ≈ 20 %, at K=96 ≈ 55 % — the watermark lives in the same spectrum.
4. P(the WER condition fails at K=96) ≈ 50 %.
5. P(accept is monotone in K across the grid) ≈ 60 %.

## What this cannot show

- One corpus, 100 clips, 20 held-out speakers, one watermark, one cloner unless HN21 runs, one held-out verifier.
- It cannot show the F-ratio ranking is optimal, only how accept behaves as more of it is used.
- Human perception of *clone* identity is not measured here.
- No comparison with SafeSpeech, CloneShield, AntiFake, E2E-VGuard or ClearMask is licensed.

## Order of operations

1. Commit this document; generation halts if it is not tracked.
2. 2-clip diagnostic of every invocation before any 100-clip run.
3. Regenerate `n1notchK12` and verify it reproduces PESQ 1.9583; a mismatch stops everything.
4. Generate K=24, 48, 96; verify length, RMS and `traced_source` per clip.
5. Clone with F5-TTS at the frozen thresholds, with both verifiers enabled.
6. Evaluate HN18–HN20, printing every condition and failure.
7. Build the blinded listening file with all four arms; score HN19 before calling HN18.
8. MaskGCT only if HN18 passes.

No expansion of scope mid-experiment.
