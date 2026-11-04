# Pre-registration N7 — is it the F-ratio, or would any twelve bins do?

Written 2026-11-04, before any N7 clip exists.

## The question

N1's claimed contribution is the **selection criterion**, not the notching. ClearMask already zeroes a fixed set
of spectrogram bins; N1 ranks them by a speaker F-ratio. If the ranking does nothing, the contribution collapses
to ClearMask's mechanism and the thesis must say so.

One piece of evidence already exists: an unsupervised **variance** criterion (PCA, N5) reaches held-out accept
0.890/0.930 where the discriminative criterion reaches 0.690. But **random bin selection has never been run**, and
it is the control a reviewer asks for first.

## Arms

All at **K = 12**, the frozen operating point, on RQA-watermarked audio. Only the rule choosing the twelve bins
differs. Same STFT (2048/512 Hann), same eligibility band (80–7800 Hz, −60 dB floor), RMS-matched, sample count
preserved, A5 curve design (PESQ reported, not matched).

| arm | selection rule | role |
|---|---|---|
| `n1notchK12` | top 12 by speaker F-ratio | the method (already measured: 0.690) |
| **`n1randK12`** | 12 bins drawn uniformly at random from the eligible set, seed 20261104 | **PRIMARY control — HN22** |
| `n1energyK12` | the 12 highest mean-energy eligible bins | secondary: a non-random naive criterion |

`n1randK12` is the primary comparison. `n1energyK12` carries no decision rule and is reported with a
Holm-corrected p-value.

Drawing random bins **from the eligible set** keeps the comparison fair: both arms are confined to the speech band
above the energy floor, so the test isolates the ranking and not the band restriction.

## Hypotheses

**HN22 (primary) — the F-ratio ranking is doing the work.**
PASSES if, for `n1notchK12` against `n1randK12` on F5-TTS, unseen text, n=100, Resemblyzer at 0.6868, **all** hold:
1. accept(F-ratio) ≤ accept(random) − 0.05;
2. one-sided exact McNemar p < 0.05;
3. speaker-bootstrap 95 % CI for the difference excludes 0.

**HN23 — quality is not the explanation.** PESQ of `n1randK12` within 0.10 of `n1notchK12`'s 1.9583. If random
bins cost markedly more or less quality, the arms are not at comparable cost and HN22's comparison is reported
with that caveat rather than as a clean criterion test.

**HN24 — the watermark.** `traced_source` ≥ 0.95 for every arm, as in N6.

## Pre-declared outcomes

| outcome | interpretation |
|---|---|
| HN22 passes | the speaker F-ratio is the contribution; N1 is a criterion, not a re-run of ClearMask |
| HN22 fails on 1–3 | **any twelve eligible bins protect equally**; the contribution is the notching, which is prior work, and the thesis reports N1 as a reproduction with a verifier-independent justification rather than a new method |
| HN22 passes but HN23 fails | the arms differ in perceptual cost; the criterion claim is weakened and reported at matched PESQ instead |
| `n1energyK12` ≈ `n1notchK12` | a simpler criterion works as well; report it, and prefer the simpler rule |

The second row is the one that matters. It is written down now, before the run, because it is the outcome that
would most change what this thesis claims — and it is a real possibility: N6 showed accept is flat from K=12 to
K=96, which is consistent with the top-ranked bins not being special.

## Predictions, before any run

1. P(HN22 passes) ≈ 55 %.
2. P(`n1energyK12` within 0.05 of `n1notchK12`) ≈ 40 % — high-energy bins and high-F-ratio bins overlap in the F0
   region.
3. P(HN23 holds) ≈ 80 %.

## What this cannot show

- One corpus, 100 clips, 20 held-out speakers, one cloner, one K.
- It cannot show the F-ratio is the *best* criterion, only whether it beats random and naive-energy selection.
- A single random draw is one sample from the space of random selections; a stronger version would average several
  seeds, which is noted as a limitation rather than run here.

## Order of operations

1. Commit this document; generation halts if it is not tracked.
2. 2-clip diagnostic of each invocation before any 100-clip run.
3. Regenerate `n1notchK12` and verify it reproduces PESQ 1.9583; a mismatch stops everything.
4. Generate `n1randK12` and `n1energyK12`; verify length, RMS and `traced_source`.
5. Clone with F5-TTS at the frozen thresholds, both verifiers enabled.
6. Evaluate HN22–HN24, printing every condition and every failure.
