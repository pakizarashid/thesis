# Pre-registration N1 — non-additive protection

Written 2026-11-03. **Revised the same day, after supervisor review and before any code exists or any N1 clip is
generated.** The revisions narrow three overclaims and fix a multiplicity error; they are recorded here rather
than as an amendment because nothing had been run.

Revision log:
- HN1 named "at least one N1 arm", which is multiplicity shopping across three arms. **`n1notch` is now the sole
  primary arm**; the others carry no decision rule.
- "architecture-agnostic by construction" removed. The *selection criterion* is verifier-independent; whether the
  transform transfers across cloning architectures is what F5 and MaskGCT test.
- Matched PESQ is not matched perceptual cost. STOI, SI-SNR, loudness and ASR intelligibility are now reported
  conditions, and one of them can fail the arm on its own.

## The claim this tests

Every mechanism tried so far adds signal to the published audio — a perturbation, or a watermark imprint — and
all six land on one protection-versus-perceptual-cost curve:

| family | arms | result |
|---|---|---|
| loudness floor | s40f, Q1, Q2, Q3 | on the curve |
| band-limiting | BP15, BP20 | on the curve (48 dB less HF energy, 41 % of the effect lost) |
| psychoacoustic masking | ME, MH | on the curve (both buy PESQ +0.20, both land at 59 %) |
| optimiser momentum | t1, t1e | on the curve once energy is matched |
| in-loop ensemble size | v1, v5, v6 | saturates at 3 |
| embedded protection (watermark) | p1g, r32 arms | on the curve; accept tracks imprint strength |

**N1 asks whether a transform that adds no signal sits below that curve** — more protection at the same
perceptual cost. A non-additive transform *moves or removes* information instead of adding it, so there is a
mechanistic reason it need not obey the same trade-off. This also matches the stated requirement that the
published recording carry no audible added noise.

## Frozen baselines

Fixed before any N1 clip, reused and never regenerated:

- **`q2`** — the registered operating point. PESQ 1.7887 on the standard 100 clips; F5-TTS accept and traced as
  already recorded in `results/ce_ac_f5tts_RQA_q2.json`.
- **`wm`** — watermark only, no protection, same checkpoint.

Both are compared by checkpoint SHA and args provenance before use, by the same audit that closed P1.

## Prior work, and what is and is not borrowed

ClearMask (arXiv 2508.17660) is noise-free: it zeroes 12 of 1025 spectrogram bins, applies DeepAFx-ST style
transfer, and convolves a ≤ 30 ms room impulse response. Reported MOS 3.12 against AntiFake's 2.82 and STOI 0.65
against 0.34, on YourTTS / DiffVC / AGAIN-VC plus ElevenLabs and Play.ht.

Its numbers do not transfer here and no comparison is implied: its cloners are voice-conversion models rather
than zero-shot TTS, its audio carries no watermark, and its thresholds and enrolment protocol are its own. N1
borrows the mechanism family, not the numbers.

**What is new here:** ClearMask zeroes a fixed set of bins. N1 selects bins by a **speaker F-ratio computed on
the 60 designated training speakers** — between-speaker variance of per-speaker mean log-magnitude over mean
within-speaker variance, per frequency bin. That is a property of speech rather than of any speaker verifier, so
it is a **verifier-independent, data-driven spectral selection criterion** and carries no in-loop inflation
risk.

Two things it is not. It is not architecture-agnostic by construction: the criterion ignores the verifier, but
the resulting transform may still transfer differently across cloning architectures, which is precisely what the
F5-TTS and MaskGCT evaluations test. And the F-ratio does not establish that the selected bins are causally
necessary for speaker identity — it ranks bins by between- versus within-speaker variation in training data. The
experiment tests whether suppressing them actually reduces clone identity preservation.

## Arms, fixed now

Pure signal processing on the RQA-watermarked waveform. No PGD, no gradients, no surrogate — about 1 s per clip,
so cloning dominates the cost. Every transform preserves sample count exactly and is RMS-matched to the input,
so no arm can gain its effect from added loudness.

| arm | mechanism | role |
|---|---|---|
| `n1notch` | zero the top-K speaker-F-ratio bins (STFT 2048/512, Hann) | **PRIMARY — HN1 is tested on this arm alone** |
| `n1rir` | convolve a synthetic RIR of length L ms (exponentially decaying noise, fixed seed) | secondary, no decision rule |
| `n1both` | notch then RIR, each at its own calibrated strength | descriptive only; changes two things at once |

Declaring one primary arm in advance is deliberate: "at least one of three passes" is not a result at this
project's effect sizes. `n1rir` and `n1both` are reported with Holm-corrected p-values as secondary points on
the same curve, exactly as `v5` was in the T1/T2 amendment.

## Calibration, done before any decisive clip and then frozen

Each arm's strength is tuned so its clean-audio PESQ equals **Q2's PESQ measured on the same calibration clips**,
so the comparison is at matched perceptual cost rather than matched hyperparameters. The target is read from
Q2's own per-clip record for clips 0–19 and frozen to `results/n1_target_pesq.json` before any sweep; the
100-clip mean (1.7887) is reported beside it but is not the target, because calibrating on 20 clips against a
100-clip mean is the mismatch that froze T1E's epsilon at the wrong value.

- `n1notch`: K ∈ {8, 16, 32, 64, 128} bins. `n1rir`: L ∈ {5, 10, 20, 30} ms.
- Clips 0–19, mean PESQ per setting, linear interpolation to the setting whose PESQ equals the frozen target. If
  the target lies outside the grid, the grid is extended once in the needed direction; if still outside, that arm
  is reported "not calibratable at this quality" and dropped.
- Written to `results/n1_calib.json` **before** any decisive clip and not retuned afterwards.
- Calibration uses PESQ only — a property of the published file. No clone, no accept rate and no verifier score
  enters it, so no outcome information leaks. Clips 0–19 overlap the evaluation set; this is disclosed and is the
  same procedure accepted for ME/MH and T1E.

## Quality is not one number

PESQ equality is the calibration target, not the definition of matched cost. All of the following are reported
for every arm beside Q2, and the two marked as conditions can fail an arm on their own:

| measure | role |
|---|---|
| PESQ | calibration target, matched by construction |
| **ASR WER on the published clip** | **condition: within 0.05 absolute of Q2's** |
| **STOI** | **condition: within 0.05 of Q2's** |
| SI-SNR | reported; a large drop indicates the transform is doing more than the PESQ match suggests |
| loudness (RMS, dBFS) | reported; RMS-matched by construction, verified per clip |
| author listening check | reported on the standard 8-clip set, as for ME/MH/BP |

A transform that holds PESQ while destroying intelligibility is not an equivalent-quality comparison, and
`n1notch` is the arm most at risk of exactly that.

## Scoring

Decisive cloner **F5-TTS** for the primary test, on cost. **MaskGCT confirms any pass** — it is the cloner on
which Q2 fails to beat same-bound random noise, so a mechanism that matters must show there too. Held-out judge
**Resemblyzer at 0.6868**, frozen 2026-10-26. ECAPA at 0.2994 reported beside it as an inflation check only.
100 clips, 20 held-out speakers, unseen text, 3 s prompt, same frozen cloners, thresholds and pairing procedure
as Q2. Exact McNemar on paired clips plus speaker-level bootstrap CI (10,000 draws); Holm correction across the
secondary arms.

## Hypotheses and decision rules

Every condition is conjunctive and **the verdict function must evaluate all of them and print which ones
failed.** The A2 cell printed `GATE MET` after checking two of three conditions; that must not recur.

**HN1 (primary) — a non-additive transform beats the curve.**
PASSES if, for `n1notch` on F5-TTS, **all** hold:
accept ≤ accept(`q2`) − 0.05; one-sided exact McNemar p < 0.05; speaker-bootstrap 95 % CI for the accept
difference excludes 0; PESQ within 0.05 of the frozen target (verified on all 100); WER within 0.05 of Q2's;
STOI within 0.05 of Q2's; traced ≥ traced(`q2`) − 0.05.

**HN2 (the real risk) — the watermark survives.**
Notching removes information and an RIR smears it; the watermark lives in SpeechTokenizer layers 2–8 and may
occupy the very bins the F-ratio selects. An arm that protects but drops traced below 0.84 (Q2's 0.89 minus
0.05) **fails HN1** and is reported as a mechanism that cannot coexist with the watermark — itself a finding,
since it would mean protection and traceability compete for the same spectral real estate.

**HN3 (generalisation) — if HN1 passes on F5-TTS, the effect appears on MaskGCT** with a spread no wider than
Q2's 9–20 points. A pass on F5 alone is not a pass.

**HN4 (descriptive) — where does it sit on the curve?** Every N1 arm is plotted beside the six additive families
on protection-versus-PESQ.

## Pre-declared outcomes

| outcome | interpretation |
|---|---|
| better protection than Q2 at matched quality, watermark retained | strongest positive result |
| better protection, traced falls below 0.84 | evidence of a protection–traceability conflict |
| protection similar to Q2 at matched quality | supports the broader quality–protection boundary |
| worse protection, or a quality condition fails | the N1 mechanism fails under this protocol |

The predictions below are expectations, not decision criteria. **The thresholds above are not changed after
seeing results.**

## Predictions, before any run

1. P(HN1 passes for `n1notch` on F5-TTS) ≈ 30 %.
2. P(HN2 holds for `n1notch`, given protection appears) ≈ 50 %. The coin-flip: speaker-informative bins and
   watermark-carrying bins are plausibly the same bins.
3. P(HN2 holds for `n1rir`) ≈ 65 %. Convolution preserves more information than zeroing.
4. P(the WER condition fails for `n1notch` at its calibrated K) ≈ 35 %.
5. P(HN3 holds, given HN1) ≈ 60 %.
6. P(at least one arm is not calibratable to the frozen PESQ target) ≈ 25 %.

## What this cannot show

- One corpus, 100 clips, 20 speakers, one watermark, two cloners if HN3 runs.
- It cannot separate the components beyond the stated ablation; `n1both` changes two things at once by design and
  is never the evidence for HN1.
- No comparison with ClearMask's published numbers is licensed.
- A failure does not show non-additive protection is impossible — only that these transforms, at Q2's perceptual
  cost, do not beat the curve.
- **If N1 lands on the curve, the defensible claim is bounded**: across seven tested protection families,
  including additive perturbations and a speaker-informed spectral transformation, no improvement is found
  beyond the established quality–protection trade-off *under the evaluated cloning and verification protocol*.
  One additional non-additive mechanism does not establish a universal limit for the voice-cloning threat model,
  and the write-up must not imply it does.
- The threat model remains zero-shot cloning. Fine-tuning attacks (SafeSpeech's setting) are not tested.

## Order of operations, enforced by the notebook

1. Copy this document into `docs/` and commit it **before any clip is generated**; the notebook's section 10
   verifies the commit exists and halts if not.
2. Compute and freeze the F-ratio ranking using only the 60 designated training speakers.
3. Freeze the calibration target from Q2's record on clips 0–19, then calibrate each arm to it and freeze all
   parameters to `results/n1_calib.json`.
4. Generate `n1notch`, `n1rir`, `n1both`.
5. Evaluate with the frozen cloners, verifier, thresholds and pairing procedure used for Q2.
6. Report all three arms, failures included, then move to the SafeSpeech baseline.

No expansion of scope mid-experiment.

## Deliverables

- `src/eval/nonadditive_transforms.py` — F-ratio selection and the two transforms
- `results/n1_fratio_bins.json`, `results/n1_target_pesq.json`, `results/n1_calib.json`
- `results/anticlone/prompts/RQA/{n1notch,n1rir,n1both}/`
- `results/ce_ac_f5tts_RQA_{n1notch,n1rir,n1both}.json`, plus MaskGCT if HN3 runs
- quality table: PESQ, WER, STOI, SI-SNR, RMS for every arm beside Q2
- an updated protection-versus-PESQ figure with the N1 arms on the same axes as the six additive families
- a result document giving the verdict against HN1–HN4, failures included
