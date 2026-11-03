# Pre-registration N4 — noise-free composition, and MaskGCT confirmation of c1

Written 2026-11-03, **after** the F5 c1 result and the author listening check, and **before** any c2 clip exists or
any MaskGCT c1 clone is run. What is already known is stated below so that it is clear what these hypotheses could
not have been fitted to.

## What is already known, and therefore not being tested again

- F5-TTS, unseen text, n=100, frozen thresholds (Resemblyzer 0.6868, ECAPA 0.2994). Worst case over the two
  verifiers: `wm` 0.950, `a0` 0.930, `rs` 0.900, `q2` 0.810, `n1notchK12` 0.810, `c1` 0.510.
- `c1` = q2 audio with the frozen K=12 notch. WC(c1) − WC(rs) = −0.391, CI [−0.520, −0.260], at a PESQ gap of 0.0178.
  So the composition gain is not a quality-cost artefact.
- **HN7 FAILED** on traced-in-clone (0.810 < 0.84). That verdict stands and is not revisited here.
- Author listening check, 2026-11-03: `q2` has audible added noise; `n1notch` at K=2, 6 and 12 has none; **`c1` has
  audible added noise**; all arms remain recognisably the same speaker to a human listener.

The mechanism behind c1 is that the two protections suppress *different* verifiers — q2 holds ECAPA at 0.050 and
leaves Resemblyzer at 0.810, the notch does the reverse. Nothing in that mechanism requires either component to be
additive. The listening result shows the cost of c1's additive half: it is the only reason c1 is noisy, because a
notch removes energy and RMS-matches while a perturbation adds energy.

**N4 asks whether the complementary-verifier gain survives when both components are non-additive.**

## The new arm

`c2` = frozen K=12 notch **plus** a synthetic RIR of length L ms (exponentially decaying noise, fixed seed),
applied to the **watermarked** audio — not to q2. Both operations are non-additive: one removes spectral energy, the
other convolves. Neither adds an independent noise signal.

`n1rir` and `n1both` were registered in N1 on 2026-11-03 as secondary arms with no decision rule and were never run.
`c2` is `n1both` promoted to a primary arm with a decision rule, which is a change of status recorded here before any
clip exists.

### Calibration of L, before any clone

L is **not** chosen by looking at protection. It is calibrated so that c2's PESQ equals **c1's measured PESQ of
1.6396**, making c2 and c1 comparable at matched perceptual cost — the same comparison that made the `rs` control
decisive.

- Grid L ∈ {5, 10, 20, 30} ms, as registered in N1. Clips 0–19, mean PESQ per setting.
- Interpolation only between **adjacent** brackets, and the chosen L is then verified by measurement on all 100
  clips. Non-adjacent interpolation across a dip is the error that put T1E's epsilon at the wrong value and that
  returned K=12 through a bracket spanning the K=8 dip.
- If the target lies outside the grid, the grid is extended once in the needed direction; if still outside, c2 is
  reported "not calibratable at c1's quality" and HN10 is not evaluated.
- Frozen to `results/n4_calib.json` before any decisive clip. No clone, verifier score or accept rate enters
  calibration.

## Hypotheses

Every condition is conjunctive and the verdict cell must evaluate all of them and print which failed.

**HN10 (primary) — protection without added noise.**
PASSES if, for `c2` on F5-TTS, **all** hold:
1. WC(c2) ≤ WC(c1) + 0.05 — c2 reaches c1's protection level;
2. WC(c2) − WC(`n1notchK12`) < 0 with speaker-bootstrap 95 % CI excluding 0 — the RIR must add something the notch
   alone does not, or c2 is just the notch;
3. PESQ(c2) within 0.05 of 1.6396, verified on all 100 clips;
4. STOI within 0.05 of c1's and WER within 0.05 of c1's;
5. **the author listening check reports no added noise in c2**, on the same 8-clip set and the same blinded player
   used on 2026-11-03.

Condition 5 is the point of the arm. An arm that reaches c1's protection while sounding like c1 is not a result.

**HN11 (the real risk) — the watermark survives convolution.**
An RIR smears the watermark across time, and the watermark lives in SpeechTokenizer layers 2–8.
PASSES if traced(c2) ≥ traced(`rs`) + 0.05 on the same clips.

This is a **matched-cost** traceability criterion, not HN7's floor. HN7 compared traced against q2 at a *higher*
PESQ, which is the same unequal-cost comparison the protection side was careful to avoid; that is a design
limitation of HN7, recorded in the c1 result document. The margin is 0.05 to mirror HN7's, chosen for continuity and
not because of any number already observed.

**HN12 — MaskGCT confirms c1.** On MaskGCT, unseen text, n=100: WC(c1) − WC(q2) < 0 with CI excluding 0, **and**
WC(c1) − WC(`n1notchK12`) < 0 with CI excluding 0. Cross-cloner spread in WC(c1) no wider than Q2's observed 9–20
points. A pass on F5 alone is not a pass. Both parents are required because a gain over q2 alone would not show the
composition does anything the notch could not.

**HN13 — matched-cost traceability for c1, on unseen data.** traced(c1) ≥ traced(`rs`) + 0.05 **on MaskGCT**.
The F5 matched-cost traced figures (rs 0.620, c1 0.810) are already known, which is exactly why this is tested on
MaskGCT and not re-run on F5.

## Pre-declared outcomes

| outcome | interpretation |
|---|---|
| HN10 and HN11 pass | noise-free composition reaches c1's protection with the watermark intact — the strongest result available to this thesis |
| HN10 passes, HN11 fails | convolution and traceability are incompatible; protection is available noise-free but not with provenance |
| HN10 fails on condition 5 | the RIR is audible too; no non-additive composition tested here is noise-free |
| HN10 fails on conditions 1–2 | the complementary-verifier gain requires the additive component; c1's noise is the price of its protection |
| HN12 fails | the c1 result does not generalise and is reported as F5-specific |

## Predictions, before any run

1. P(HN10) ≈ 30 %. Reverberation is a poor match to whatever ECAPA keys on, and the RIR may simply fail to suppress
   the verifier the notch leaves standing.
2. P(HN10 fails specifically on condition 5) ≈ 35 %. A 20–30 ms RIR is audible as colouration even though it adds no
   noise signal, and the listening question is "added noise", not "indistinguishable".
3. P(HN11 | HN10) ≈ 45 %. Smearing is the mechanism most likely to destroy a time-localised watermark.
4. P(HN12) ≈ 65 %. The complementary-verifier mechanism is not obviously cloner-specific.
5. P(HN13 | HN12) ≈ 70 %.
6. P(c2 not calibratable to 1.6396 within the grid) ≈ 30 %.

## What this cannot show

- One corpus, 100 clips, 20 held-out speakers, one watermark checkpoint, two cloners, two in-loop verifiers plus one
  held-out judge.
- Only Resemblyzer is held out from every method; ECAPA is in Q2's loop, so WC is a floor on q2-derived arms and not
  a verifier-independent measure. Worst case over two verifiers is not worst case over all verifiers.
- A failure of HN10 does not show noise-free protection is impossible — only that a notch plus a synthetic RIR, at
  c1's perceptual cost, does not reach c1's protection under this protocol.
- Nothing here licenses a comparison with SafeSpeech, CloneShield, AntiFake, E2E-VGuard or ClearMask, none of which
  has been run under matched conditions.
- The threat model remains zero-shot cloning. Fine-tuning attacks are not tested.

## Order of operations, enforced by the cell

1. Write this document to `docs/`, commit and push it, and verify it is tracked. Generation halts if not.
2. Calibrate L on clips 0–19 against the frozen target 1.6396; freeze `results/n4_calib.json`.
3. Generate `c2` on all 100 clips; verify PESQ and per-clip RMS match.
4. Clone with F5-TTS; evaluate at the frozen thresholds.
5. Evaluate HN10 and HN11, printing every condition and every failure.
6. Rebuild the blinded listening file including c2 for condition 5, which is scored by the author and is the only
   condition not decided by the cell.
7. MaskGCT runs separately for HN12 and HN13.

No expansion of scope mid-experiment.
