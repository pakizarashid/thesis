# Pre-registration N3 — composing the notch with Q2

Written 2026-11-03, before any N3 clip exists and before the code change it needs. Extends
docs/preregistration_nonadditive_2026-11-03.md (N1, A1-A5). Everything not restated is inherited.

## Why, stated as the problem it fixes

N1 does not dominate Q2. Measured on F5-TTS, unseen text, 100 clips:

| | accept @ Resemblyzer | accept @ ECAPA | worst case |
|---|---|---|---|
| wm (unprotected) | 0.950 | 0.950 | 0.950 |
| q2 | 0.810 | 0.050 | 0.810 |
| n1notch K=12 | 0.690 | 0.810 | 0.810 |

Q2 collapses ECAPA similarity and barely moves Resemblyzer; the notch does the reverse. A defender does not
choose which verifier an attacker uses, so the operationally honest figure is the worst case over verifiers,
and there the two methods are TIED at 0.810. N1's advantages are real but they are quality, compute (about 1 s
per clip against Q2's 40-step PGD at ~60 s, no differentiable surrogate) and cross-architecture consistency --
not worst-case protection.

The two mechanisms attack different things. N3 asks whether they add.

## Primary statistic, fixed now

Worst-case accept across the two calibrated verifiers, per arm:

    WC = max( accept @ Resemblyzer 0.6868 , accept @ ECAPA 0.2994 )

Chosen because it is the only figure that does not let a method hide behind the verifier it was optimised
against. Both thresholds were frozen long before N1 existed (2026-10-26, 2026-10-28). Reported beside it, but
not primary: each verifier separately, and the N1-style curve offset.

## Arms

| arm | construction | new PGD compute |
|---|---|---|
| c1 (primary) | apply the frozen K=12 notch to Q2's existing prompts | none |
| c2 (secondary) | re-run Q2's PGD starting from the notched audio | ~100 min |

c1 is primary because it is nearly free and detects composition if composition exists. c2 is the principled
version -- PGD can adapt to the notched signal instead of being applied first and partly removed -- and runs
ONLY if c1 is promising or ambiguous.

K is NOT swept. K = 12 is fixed from N1. N3 introduces no new free parameter.

## Decision rules, fixed now

The verdict function evaluates every condition and prints each one's pass/fail.

HN7 (primary) -- composition beats both parents on worst-case protection. On F5-TTS, PASSES if all:
  * WC(c1) <= 0.760, i.e. at least 0.05 below the 0.810 both parents tie at;
  * the speaker-bootstrap 95% CI for WC(c1) - WC(q2) excludes 0, and likewise against WC(n1notchK12);
  * traced >= 0.84;
  * clean-audio PESQ reported, with no floor -- see below.

NO QUALITY GATE, deliberately, and this is a weakening that must be disclosed. c1 stacks two mechanisms, so
its PESQ will be worse than either parent (Q2 is 1.7887; the notch costs ~0.27 from wm; expect roughly 1.6).
N1's whole argument was protection at BETTER quality, and c1 abandons that. It is registered anyway because
worst-case protection is the quantity the thesis claims to care about, but a c1 pass is a DIFFERENT claim from
N1's: "the mechanisms compose" rather than "protection gets cheaper". Both numbers go in the table and neither
is reported without the other.

HN8 (the real risk) -- the watermark survives two stacked mechanisms. Q2 already runs at traced 0.89 on F5-TTS.
Notching on top may push it under. traced and acc_source are reported per arm; traced < 0.84 fails HN7 and is
reported as the two mechanisms competing for the same signal, which is itself a finding about combined systems.

HN9 (confirmation) -- if HN7 passes on F5-TTS, MaskGCT confirms, same rule as A5's HN3. A pass on one cloner is
not a pass.

## Predictions, before any run

1. P(HN7 passes for c1) ~ 40%. The mechanisms are mechanistically independent -- additive energy against
   spectral removal -- which argues for addition; but Q2's perturbation lives partly in the bins the notch
   zeroes, so the notch may remove the perturbation along with the F0 energy.
2. P(c1 PESQ below 1.65) ~ 65%.
3. P(HN8 holds, traced >= 0.84 for c1) ~ 55%. The least comfortable prediction: Q2 starts at 0.89 with only
   0.05 of margin.
4. P(c2 beats c1 on WC, if both run) ~ 70%.
5. P(either composition arm reaches WC <= 0.50) ~ 15%. Even a pass leaves most clones accepted by some
   verifier. Neither method nor their composition is close to stopping cloning outright, and the thesis must
   not imply otherwise.

## What this cannot show

- Two verifiers define the worst case. A third verifier (N2's HN6) could lower either parent's WC and change
  the comparison. If HN6 completes first, its verifier joins the WC definition and that is stated, not quietly
  folded in.
- One corpus, 100 clips, 20 held-out speakers, one watermark checkpoint, one or two cloners.
- c1 tests one order of composition. A c1 failure does not show the mechanisms are incompatible, only that
  notch-after-PGD does not add -- which is what c2 exists to separate.
- Nothing here licenses a comparison with SafeSpeech, ClearMask or AntiFake.

## Deliverables

- src/eval/nonadditive_transforms.py with --prompt_in and --out_arm_name (needed by c1)
- results/anticlone/prompts/RQA/c1/, and c2/ only if run
- results/ce_ac_f5tts_RQA_c1.json, plus MaskGCT under HN9
- a worst-case table: WC, both verifiers separately, traced, acc_source, PESQ, STOI, WER for
  wm, q2, n1notchK12, c1
- a result document giving HN7-HN9 against the rules above, failures named
