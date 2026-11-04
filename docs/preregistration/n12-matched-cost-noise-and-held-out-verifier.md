# Pre-registration N12 -- matched-cost noise, and a verifier never in the loop

Written 2026-11-05, BEFORE any N12 audio exists and before any N12 score.

TWO STRUCTURAL DEFECTS BEING FIXED.
(1) rs is NOT PESQ-matched to the notch. It was built as the matched control for c1
(1.6218 vs 1.6396). The notch is 1.9583, a gap of 0.337 -- 7x N8's tolerance. Every
"beats/loses to matched-cost noise" statement compared arms a third of a PESQ apart,
and the traceability comparison (rs 0.620 vs notch 0.980) is confounded the same way.
(2) ONLY RESEMBLYZER WAS EVER HELD OUT. Q2's loop contained ECAPA,
microsoft/wavlm-base-plus-sv and speechbrain/spkrec-xvect-voxceleb
(f5_whitebox_pgd.py::make_verifiers) -- every verifier N9 and N11 treated as an
independent arbiter. N11's claim-4 withdrawal is retracted.

STAGE A -- VALIDATE THE MEASUREMENT FIRST. rs, wm and n1notchK12 audio are on disk, so
recompute PESQ against the same clean references and compare with recorded values
wm 2.2315, n1notchK12 1.9583, rs 1.6218.
HN37 (validity gate): all three must reproduce within 0.05, else N12 HALTS -- no arm is
built on an unvalidated measurement.
Stage A also measures the residual rs - wm per clip (max|r|, sd, fraction at +-max) to
determine rs's construction EMPIRICALLY rather than by guesswork. The matched arm is
built in rs's own measured family, scaled down.

STAGE B -- rsmatched: rs's measured construction with amplitude BISECTED until mean PESQ
over the same 100 clips is within 0.05 of 1.9583. RMS-matched, sample count preserved.
PESQ only, no model, no GPU.
HN38 gate: if bisection cannot reach the target in 12 iterations, the arm is not built
and claims 3 and 4 are reported untestable by this route.

STAGE C -- THE HELD-OUT VERIFIER. Provenance assertion ENFORCED IN CODE: a candidate is
rejected outright if it is in Q2's in-loop set {ECAPA, microsoft/wavlm-base-plus-sv,
speechbrain/spkrec-xvect-voxceleb}. Resemblyzer is also excluded -- it is the incumbent,
not a second opinion. Primary candidate: speechbrain/spkrec-resnet-voxceleb (ResNet not
TDNN, never in any loop, speechbrain already installed so a failed install cannot
silently bias the set).
SENSITIVITY GATE REBUILT WITHOUT Q2 -- Q2 cannot be a positive control for any verifier
it was optimised against. The replacement control is rs, never tuned against anything:
  G1  clean-audio EER <= 0.10
  G2  unprotected wm accept >= 0.80   (headroom must exist)
  G3' |accept(rs) - accept(wm)| >= 0.10  (can it see a large UNOPTIMISED perturbation?)
Enrolment is the FULL clean utterance for every verifier, as clone_eval_v2 already does;
N11 showed the 3 s crop was the dominant handicap. Per-verifier EER thresholds are
computed on clean audio only and frozen before any arm is scored.

HYPOTHESES
HN39 (PRIMARY) -- does the notch beat noise AT MATCHED COST? n1notchK12 vs rsmatched,
F5-TTS, unseen, n=100, RESEMBLYZER at 0.6868 (the only held-out incumbent). All must hold:
  1. rsmatched PESQ within 0.05 of 1.9583
  2. accept(notch) <= accept(rsmatched) - 0.05
  3. one-sided exact McNemar p < 0.05
  4. speaker-bootstrap 95% CI (10,000 draws, 20 speakers) excludes 0
HN40 -- the same comparison on any ADMITTED never-in-loop candidate, threshold-free plus
accept at its frozen EER threshold.
HN41 -- CLAIM 3 AT MATCHED COST. Published-file traceability (acc_source >= 13/16, from
the clone-eval records) and clone-survival traced, for n1notchK12 vs rsmatched. rs's
0.620 was measured 0.34 PESQ away and does not count.

PRE-DECLARED OUTCOMES
| HN39 passes | CLAIM 4 IS ESTABLISHED on the one held-out verifier: the notch beats additive noise at matched perceptual cost |
| HN39 fails | CLAIM 4 IS REFUTED on the held-out verifier; the anti-cloning contribution rests on claim 1 plus cost/compute, not on beating noise |
| HN39 passes, HN40 fails | verifier-dependent; no majority claimed, in-loop/held-out distinction given prominently |
| no candidate passes the gate | HN40 not evaluated; the absence of an independent arbiter is reported as a project-level limitation |
| HN41 shows rsmatched traced ~ notch traced | CLAIM 3 COLLAPSES -- traceability preservation was a cost artefact |
| HN37 or HN38 gate fails | N12 halts; claims 3 and 4 stay untestable and nothing is inferred |
Rows 2 and 5 matter most, and both are written now: this run can take away claim 3 as
well as claim 4.

PREDICTIONS (not adjusted by recent calls; the N11 miss was wrong mechanistic reasoning,
not a miscalibrated prior): P(HN37 passes) ~ 85%; P(bisection reaches target) ~ 90%;
P(HN39 passes) ~ 50% -- genuinely uncertain: rs at 0.337 worse PESQ already gives
Resemblyzer 0.900 vs the notch's 0.690, so weakening noise to match should favour the
notch, but 0.690 is the quantity under test and Resemblyzer is unusually sensitive to F0
removal; P(resnet passes the gate) ~ 70%; P(HN41 favours the notch) ~ 65%.

WHAT THIS CANNOT SHOW. One cloner, 100 clips, 20 held-out speakers, one K, one
checkpoint, one noise family. PESQ is one proxy and c1's audible echo shows it disagrees
with listening; a listening comparison of notch vs rsmatched is the natural follow-up and
is NOT run here. Two independent verifiers is still a small sample. No comparison to
SafeSpeech, CloneShield, AntiFake, E2E-VGuard or VoiceMark is implied.

REPORTING RULES, plus one new one. Accept rates never reported for one verifier alone.
Thresholds not changed after seeing results; no candidate added after the gate is
evaluated. Differences below 0.026 within a session and 0.09 across sessions are not
evidence. NEW -- VERIFIER PROVENANCE: every pre-registration treating a verifier as held
out must state which optimisation loops it was absent from, and the cell must assert it.
This is the rule whose absence produced the N9 and N11 errors.
