# Pre-registration N11 -- a sensitivity-gated third verifier

Written 2026-11-05, BEFORE any N11 score is computed.

N9 got HN28 UNRESOLVED (+0.0043 CI [-0.0072,+0.0179]) and HN29 FAIL (-0.0038
CI [-0.0084,+0.0004]) from WavLM-base-plus-sv, then turned out the instrument could
not have resolved either: clean EER 0.295, unprotected accept 0.520, response to Q2
only -0.090 against ECAPA's -0.890 and Resemblyzer's -0.150. It never bound the worst
case either, so it changed the registered primary statistic by zero.

TWO DEFECTS, BOTH MINE. (1) No sensitivity gate was pre-registered, so a verifier was
admitted without any requirement that it see a known-large effect. N9's outcomes cannot
be retroactively voided -- they stand as reported -- but the question stays open.
(2) UNEQUAL ENROLMENT: clone_eval_v2 enrols Resemblyzer and ECAPA on the FULL clean
utterance; N9's WavLM used the 3 s crop, because that is what PR/<arm>/clean/ holds.
WavLM was handicapped relative to the verifiers it was meant to arbitrate between.

NO NEW AUDIO. N11 re-scores results/clones_n9/, which HN30 confirmed faithful (fresh
Resemblyzer accepts within +-0.030 of frozen on all five arms).

THE GATE, FIXED NOW. A candidate is admitted as a tiebreaker only if ALL hold:
  G1 clean-audio EER <= 0.10          -- must separate speakers on the easy case
  G2 unprotected wm accept >= 0.80    -- headroom must exist for protection to show
  G3 |accept(q2) - accept(wm)| >= 0.10 -- positive control; a verifier that cannot see
                                         a 40-step PGD attack cannot see less
All measured BEFORE any notch-vs-noise comparison, all properties of the instrument
rather than the hypothesis. Q2 was optimised against ECAPA and against NEITHER
candidate here, so for these it is a clean sensitivity probe.
If NO candidate passes: the third-verifier question cannot be answered with the
candidates tried; claim 4 remains "Resemblyzer yes, ECAPA reversed, no majority".
NO further candidates are added afterwards -- that would be selection on the outcome.

CANDIDATES (zero new dependencies, so a failed install cannot silently bias the set):
  wavlm_full    microsoft/wavlm-base-plus-sv, FULL clean utterance   -- primary
  wavlm_crop3s  same, 3 s crop -- reproduces N9's config so the enrolment effect is
                visible. NOT a tiebreaker candidate; a control, reported either way.
  xvect_full    speechbrain/spkrec-xvect-voxceleb, full utterance -- second
                architecture (plain x-vector TDNN vs ECAPA-TDNN), already installed.
Full utterances are resolved from the cached LibriSpeech tree by utterance id. If fewer
than 90 resolve, N11 HALTS rather than silently falling back to crops.

PRIMARY STATISTIC STAYS THRESHOLD-FREE: paired per-clip cosine difference with a
speaker-level bootstrap CI (10,000 draws, 20 speakers). Accept rates at a per-verifier
EER threshold, from clean audio only and frozen before any arm is scored, are secondary
and are what G1-G3 are evaluated on.

HYPOTHESES
HN34 (gate) -- report EER, wm accept and Q2 effect per candidate; admit those passing
  G1-G3. Descriptive, but determines what follows.
HN35 (PRIMARY) -- for each ADMITTED verifier, n1notchK12 vs rs:
  sides with RESEMBLYZER if mean d < 0 and CI excludes 0
  sides with ECAPA       if mean d > 0 and CI excludes 0
  UNRESOLVED             if CI spans 0
HN36 -- for each admitted verifier, n1notchK12 vs n1randK12: mean d < 0, CI excludes 0.
  N10 established the criterion claim on Resemblyzer and ECAPA; failure on a GATED
  third verifier would genuinely weaken it.

PRE-DECLARED OUTCOMES
| HN35 sides with Resemblyzer on every admitted verifier | ECAPA is the outlier; CLAIM 4 BECOMES "holds on 2 of 3" |
| HN35 sides with ECAPA on every admitted verifier | CLAIM 4 IS WITHDRAWN; the Resemblyzer result is specific to that embedding |
| HN35 splits across admitted verifiers | reported as split; claim 4 stays "no majority"; verifiers genuinely disagree |
| HN35 unresolved on every admitted verifier | claim 4 stays "no majority" |
| no candidate passes the gate | unanswerable with these candidates; claim 4 stays "no majority" |
| HN36 fails on an admitted verifier | CLAIM 1 WEAKENS from "established" to "holds on Resemblyzer and ECAPA, fails on a sensitivity-gated third verifier" -- reported even though N10 passed decisively |
The last row matters: N11 is given real power to damage claim 1, not only to confirm it.

PREDICTIONS (adjusted upward; this project's predictions were over-pessimistic in N8,
N9 and N10): P(wavlm_full passes all gates) ~ 60%; P(xvect_full passes) ~ 65%;
P(>=1 passes) ~ 80%; P(HN35 sides with Resemblyzer | >=1 admitted) ~ 55% -- N10 showed
ECAPA separates the notch from random BIN SELECTION but not from broadband NOISE, which
reads as an ECAPA-specific insensitivity to narrow F0 removal rather than a general
truth; P(HN36 holds on every admitted) ~ 75%.

WHAT THIS CANNOT SHOW. One cloner (F5-TTS), 100 clips, 20 held-out speakers. The ECAPA
reversal was strongest on MaskGCT, XTTS and YourTTS, which this cannot speak to. Four
verifiers is still a small sample of speaker encoders. Cosine of embeddings is one way
to use each model. Nothing here bears on perceptual quality. No comparison to
SafeSpeech, CloneShield, AntiFake, E2E-VGuard or VoiceMark is implied.
