# Pre-registration N9 -- a third, independently held-out verifier

Written 2026-11-04, BEFORE any N9 clip is cloned or scored.

Two claims behave differently across the two verifiers measured so far:
 - F-ratio bins beat RANDOM BINS: holds on both (-0.250 Resemblyzer, -0.130 ECAPA on F5).
 - The notch beats matched-cost ADDITIVE NOISE: Resemblyzer only. On ECAPA rs is as
   protective or more, winning 3 of 5 cloners (mean delta -0.138 rs vs -0.090 notch).
Neither rs nor n1notchK12 saw either verifier, so contamination does not explain the
disagreement. Two verifiers split 1-1 give no majority. A third breaks the tie.
Q2's ECAPA figures are excluded: ECAPA was inside Q2's loop.

VERIFIER: microsoft/wavlm-base-plus-sv (WavLMForXVector), cosine of x-vectors, 16 kHz
mono. Architecturally independent of both incumbents (SSL waveform frontend + x-vector
head, vs Resemblyzer's GE2E LSTM over 40 log-mels and ECAPA's TDNN over 80 fbanks), and
not gated on HuggingFace. pyannote would aid the CloneShield comparison but is gated.
It enters no loss and no selection rule.

PRIMARY STATISTIC IS THRESHOLD-FREE. For clip i, s_X[i] = cos(WavLM(clone of arm X,
clip i), WavLM(clean enrolment i)). The statistic is the paired per-clip difference
d[i] = s_A[i] - s_B[i] with a speaker-level bootstrap CI (10,000 draws, 20 speakers).
Lower similarity = better protection, so mean d < 0 means A protects more than B.
Accept at an EER threshold is SECONDARY, for comparability with the other two tables.
The threshold is derived from CLEAN audio only (within-speaker genuine pairs,
cross-speaker impostor pairs), fixed at EER, frozen before any protected audio is
scored, and identical for every arm -- so it cannot change the direction of any
comparison, only absolute rates. Calibrating on the eval speakers is a disclosed
limitation, acceptable because this fixes an operating point for an off-the-shelf
verifier rather than tuning any method.

ARMS (re-cloned on F5-TTS, unseen text, n=100, --save_audio_dir):
  wm           watermark-only baseline for deltas
  rs           matched-cost additive noise -- THE DISPUTED COMPARISON
  n1notchK12   the method
  n1randK12    random bin selection -- confirms the criterion result
q2 is EXCLUDED: ECAPA was in its loop, so it is not a clean test of verifier transfer.
Outputs go to results/ce_ac_f5tts_RQA_<arm>_n9.json; the frozen N2/N7 JSONs are NOT
overwritten.

HYPOTHESES
HN28 (primary) -- which verifier does the third side with, on the disputed claim?
  For n1notchK12 vs rs on WavLM:
    sides with RESEMBLYZER if mean d < 0 and the 95% CI excludes 0
    sides with ECAPA       if mean d > 0 and the 95% CI excludes 0
    UNRESOLVED             if the CI spans 0
HN29 -- does the criterion result hold on a third verifier? n1notchK12 vs n1randK12:
  mean d < 0 with CI excluding 0, matching both existing verifiers.
HN30 -- is the re-clone faithful? F5 sampling is stochastic, so these are not the
  clones that produced the frozen numbers. Fresh RESEMBLYZER accepts must land within
  0.09 of wm 0.950, rs 0.900, n1notchK12 0.690, n1randK12 0.940. Outside that band
  invalidates the RE-CLONE, not the arm, and HN28/HN29 are not evaluated. The band is
  taken unchanged from the variance study and is not adjusted after seeing results.

PRE-DECLARED OUTCOMES
| HN28 sides with Resemblyzer | ECAPA is the outlier; claim 4 stands as "holds on 2 of 3", ECAPA exception reported |
| HN28 sides with ECAPA | CLAIM 4 IS WITHDRAWN: the notch does not beat matched-cost additive noise on held-out verifiers generally, and the Resemblyzer result is specific to that embedding. Reported as a negative result; the contribution rests on claims 1-3 |
| HN28 unresolved | reported as unresolved, NO claim either way |
| HN29 holds | the criterion result is verifier-robust across three verifiers |
| HN29 fails | N7's result is Resemblyzer-and-ECAPA-specific |
| HN30 fails | re-clone rejected, nothing else evaluated, run repeated |
Row 2 is the one that matters and is written now, before the run.

PREDICTIONS, BEFORE ANY RUN
1. P(HN28 sides with Resemblyzer) ~ 45%. Resemblyzer's 40-mel frontend plausibly leans
   harder on the F0 region the notch removes, while WavLM's learned waveform frontend is
   likely more robust, like ECAPA. The lean is slightly AGAINST the method. Noted for
   calibration: this project's predictions have twice been over-pessimistic.
2. P(HN28 unresolved) ~ 20%
3. P(HN29 holds) ~ 75%
4. P(HN30 passes) ~ 85%

WHAT THIS CANNOT SHOW
One corpus, 100 clips, 20 held-out speakers, ONE cloner (F5-TTS) -- the ECAPA reversal
was strongest on MaskGCT, XTTS and YourTTS and this run cannot speak to those. Three
verifiers is a small sample of speaker encoders; a 2-1 split is suggestive, not decisive.
Cosine of x-vectors is one way to use WavLM. It says nothing about perceptual quality or
deployability. No comparison to SafeSpeech, CloneShield, AntiFake, E2E-VGuard or
VoiceMark is implied.
