# Pre-registration N10 -- separating the BAND from the RANKING

Written 2026-11-04, BEFORE any N10 clip exists.

N8 showed matched PESQ is UNREACHABLE by varying K. Direction B (F-ratio at K<12,
target 2.2134): closest point K=6 at PESQ 2.0311, |d| 0.1823, and the curve plateaus
from K=9 to K=12 (1.9519-1.9601), so all the perceptual cost is paid by K~8; reaching
2.2134 would need K~3-4. Notching six F0 bins already costs more than twelve random
bins. Direction A (random nest at K>12, target 1.9583): monotone but still 0.0782
short at K=96, matching only around K~130-145, where (i) contamination probability is
~78% and (ii) a 130-bin notch is not a comparable intervention to a 12-bin one even at
equal PESQ. Both HN25 directions were NOT EVALUATED.

So the HN23 cost confound is still open: the notch costs ~0.25 more PESQ than
random-bin notching and protects ~0.255 more, and nothing separates those.

N8 did strengthen the other side: four independent random-12 draws gave accept
0.940/0.940/0.950/0.950 (sd 0.006, McNemar 25-26/0 each) -- random bin selection
protects NOT AT ALL against the unprotected 0.950.

The untested question underneath: is the contribution the RANKING, or merely the BAND
the ranking selects? Both of N7's non-random arms land low in frequency (F-ratio
86-164 Hz, energy 172-250 Hz) and were statistically indistinguishable
(-0.099 [-0.300,+0.100]).

## Design

Pool: eligible bins with centre frequency in 80-500 Hz (bins ~11-64 at 7.8125 Hz),
EXCLUDING the F-ratio top-12. K = 12, the frozen operating point. Three independent
draws, seeds 11, 12, 13. Everything else inherited: same STFT (2048/512 Hann), -60 dB
floor, RMS-matched, sample count preserved, A5 curve design, checkpoint
route2_epoch19.pt SHA-256 verified, n1notchK12 must reproduce PESQ 1.9583 first.

INHERENT LIMITATION, stated now: excluding the F-ratio top-12 NECESSARILY shifts the
pool upward in frequency, because the F-ratio is near-monotone decreasing with
frequency and its top-12 are the lowest admissible bins. So "frequency-matched" means
ADJACENT-band, not identical-band, and the pool overlaps N7's energy arm (bins 22-32,
52). Overlap with the energy set is reported per draw. The operative control is
therefore PESQ, not frequency: a draw is only evaluated if its PESQ lands within 0.10
of 1.9583. Frequency matching is the mechanism for achieving that, not the claim.

SECOND LIMITATION: the pool holds only ~43 bins, so three 12-bin draws share ~3.3 bins
pairwise in expectation. The draws are CORRELATED, not independent like HN26's four,
and pairwise overlaps are reported.

## Hypotheses

HN31 (PRIMARY) -- the ranking matters beyond the band. For n1notchK12 against each
frequency-matched draw, F5-TTS, unseen, n=100, Resemblyzer at 0.6868, ALL must hold:
  1. PESQ of the draw within 0.10 of 1.9583 -- the cost control. If it fails, that draw
     is NOT EVALUATED, exactly as in N8.
  2. accept(F-ratio) <= accept(draw) - 0.05
  3. one-sided exact McNemar p < 0.05, Holm-corrected across the evaluated draws
  4. speaker-bootstrap 95% CI (10,000 draws, 20 speakers) excludes 0
Reported as passing only if it holds on EVERY evaluated draw. A split result is
reported as split, not rounded up.

HN32 -- ECAPA agrees. Same direction on ECAPA at 0.2994, as in N7 (-0.130 there).
Secondary; accept rates for both verifiers are printed for every arm.

HN33 -- traced_source >= 0.95 for every new arm.

## Pre-declared outcomes

| HN31 passes at matched PESQ | THE RANKING IS THE CONTRIBUTION. The HN23 cost confound is discharged by a route that works, and N1 is a selection criterion rather than ClearMask's mechanism re-run |
| HN31 fails at matched PESQ | ANY LOW-FREQUENCY BINS WILL DO. The F-ratio is unnecessary, the contribution reduces to "notch the F0 band", and since ClearMask already notches a fixed band the novelty largely disappears. The thesis reports N1 as a reproduction with a verifier-independent justification for WHERE to notch |
| no draw lands within 0.10 PESQ | matched cost unreachable by this route too; the HN23 caveat is permanent and the criterion claim rests on N8's 4/4 random result and the energy arm's inverted ordering, reported as weakened |
| HN31 splits across draws | reported as split; no claim either way, per-draw numbers given |

The second row is the one that matters and is written now, before the run. It is a
live possibility: the energy arm already protects more than random (0.790 vs 0.945)
using bins the F-ratio did not choose.

## Predictions, before any run

1. P(at least two draws land within 0.10 of PESQ 1.9583) ~ 65%. The energy arm, drawn
   from an overlapping region, came in at 1.8552 -- 0.103 away, just outside. The pool
   reaches higher in frequency than the energy top-12, so draws should be slightly
   cheaper and land closer.
2. P(HN31 passes on every evaluated draw) ~ 55%. The energy arm points the right way
   but its CI spanned zero, and a random low-band draw will likely land between the
   energy arm's 0.790 and the notch's 0.690.
3. P(HN32 agrees, if HN31 is evaluated) ~ 70%.
4. P(HN33 passes) ~ 90%.
Predictions in this project have twice been over-pessimistic; prediction 2 is close to
a coin flip either way.

## What this cannot show

One corpus, 100 clips, 20 held-out speakers, one cloner (F5-TTS), one K, one watermark
checkpoint. Three correlated draws characterise the low-band random space coarsely.
"Frequency-matched" is adjacent-band, not identical-band. PESQ is one perceptual proxy
and c1's audible per-word echo is a standing reminder that PESQ and listening disagree;
no listening test is run here. A pass would show the ranking beats random selection
WITHIN the F0 region; it would still not show the F-ratio is the BEST criterion -- N7
found it indistinguishable from naive energy selection and nothing here revisits that.
No comparison to SafeSpeech, CloneShield, AntiFake, E2E-VGuard, ClearMask or VoiceMark
is implied.

## Reporting rules carried forward

Accept rates are never reported for one verifier alone; both are printed. Resemblyzer
stays held out of every loss and every selection rule. Differences below 0.026 within a
session and 0.09 across sessions are not evidence. Thresholds are not changed after
seeing results.
