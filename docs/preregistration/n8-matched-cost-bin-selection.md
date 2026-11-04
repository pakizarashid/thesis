# Pre-registration N8 -- does the criterion survive at matched perceptual cost?

Written 2026-11-04, after N7's verdict and BEFORE any N8 clip exists.

N7 established that random bin selection gives no protection (accept 0.940 vs a
watermark-only 0.950; McNemar 25/0; CI -0.249 [-0.400,-0.110]). HN22 passed.
HN23 FAILED: PESQ 1.9583 (F-ratio) vs 2.2134 (random), |d| 0.2551 against a 0.10
tolerance. The F-ratio arm is the MORE costly one, so N7 alone cannot separate
"the criterion" from "more damage". N7's pre-registered consequence was fixed in
advance: the criterion claim is weakened and reported at matched PESQ instead.

Counter-evidence already in hand, not sufficient alone: n1energyK12 has the worst
PESQ of N7's arms (1.8552) and only intermediate protection (0.790), so cost does
not order protection.

## Matching in two directions

Direction B (PRIMARY) -- lower the F-ratio K until PESQ rises to 2.2134.
Sweep K in {6,7,8,9,10,11}; pick the K minimising |PESQ - 2.2134|. The F-ratio
prefix is deterministic: no draw-to-draw variation, no contamination risk. N6
found accept flat from K=12 to K=96 but measured nothing below 12.

Direction A (secondary) -- raise the random K until PESQ falls to 1.9583.
Sweep K in {16,24,32,48,64,96}, nested prefixes of a permutation whose first 12
bins ARE N7's measured random-12 set, so the curve is anchored at 2.2134.

Direction A carries a contamination risk. With 988 eligible bins a random set of
size K contains at least one of the F-ratio top-12 with probability
~1-(1-K/988)^11: about 42% at K=48, 66% at K=96. A contaminated set may protect
for the right reason, failing HN25a for a reason unrelated to cost.

MANDATORY AUDIT for Direction A: report (i) how many F-ratio top-12 bins the
matched set contains and (ii) how many of its bins lie below 200 Hz. If either is
>= 1, a second matched arm is drawn at the same K from a pool EXCLUDING the
F-ratio top-12, and both are reported.

## Hypotheses

HN25b (PRIMARY) -- the criterion survives at matched cost, matching downward.
For n1notchK<b> vs n1randK12 (PESQ matched within 0.05), F5-TTS, unseen, n=100,
Resemblyzer at 0.6868, ALL must hold:
  1. accept(notch, matched K) <= accept(randK12) - 0.05
  2. one-sided exact McNemar p < 0.05
  3. speaker-bootstrap 95% CI (10,000 draws, 20 speakers) excludes 0

HN25a (secondary) -- matching upward agrees. Same three conditions for
n1notchK12 vs n1randK<a>, with the contamination audit and a Holm-corrected p.

HN26 -- N7's single draw was not a fluke. Across four random-12 draws (N7's plus
seeds 2-4) report the accept range, mean and sd, and whether 0.690 lies below all
four. Per-draw McNemar, Holm-corrected. DESCRIPTIVE: four draws do not support a
p-value over the space of random selections and none will be computed.

HN27 -- traced_source >= 0.95 for every new arm.

FEASIBILITY GATE: if a sweep cannot bring PESQ within 0.05 of its target, the
corresponding hypothesis is NOT EVALUATED and that is reported as a failure to
match, not as a pass or a fail. PESQ is assumed monotone in K; this is not
guaranteed and the full curve is reported either way.

## Pre-declared outcomes

| outcome | interpretation |
|---|---|
| HN25b passes | criterion claim RESTORED at matched cost; HN23 caveat discharged |
| HN25b fails | perceptual COST explains N7's gap; the criterion claim is withdrawn and the thesis reports that protection tracks perceptual damage in the F0 band rather than the ranking |
| HN25b passes, HN25a fails, contamination >= 1 | upward match uninformative; HN25b stands, contamination reported |
| HN25b passes, HN25a fails, contamination = 0 | the directions disagree; NEITHER claim is made, reported as unresolved |
| HN26 puts 0.690 inside the random range | N7's draw was unrepresentative; criterion claim withdrawn REGARDLESS of HN25b |
| neither sweep matches | HN23 caveat becomes permanent; the claim rests only on N7's 25/0 and the energy arm's inverted ordering, and is reported as weakened |

Rows 2 and 5 are the ones that matter. Written now because they are the outcomes
that would most change what this thesis claims.

## Predictions, before any run

1. P(Direction B matches within 0.05) ~ 85%
2. P(Direction A matches within 0.05 by K=96) ~ 45% -- random high-frequency bins
   are low-energy and may never degrade PESQ to 1.9583
3. P(HN25b passes) ~ 60% -- the energy arm favours the criterion, but N1 may need
   the whole F0 band, and K=8 or 9 may not cover it
4. P(HN26 leaves 0.690 below all four draws) ~ 85%
5. P(Direction A's matched set is contaminated, if it matches) ~ 50%

## What this cannot show

One corpus, 100 clips, 20 held-out speakers, one cloner, one checkpoint. PESQ is
one perceptual proxy: matching PESQ does not match perceived quality, and c1's
audible per-word echo is a standing reminder that PESQ and listening disagree. No
listening test is run here. Four random draws characterise the random-selection
space only coarsely. It cannot show the F-ratio is the BEST criterion -- N7 found
it statistically indistinguishable from naive energy selection
(-0.099 [-0.300,+0.100]) and nothing here revisits that. No comparison to
SafeSpeech, CloneShield, AntiFake, E2E-VGuard or VoiceMark is implied.

## Reporting rules carried forward

Accept rates are never reported for one verifier alone; both Resemblyzer and
ECAPA are printed for every arm. Resemblyzer stays held out of every loss and
every selection rule. Within-session differences below 0.026 and across-session
differences below 0.09 are not evidence. Thresholds are not changed after seeing
results.
