# Pre-registration L3 — clone-identity test with a full-length reference

Written 2026-11-04, after L2 and before any L3 trial is played.

## Why L2 could not answer the question

L2 ran as registered on 2026-11-04 and returned **HL2 REFUTED**: c1 "same person" rate 0.000 against wm's 0.250,
a difference of 0.250 against a registered threshold of 0.15. That verdict stands and is not revised.

A sensitivity analysis performed after the fact makes it uninterpretable:

| | hit rate | false-alarm rate (impostor catches) | d′ |
|---|---|---|---|
| wm (unprotected) | 2/8 = 0.250 | 2/8 = 0.250 | **0.00** |

The rater was **at chance on the unprotected baseline** — a genuine clone and an impostor clone were equally likely
to be called "same person". HL2 assumed listeners identify unprotected clones reliably and asked whether protection
reduces that. With the baseline at chance there is nothing for protection to reduce, so the comparison measures
nothing about protection. Eight trials per arm also cannot resolve a 0.15 threshold: 0/8 against 2/8 is two clips,
Fisher's exact p ≈ 0.47.

## The suspected cause, and what L3 changes

**One change only: the reference.** L2's reference was the 3-second prompt — the same short excerpt used to drive
the cloner. That may be too little audio for a listener to form an impression of a voice at all.

L3 uses the **full-length LibriSpeech utterance** (typically 10–15 s) of the same speaker as the reference. Sample
clips, arms, clip indices, trial counts and the response scale are unchanged from L2.

Everything else held constant is deliberate: if L3 differs from L2, reference length is the only candidate cause.

## Design

- **24 genuine trials** — 8 clips × 3 arms (`wm`, `n1notchK12`, `c1`), sample = the F5 clone.
- **8 impostor catch trials** — sample = a `wm` clone of a *different* speaker.
- Reference = full-length clean utterance of the reference speaker.
- 32 trials, order randomised from a fixed seed, arm labels hidden until completion.
- Response: same person / different person / unsure.

**Validity, fixed in advance and stricter than L2's.** A rater's data is analysed only if **both**:
1. ≥ 6 of 8 catch trials answered "different person"; and
2. **d′ on the unprotected arm > 0.5** — hit rate on `wm` exceeds the false-alarm rate on catches by enough to show
   the rater can do the task at all.

Condition 2 is the one L2 lacked. Without it a floor effect is reported as a protection effect.

## Hypotheses

**HL3a (the prerequisite) — the task is doable with a full-length reference.**
d′ on `wm` > 0.5. If this fails, **no claim about protection is made**, and the reported finding is instead:
*listeners cannot identify the speaker of an F5 zero-shot clone made from a 3-second prompt, even against a
full-length reference and even with no protection applied.* That is a finding about the threat model, not about
this thesis's methods, and it would mean automatic speaker verification accept rates substantially overstate how
convincing these clones are to people — `wm` is accepted by the verifier at 0.950.

**HL3b (the original question) — evaluated only if HL3a passes.**
SUPPORTED if c1's "same person" rate is within 0.15 of `wm`'s: perception does not follow the verifier, which on
this material reports EER 0.060 → 0.202 and 38 % of c1 clones scoring no better than an impostor.
REFUTED if c1's rate falls more than 0.15 below `wm`'s: protection does reduce perceived identity.

## Pre-declared outcomes

| outcome | interpretation |
|---|---|
| HL3a fails | 3-second-prompt F5 clones are not humanly identifiable at all; ASV accept rates overstate the threat |
| HL3a passes, HL3b supported | decoupling confirmed: the verifier's measure does not track human perception |
| HL3a passes, HL3b refuted | protection reduces perceived identity — a stronger result than the verifier alone shows |
| catches failed | void; reported as void |

## Predictions

1. P(HL3a passes with the longer reference) ≈ 45 %. L2's floor may be the clone's fault rather than the
   reference's.
2. P(HL3b supported | HL3a passes) ≈ 65 %.
3. P(the author's catch performance ≥ 6/8 again) ≈ 85 %.

## Limits

- Eight clips per arm, one non-naive rater. Indicative, not a powered listener study, and reported as such.
- The author has now heard this material twice; familiarity may raise or lower performance and cannot be
  controlled for in a single-rater test.
- One cloner, one corpus, English, 3-second cloning prompts throughout.
- No claim about real-world deception is licensed from forced-choice judgements on short clips.
