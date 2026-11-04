# Pre-registration L2 — human clone-identity test

Written 2026-11-04, before any clone is played to any listener.

## Why

Every protection number in this project is a verifier's judgement. On F5-TTS the verifier now says a great deal:
c1 triples the EER (0.060 → 0.202), halves d-prime (3.61 → 1.62), and for **38 % of clones** scores the clone no
higher against its own speaker than against a stranger.

Nothing has asked whether a *person* agrees. Informal and blinded author listening says every clone still sounds
like the speaker, including c1's. If that holds under a proper test, the finding is that automatic speaker
verification and human identity perception come apart under these perturbations — which bears on every
accept rate in this literature, not only on this thesis.

## Design, fixed now

Material: the F5-TTS clones saved on 2026-11-04 (`results/clones_f5tts/`), unseen text, and the matching clean
reference recordings.

- **24 genuine trials** — 8 clips × 3 arms (`wm`, `n1notchK12`, `c1`). Reference = the clean recording of speaker
  S; Sample = a clone made from S's protected audio.
- **8 impostor catch trials** — Reference = speaker A's clean recording; Sample = a `wm` clone of a *different*
  speaker B.
- 32 trials total, order and left/right randomised from a fixed seed, arm labels hidden until the end.
- Response: **same person / different person / unsure**.

The catch trials are not padding. Without them a listener who answers "same" to everything is indistinguishable
from a listener for whom protection genuinely failed.

**Validity condition, fixed in advance:** a rater's responses are used only if they answer "different person" on
**≥ 6 of 8** catch trials. A rater below that is reported as excluded, with the count, and their genuine responses
are not analysed.

## Raters

The author's ratings are recorded but reported **as the author's**, never as a listener study: she knows the arms,
the hypothesis and the expected answer. Any naive listeners are reported separately and are the evidence that
carries weight. The test is designed so a naive rater needs no briefing beyond the on-screen instructions.

## Hypothesis

**HL2 — perception does not follow the verifier.**
SUPPORTED if, for raters passing the validity condition:
1. the "same person" rate for `c1` is within **0.15** of the rate for `wm`; and
2. the catch trials confirm raters can tell speakers apart (validity condition met).

If supported, the decoupling claim is made in this form: *for 38 % of c1 clones the verifier can no longer
distinguish the speaker from a stranger, while listeners identify the speaker at a rate indistinguishable from
unprotected clones.*

**REFUTED if** the `c1` "same person" rate falls more than 0.15 below `wm`'s — in which case the protection does
reduce perceived identity, the decoupling claim is withdrawn, and c1's perceptual effect is reported as a positive
result instead.

## Pre-declared outcomes

| outcome | interpretation |
|---|---|
| c1 within 0.15 of wm, catches passed | decoupling confirmed; the field's metric does not track human perception |
| c1 more than 0.15 below wm | protection reduces perceived identity; a stronger result than the verifier alone shows |
| catch trials failed | the test is void; no claim either way, and it is reported as void |
| author-only ratings | reported as a single non-blind rater, not as a listener study |

## Predictions

1. P(HL2 supported) ≈ 75 %.
2. P(the author passes the catch condition) ≈ 95 %.
3. P(`n1notchK12` sits between `wm` and `c1`) ≈ 60 %.

## Limits

- Eight clips per arm and few raters. This is an indicative test, not a powered listener study, and is reported as
  such.
- One cloner (F5-TTS), one corpus, English only.
- "Same person" judgements from short clips are not the same as being deceived in a real interaction; no claim
  about real-world deception is licensed.
- The author is not a naive rater.

## Order of operations

1. Commit this document before the player is opened.
2. Rate all 32 trials; the key stays hidden until every trial is answered.
3. Export the CSV; report catch-trial performance first, then the per-arm rates.
4. Report the author's ratings separately from any naive raters'.
