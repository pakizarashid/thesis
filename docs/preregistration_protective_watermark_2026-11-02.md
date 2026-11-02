# Pre-registration P1 — a watermark trained to protect

Written 2026-11-02, before any protective-watermark run. Code: `src/train_route2_clone_aware.py` (new
`--lambda_protect`, `--protect_verifier`), evaluated with the existing harness.

## The observation this is built on

The R0 watermark (clone-aware, no quality terms) lowers clone acceptance **with no perturbation applied at all**:

| | F5-TTS | MaskGCT | CosyVoice | needs a surrogate at publish time? |
|---|---|---|---|---|
| Q2's PGD effect (wm → q2, RQA) | −14 | −20 | −9 | yes: 40 steps, ~60 s per clip |
| **R0 watermark alone** (R0 wm vs RQA wm) | **−12** | **−23** | **−7** | **no** |

An embedded signal achieves protection comparable to 40 steps of PGD, evenly across three architectures, without
any cloner-specific optimisation. That is the generalisation property adversarial perturbation lacks here: our
own A0 arm reaches 0.27 acceptance white-box on YourTTS and 0.89 on MaskGCT, which it never saw.

**This effect was not designed.** It appears in R0 as a side effect and is currently reported as a confound in
`claude/t1-t2-verdicts-2026-10-30.md` and on slide 15. P1 asks whether it can be produced deliberately, and
whether it is a targeted identity effect or simply the consequence of a noisier watermark.

## Method

One term added to the existing clone-aware training loss. The clone tensor already exists in the loop
(`src/train_route2_clone_aware.py`, the `cloned = make_clone(...)` line), so the change is local:

```
L = lambda_clean * Ldec(recon_wm)
  + lambda_clone * Ldec(clone(recon_wm))
  + lambda_protect * cos( spk(clean), spk(clone(recon_wm)) )      <- new
  + the quality terms (mel, cos, adv, aug, vad), unchanged
```

`spk` is ECAPA-TDNN, differentiable, the same loader the perturbation uses. Resemblyzer is the held-out judge
and never enters this loss, as everywhere else in this work.

`--train_msg_processor` is required: without it the in-loop clone is detached and no gradient reaches the message
processor, so the protective term would have no path to the embedded signal.

## The control that decides it, fixed now

The danger is the one that killed T1: an effect that is really just extra perturbation energy. Here the
equivalent risk is that a protective watermark protects only because it is louder.

So the primary comparison is **at matched audio quality**, not at matched hyperparameters:

- `lambda_protect` ∈ {0.5, 2.0} trained; **`lambda_protect` = 0 is the existing R0 checkpoint**, reused.
- Each arm's clean-audio PESQ is measured on the standard 100 clips.
- The comparison is protection **against PESQ**, the same curve analysis used in
  `claude/hmask-result-2026-10-22.md`.

**P1 passes only if a protective arm sits above the curve traced by quality loss alone** — that is, if at the
same PESQ it protects more than a watermark that merely degrades audio by the same amount.

If the arms land on the curve, the effect is loudness again, P1 fails, and slide 15's "side effect" wording
stands unchanged.

## Pilot before the full runs

Five epochs at `lambda_protect` = 2.0, checked for three things before any 20-epoch run is started:

1. the protective term falls over training (the gradient reaches the message processor at all);
2. clean-audio ACC stays ≥ 0.90 (the watermark is still readable in the published file);
3. in-loop ECAPA similarity on the eval set falls relative to `lambda_protect` = 0.

If (1) or (2) fails the design is wrong and P1 stops here; no re-tuning on results.

## Hypotheses

**HP1 (primary).** At matched PESQ, a protective watermark lowers MaskGCT acceptance by ≥ 5 points more than the
`lambda_protect` = 0 watermark at the same PESQ, with exact McNemar p < 0.05 after Holm across cloners.

**HP2 (traceability).** Traced rate on MaskGCT stays within 0.05 of R0's. A watermark that protects but cannot be
read has abandoned the thesis's own claim.

**HP3 (generalisation, the point of the whole direction).** The effect appears on all of F5-TTS, MaskGCT,
CosyVoice 2, XTTS-v2 and YourTTS without any per-cloner tuning, and its spread across cloners is **no wider than
Q2's**. Q2's spread is 9–20 points; a protective watermark that only works on one architecture fails this and is
reported as such.

**HP4 (descriptive).** Compare against Q2 directly: does the protective watermark alone, with no PGD, reach Q2's
acceptance? If so the perturbation becomes optional, which is the practical claim.

## Predictions, before the run

1. P(HP1 passes) ≈ 40 %. Higher than the perturbation experiments because the mechanism is different in kind
   rather than a reshaping of the same noise — but the loudness explanation is live and has won five times.
2. P(HP2 holds) ≈ 70 %. The clone-aware term and the protective term pull in opposite directions on the same
   latents; this is the real tension in the design.
3. P(HP3 holds, given HP1) ≈ 75 %. The effect is not surrogate-specific by construction.
4. Clean ACC ≥ 0.90 in the pilot ≈ 85 %.

## What this cannot show

- It cannot show that embedded protection beats adversarial perturbation in general — one codec, one watermark
  architecture, one corpus.
- It does not test the fine-tuning threat model (SafeSpeech's setting); the attacker here is zero-shot.
- A pass does not make the perturbation redundant unless HP4 also holds.
- The in-loop verifier is ECAPA, so the same inflation caveat applies: only the held-out Resemblyzer numbers are
  evidence.

## Deliverables

- `checkpoints/route2_protect_{0.5,2.0}/`
- clean-audio PESQ and ACC for every arm on the standard 100 clips
- `results/ce_ac_*_P{0.5,2.0}_wm.json` for all five cloners
- a protection-versus-PESQ figure with R0, RQA and the protective arms on one axis
- a result document stating HP1–HP4 against the rules above, failures included
