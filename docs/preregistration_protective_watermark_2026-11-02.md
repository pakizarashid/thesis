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

## Amendment A3 (2026-11-03, written before any run under A2) — two defects in the pilot, not in the idea

A2 kept A1's gate: `mean_sim_clone` must fall by >= 0.05. Two separate defects mean neither A1 nor A2 could have
answered P1's question. A2 is **put on hold** and this amendment comes first. No rank-32 clip exists at the time
of writing.

### Defect 1 — the pilot changed the clone-text protocol (a missing flag)

`src/train_route2_clone_aware.py` takes `--clone_text {fixed,sampled}`, **defaulting to `fixed`**. The pilot
command omitted it. Both pilot logs therefore open with:

> `[main] clone text: fixed sentence 'This is a test sentence for voice cloning.'`

while R0 — the checkpoint these runs continue — was trained with `sampled`, confirmed by the presence of
`checkpoints/route2_vmaug_R0/clone_text_pool.json` (60 transcripts; `route2_vmaug_RQA` likewise). That file is
written only on the `sampled` path.

So every in-loop clone in both pilots spoke the **same nine-word sentence**, for every speaker, at all ~880
steps of all 5 epochs. Three consequences:

1. The protective gradient came from one constant utterance. Reducing `cos(spk(clean), spk(clone))` under fixed
   content can only be achieved by exploiting that single sentence's acoustic realisation, which cannot
   generalise to the varied text a real attacker would synthesise.
2. `Ldec_clone` (gradient norm 10.71, by far the largest term) was computed on that same fixed clone and pulls
   directly against the protective term on it. The two terms were fighting over one example.
3. It depresses the measured similarity on its own: a fixed-text clone scored against varied real utterances
   carries a content and prosody mismatch that suppresses ECAPA similarity before speaker identity is reached.
   Part of the 0.29 floor in Defect 2 is this artefact.

The pilot was therefore **not** a 5-epoch continuation of R0 under R0's own protocol. It changed the clone-text
protocol and added the protective term in the same run — the two-things-at-once confound that invalidated `t1`.
Every run under this pre-registration from here on passes `--clone_text sampled` explicitly.

### Defect 2 — the gate variable was floored

ECAPA similarity of a watermarked clip's clone, real pipeline, RQA watermark, 100 clips,
against the calibrated ECAPA threshold of **0.2994**:

| cloner | ECAPA sim | headroom above threshold |
|---|---|---|
| CosyVoice 2 | 0.4675 | +0.168 |
| F5-TTS | 0.4551 | +0.156 |
| MaskGCT | 0.4159 | +0.117 |
| XTTS-v2 | 0.3399 | +0.041 |
| YourTTS (the in-loop cloner) | 0.3423 | +0.043 |
| **in-loop differentiable rollout, epoch 0** | **0.2927 (lam 2) / 0.2962 (lam 10)** | **-0.007 — already below** |

The protective term was asked to reduce a similarity already below the threshold. Every verification decision on
that quantity was a reject before training started, so no reduction in it can change an accept decision. The
0.05 bar was derived from Q2's measured relationship (0.37 of similarity buys 14-20 accept points), but that
relationship holds only where similarity crosses the threshold; at 0.29 the mapping from similarity to decision
is flat.

Noise compounds it. Per-step `SIM_clone` in the lam = 10 log spans 0.1377 to 0.5131, so sigma ~ 0.1; over ~880
steps per epoch the standard error of the epoch mean is ~0.0034. The observed 0.0075 fall is about 2 SE. The
harness could not have resolved a small real effect even if one existed.

**Ruled out while diagnosing, recorded so they are not revisited.** Sample rate (YourTTS's `waveform_decoder`
emits 16 kHz, matching the codec and the rate passed to `ecapa_embed`); clone duration (text pool is 6-14 words,
2-4 s, adequate for ECAPA); gradient flow (the A1 audit measured 3.37 across all 64 trainable tensors).
There is no coding error in the protective term. The error is in what was graded.

### The two changes

**Change 1: `--clone_text sampled` on every run,** matching R0. The arms then differ from R0 in exactly one
respect, `lambda_protect`.

**Change 2: the gate becomes the registered hypothesis, measured held out.** `mean_sim_clone` is retained as a
training diagnostic only and carries no decision rule. The pilot gate is now:

> On **F5-TTS**, a cloner never in the training loop and the one with the most headroom among the fast cloners:
> accept rate at Resemblyzer 0.687 falls by >= 0.05 relative to the matched lambda = 0 control, with exact
> McNemar p < 0.05, and clean-audio PESQ within 0.05 of the control's.

F5-TTS replaces MaskGCT as the *pilot* cloner purely on cost; MaskGCT remains decisive for HP1 in the full run.
Resemblyzer is the judge because it has never entered any loss here. ECAPA accept rate is reported beside it,
and a fall that appears on ECAPA but not Resemblyzer is reported as inflation, not as protection.

**The control, and why a fresh one is needed.** Both pilot checkpoints were lost when the Kaggle session
restarted, so nothing can be re-scored and a retrain is required regardless. R0 itself is not the control: it is
a 20-epoch checkpoint and these arms are 5-epoch continuations of it, so 5 epochs of lambda = 0 continuation
from R0 is the matched control.

| arm | msgproc rank | lambda_protect | role |
|---|---|---|---|
| `p1g_lam0` | 8 (R0's) | 0 | control |
| `p1g_lam2` | 8 (R0's) | 2.0 | the arm A1 rejected, now graded properly |

Everything else as the original pilots: 5 epochs, YourTTS in loop, same seeds — but **both** with
`--clone_text sampled`, so the control absorbs the protocol change and only `lambda_protect` differs between the
two arms.

**Decision rule, fixed now.**
- Gate met: A1's negative was a measurement artefact. Proceed to the full 20-epoch run and the matched-PESQ
  comparison as originally registered.
- Gate missed: lambda is confirmed dead *on a working instrument*, and **A2 resumes** — the rank-32 pair runs,
  graded by this same held-out gate, never by `mean_sim_clone`.
- Clean ACC below 0.90 in either arm: that arm is reported unusable and P1 stops.

**Predictions before the run.** P(`p1g_lam2` clears the gate) about 35 %, revised up from the 25 % I would have
given on Defect 2 alone: sampled text gives the protective term varied content to generalise over, which is the
condition under which a transferable identity effect is even possible. P(the gate moves at all, i.e.
|delta accept| > 0.02 in either direction) about 60 %. P(`mean_sim_clone` under sampled text starts materially
above the fixed-text 0.29, i.e. > 0.32) about 65 % — if it does, that confirms part of the floor was the
fixed-text artefact.

**What A1 and A2 are each worth, stated now so neither is overclaimed.** A1 reported that the protective effect
is not weight-limited. That reading is withdrawn too: the runs behind it used a clone-text protocol different
from the checkpoint they continued, and were graded on a floored in-loop quantity. **A1 and A2 together
establish nothing about whether a watermark can be made protective.** Only a held-out result under matched
protocol licenses any statement either way.

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
