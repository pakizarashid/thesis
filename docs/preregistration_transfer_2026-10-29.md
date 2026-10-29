# Pre-registration T1 / T2 — transfer boosters and verifier count

Written 2026-10-29, before any arm was generated. Notebook: `30_transfer_boosters.ipynb`.
Code: `src/eval/f5_whitebox_pgd.py` (momentum, input diversity, verifier set), `src/eval/protect_prompts_v2.py`
(`--momentum --div_prob --div_shift_ms --div_gain --verifiers --only_arm`), `src/eval/accept_rates.py`.

## Why

Q2 is optimised through one surrogate, F5-TTS, and its effect must carry to cloners it never saw. It does so
unevenly: against random noise at the same bound it wins on F5-TTS and CosyVoice but **not** on MaskGCT
(CI [−0.009, +0.011]). The design-space sweep is closed — six settings across three mechanisms showed the
trade-off is governed by loudness, not by where the perturbation sits in frequency
(`claude/hmask-result-2026-10-22.md`). The remaining lever is **transfer**, which is the question the
transferable-adversarial-example literature in vision was built around.

Two interventions, neither tested here before.

**T1 — transfer boosters.** Momentum (MI-FGSM: an L1-normalised gradient accumulated across the 40 steps, decay
1.0) plus input diversity (DI: with probability 0.7 each step, the candidate fed to the F5 rollout is shifted in
time by up to ±10 ms and scaled by a gain in [0.95, 1.05]). Momentum stops the search chasing per-step quirks of
one surrogate; diversity stops it overfitting one alignment. The watermark branch never sees the diversity
transform — the message must stay readable in the file an attacker downloads.

**T2 — verifier count.** The three in-loop verifiers were chosen by following prior work and never ablated.
E2E-VGuard uses six. Arms: 1 verifier (ECAPA), 3 verifiers (the registered set — this is the existing `q2` arm,
not regenerated), 5 verifiers (+ SpeechBrain ResNet, + WavLM-base-sv).

Resemblyzer is the held-out judge and may never enter any loss; `make_verifiers` asserts this.

## Design, fixed now

- Everything matches the registered Q2 except the named change: RQA watermark
  (`checkpoints/route2_vmaug_RQA/route2_epoch19.pt`), peak 0.015, envelope alpha 1, floor 0.003, 40 steps,
  step 0.0015, dis:wm 1.732, lambda_wm 1, F5 K = 6, CFG 2, seeds noise 5000+i / f5 7000+i, 100 clips,
  20 held-out speakers, unseen text, 3 s prompt.
- New arms: `t1` (momentum + DI, 3 verifiers), `v1` (1 verifier), `v5` (5 verifiers). The 3-verifier point of T2
  is the existing `q2` arm, reused, so it is not re-generated and cannot drift.
- Cloners: F5-TTS (the surrogate, in-domain), MaskGCT and CosyVoice 2 (transfer). MaskGCT is the decisive one.
- Scoring: `accept_rates.py`, at the Resemblyzer threshold **0.687**, frozen on 2026-10-26 and unchanged.
  Exact McNemar on paired clips, speaker-level bootstrap CIs (10,000 draws), Holm correction across the three
  cloners within each hypothesis.

## Hypotheses and decision rules

Effect bar: **5 percentage points** of accept rate. Chosen because Q2's own margin over same-bound random noise on
MaskGCT is about 3 points, so anything smaller cannot be told apart from the noise control.

**HT1 (primary) — transfer boosters help on a cloner the search never saw.**
PASSES if, on **MaskGCT**: accept(`t1`) ≤ accept(`q2`) − 0.05, one-sided exact McNemar p < 0.05 after Holm, AND
the quality gates hold: PESQ(`t1`) ≥ PESQ(`q2`) − 0.05 and traced(`t1`) ≥ traced(`q2`) − 0.05.
Otherwise NO GAIN SHOWN. A gain on F5-TTS alone is explicitly **not** a pass: F5-TTS is the surrogate, so an
in-domain gain is the self-graded result this thesis already argues against.

**HT2 — more verifiers help.**
PASSES if, on MaskGCT: accept(`v5`) ≤ accept(`q2`) − 0.05 with p < 0.05 after Holm and the same quality gates.

**HT3 (control, confirmatory) — one verifier is worse than three.**
Expected: accept(`v1`) ≥ accept(`q2`). If `v1` instead *beats* three verifiers on MaskGCT, the ensemble argument
used on slide 13 is wrong and must be retracted there, not defended.

**HT4 (descriptive) — in-loop versus held-out gap.**
Report the ECAPA accept rate (threshold 0.299) beside the Resemblyzer one for every arm. A booster that widens
that gap is improving the self-graded number only, and is reported as such.

## Predictions, before the run

1. P(HT1 passes) ≈ 35 %. Momentum and DI are reliable in vision, but the surrogate gap here is architectural
   (flow-matching F5 versus masked-token MaskGCT), not just a decision-boundary difference.
2. P(HT2 passes) ≈ 25 %. Four of the five verifiers are VoxCeleb-trained, so the ensemble is less diverse than
   its size suggests.
3. P(HT3 confirms) ≈ 80 %.
4. `t1` PESQ within ±0.05 of `q2`: ≈ 85 % (the bound is unchanged, only the direction of the steps).
5. If anything passes, the in-domain (F5-TTS) gain is larger than the transfer gain: ≈ 90 %.

## What this cannot show

- It cannot establish that momentum or DI help perturbation transfer in general; one surrogate, three cloners,
  one corpus, 100 clips, 20 speakers.
- It cannot separate momentum from input diversity: `t1` changes both at once. If `t1` passes, the separation is
  a follow-up arm, not a claim made from this run.
- `v5` is not E2E-VGuard's six-encoder ensemble and no comparison with that paper is made.
- A failure does not show the boosters are useless at a larger budget or step count; the budget is held at Q2's.

## Deliverables

- `results/anticlone/prompts/RQA/{t1,v1,v5}/` and their `records_*`
- `results/ce_ac_{f5tts,maskgct,cosyvoice}_RQA_{t1,v1,v5}.json`
- `results/accept_rates.json`, regenerated to include the new arms
- a result document stating each verdict against the rules above, failures included
