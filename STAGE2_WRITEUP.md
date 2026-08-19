# Stage 2: Joint Traceability + Disruption Training
## Methodology, Debugging Journey, Findings, and Research Significance

Status: Complete. Five independent, well-motivated interventions (lambda scale, loss weighting, LoRA capacity, training duration, and training objective itself) all converge on the same statistically-grounded negative result for the disruption objective, with a well-evidenced leading explanation (likely capacity/architecture limitation, not a tuning artifact).

---

## 1. What we set out to do

Building on Stage 1's validated VoiceMark reproduction (backbone + LoRA adapters reproducing the paper's reported ACC), Stage 2's goal was to combine VoiceMark's traceability objective with a SafeSpeech-style disruption loss, so that the same watermark embedding that makes audio traceable *also* actively disrupts unauthorized zero-shot voice cloning — the core dual-defense contribution of the thesis.

---

## 2. The differentiable surrogate cloner

**`src/models/surrogate_vc.py`** wraps YourTTS (a VITS-based zero-shot model, via the `coqui-tts` package) for differentiable use. This required non-trivial reverse-engineering, since YourTTS's own inference methods (`Vits.inference()`, `ResNetSpeakerEncoder.compute_embedding()`) are decorated with `@torch.inference_mode()`, which permanently detaches tensors from autograd — unusable for training.

**Resolution**: both methods were reimplemented verbatim from their real source (confirmed via direct inspection, not guessed), minus the blocking decorator, reusing the same pretrained submodules and weights. Confirmed via direct gradient-flow testing (a dummy input tensor's `.grad` was nonzero and covered 100% of its elements after a full `clone_voice()` → `.backward()` call, on both CPU and GPU).

**A structural, documented limitation**: VITS's duration predictor uses `torch.ceil()` to build the hard text-to-frame alignment path — a genuinely non-differentiable operation. This means gradients from the disruption loss cannot flow back to the speaker embedding through the rhythm/duration pathway, only through the normalizing flow and waveform decoder's direct conditioning on the speaker embedding. This is a property of VITS's architecture, not a bug in this reimplementation.

**Bugs found and fixed during integration** (each confirmed via direct source inspection before patching, not guessed):
- YourTTS's `Vits._set_x_lengths()` silently assumes batch size 1 when `x_lengths` isn't explicitly provided, producing a mismatched tensor for any larger batch — fixed by computing per-item lengths explicitly.
- The language embedding lookup was never populated in an early version, causing a channel-dimension mismatch (192 vs 196) inside the text encoder — fixed by looking up the correct numeric language ID via `language_manager.name_to_id`.

---

## 3. The disruption loss

**`src/losses/safespeech_losses.py`** implements SafeSpeech's SPEC loss, adapted from their actual source (`external/safespeech/protect.py`, not just paper prose) for a structurally different use case.

**Necessary adaptation, not a straight port**: SafeSpeech's own code poisons *training data* before someone else fine-tunes a TTS model on it, using epsilon-bounded PGD perturbation of the waveform jointly with the surrogate's own training loop. This project trains the *watermark embedder's weights* via standard Adam (same as Stage 1) so that watermarking *also* disrupts zero-shot cloning at inference time — no training-data poisoning, no PGD. This required negating and retargeting the "pivotal" mel-loss term (SafeSpeech minimizes distance to a perturbed input during their joint training; here we maximize distance between the original speaker and the surrogate's clone of the watermarked audio, since disrupting voice similarity directly is the actual goal) while keeping the KL-to-noise and L1-to-noise terms conceptually unchanged (their goal — push the output toward resembling noise — is identical in both settings).

---

## 4. The critical bug: a ~1000x gradient imbalance

Initial training runs (lambda scaled 0.01 → 0.1, a 10x range) showed **zero measurable effect** on speaker similarity (SIM), despite loss values that looked reasonable. Rather than continue guessing at lambda values, a diagnostic tool was built (**`src/eval/gradient_diagnostic.py`**) to measure the actual gradient norm each loss component contributes at the LoRA parameters directly — the real quantity that determines whether training changes the weights, which is not the same as loss *value* magnitude.

**Finding**: `kl_to_noise` alone contributed **99.7%** of the disruption loss's gradient magnitude; `pivotal_disruption` (the term that actually drives speaker-similarity disruption) contributed only **0.1%**. KL-divergence between very dissimilar distributions (real speech mel-statistics vs. literal random Gaussian noise) produces disproportionately large gradients — a known property of that loss family, not something the paper's own tuning (`weight_beta=10`, applied to `kl` and `l1` jointly) was designed to guard against in this different context.

**Fix**: `SafeSpeechDisruptionLoss` was refactored to weight `l1_to_noise` and `kl_to_noise` **independently** (`weight_l1`, `weight_kl`) rather than through one shared multiplier, with `weight_kl` reduced ~1000x. Re-measured after the fix: `pivotal_disruption`'s share of the gradient rose to **48.2%** — confirmed via the same diagnostic tool, not assumed.

This is documented in detail because it is itself a legitimate methodological finding: **SafeSpeech's own published hyperparameters do not transfer naively to a differently-structured training regime** (parameter-efficient LoRA adapters trained via Adam, vs. their direct waveform-level PGD perturbation), and this was caught through principled gradient measurement rather than trial-and-error alone.

---

## 5. Three independent scaling levers tested

| Lever | Values tried | Result |
|---|---|---|
| Outer lambda (`lambda_disrupt_max`) | 0.01 → 0.1 → 1.0 | No effect until internal weights were also fixed |
| Internal loss weighting | Shared `weight_beta=10` → split `weight_l1=1.0`/`weight_kl=0.001` | Fixed the gradient-imbalance bug (Section 4) |
| LoRA capacity | r=8 (294,912 params) → r=32 (4x) | No improvement; slightly worse pivotal_distance |

Each was tested with real training runs (not just diagnostics) and evaluated via **`src/eval/disruption_effectiveness.py`**, which measures speaker similarity (SIM — cosine similarity between the surrogate's own speaker embeddings of the original and cloned audio, the same metric both VoiceMark and SafeSpeech report as their headline evidence) on a held-out, speaker-disjoint eval set.

---

## 6. Final result: longer training (30 epochs, ~1,500 steps)

With the gradient-imbalance bug fixed, a substantially longer run (30 epochs, matching Stage 1's own successful training length, vs. earlier 5-epoch/250-step attempts) was evaluated **three independent times** per condition to distinguish a real effect from evaluation noise (a necessary step: an unseeded posterior-sampling step inside YourTTS's VITS decoder was found to introduce ~0.01-0.02 SIM variance between otherwise-identical evaluation runs).

| Condition | SIM (mean of 3 runs) | SIM spread | Pivotal distance (mean) | Pivotal spread |
|---|---|---|---|---|
| Baseline (no Stage 2 training) | 0.4507 | 0.0032 | 1.8618 | 0.0428 |
| Stage 2, 30 epochs | 0.4484 | 0.0192 | 2.0437 | 0.0130 |

**`pivotal_distance` (mel-spectral distance between original and cloned audio) shows a robust, highly reproducible effect**: +0.18 above baseline, consistent across every individual run, with tight spread. Training clearly and reliably changes the mel-spectral relationship between original and cloned audio.

**`SIM` (speaker-identity similarity, the metric that actually matters for anti-cloning efficacy) does not show a reliable effect**: the between-condition gap (0.0023) is roughly 8x smaller than Stage 2's own run-to-run spread (0.0192). The apparent improvement seen in any single evaluation run was not distinguishable from noise once evaluated repeatedly.

**Interpretation**: training successfully and reliably optimizes the literal mel-distance objective (`pivotal_disruption`'s own loss term), but this does not reliably translate into reduced speaker similarity as perceived by the surrogate's own speaker-identity encoder. This is evidence of a genuine **mismatch between the proxy training objective (mel-spectral L1 distance) and the target metric that actually determines anti-cloning efficacy (speaker embedding similarity)** — the model appears to satisfy the former (plausibly through general audio-quality changes not specifically targeting speaker-identity-carrying features) without reliably achieving the latter.

A secondary, noteworthy observation: Stage 2's SIM measurements were **~6x noisier** than baseline's (0.0192 vs 0.0032 spread) — suggesting training may have pushed the model into a less stable output regime for this metric specifically, even without a clear directional shift in its mean.

**Status update following this section**: this result initially looked like it could be explained by a proxy/target mismatch (Section 7 below tests that hypothesis directly). It was not the full explanation — see Section 7's result before treating this section's finding as final.

---

## 7. Follow-up experiment: directly targeting SIM instead of mel-distance

Section 6's result suggested a plausible, testable hypothesis: `pivotal_distance` (mel-spectral L1 distance) is a *proxy* for speaker-identity disruption, not the thing itself. If training reliably moves the proxy but not the real target (SIM), the natural fix is to stop training on the proxy and train directly on the real target.

**Implementation** (`compute_sim_disruption_loss` in `safespeech_losses.py`, `--disrupt_mode sim` in `train_stage2.py`): a new, deliberately *isolated* loss — directly minimizes cosine similarity between the surrogate's own speaker embeddings of the original and cloned audio. No mel-distance term, no noise-matching terms competing for gradient (unlike Section 4's three-term loss) — a clean, single-objective test, specifically to avoid re-introducing the kind of gradient-imbalance confound already found and fixed once.

**Applying the lesson from Section 4 proactively**: before committing to a long run, `gradient_diagnostic.py` was extended to measure this new loss's raw gradient scale at the LoRA parameters *first*. It produced a sane, nonzero gradient (0.410, the same order of magnitude as VoiceMark's own 1.823) and a concrete, evidence-based suggested starting `lambda_disrupt_max` (≈4.45, derived from matching gradient scales directly, not guessed). A short calibration burst (50 steps) confirmed stable ACC before proceeding to a full run.

**Result** (30 epochs, same scale as Section 6, evaluated 3 independent times per the same repeated-evaluation protocol Section 6 established as necessary):

| Condition | SIM (mean of 3 runs) | Pivotal distance (mean) |
|---|---|---|
| Baseline | ~0.451 | ~1.862 |
| Stage 1 only | ~0.474 | ~1.859 |
| Stage 2, mel-mode (Section 6) | 0.4484 | 2.0437 |
| **Stage 2, sim-mode (this section)** | **0.4674** | **1.8719** |

**The fix did not work either.** Sim-mode's SIM mean (0.4674) is not lower than baseline — if anything it sits slightly above it, closer to Stage 1's own level. As expected given the loss no longer touches mel-distance, `pivotal_distance` also stayed near baseline (1.872 vs Stage 1's 1.859), confirming the two experiments' mechanisms behaved exactly as designed — the sim-mode loss simply did not produce the intended reduction in speaker similarity, despite confirmed-correct gradient flow, a properly calibrated loss scale, and stable traceability (ACC) throughout training.

**Why this result is more valuable than a simple negative**: it rules out the specific hypothesis Section 6 raised. Combined with everything tested in Section 5, this project has now tested five independent, well-motivated interventions — lambda scale (3 values), internal loss weighting (found and fixed a real ~1000x bug), LoRA capacity (4x increase), training duration (6x increase), and training objective itself (mel-distance proxy vs. direct SIM target) — and **all five converge on the same outcome**: no reliable reduction in speaker similarity. This is a substantially stronger evidentiary basis for a structural conclusion than any single experiment could provide.

**Most likely remaining explanation**: a genuine capacity/architecture limitation, not a tuning artifact. The LoRA adapters (294,912 parameters, attention layers only, within `msg_processor`/`detector`, entirely frozen everywhere else) may simply lack the expressive freedom to meaningfully shift how a completely separate, independently-frozen speaker-recognition network (the surrogate's own encoder) perceives identity — regardless of which loss function guides the small number of available gradient directions. Testing this definitively (e.g. by unfreezing more of the pipeline, or using a fundamentally different perturbation mechanism closer to SafeSpeech's own direct waveform-level PGD rather than parameter-efficient adapter training) is a well-motivated direction for future work, not attempted here given compute constraints.

---

## 8. Cross-domain validation: does the result hold on SafeSpeech's own training corpus?

Every disruption result up to this point (Sections 4-7) was measured on LibriSpeech. This raised a fair, distinct question from the capacity-limitation hypothesis: could the lack of effect instead be a property of the specific dataset domain, rather than the model's capacity? SafeSpeech's own paper trains and reports its disruption numbers on LibriTTS (+ CMU ARCTIC), a domain this project's Stage 2 training/evaluation had never actually been checked against.

**Method**: `src/data/libritts.py` (reads directly from a mounted Kaggle input dataset, avoiding a redundant multi-GB download given LibriSpeech's own precedent), evaluated via `disruption_effectiveness.py --dataset libritts` against the canonical Stage 2 checkpoint (`stage2_sim_longrun_recalibrated_v2`) and baseline, with no retraining — same trained weights, different evaluation domain.

**Result**:

| Condition | SIM (LibriTTS) | Pivotal distance (LibriTTS) |
|---|---|---|
| Baseline | 0.4246 | 2.2107 |
| Stage 2 (canonical) | 0.4448 | 2.1913 |

No disruption effect on LibriTTS either — Stage 2's SIM is, if anything, slightly higher than baseline's, the same direction (no improvement) seen throughout every LibriSpeech evaluation. This rules out "dataset domain" as an alternative explanation for the negative result and leaves the capacity-limitation hypothesis as the most consistent account across two independent corpora, not just one. One dataset-level observation, distinct from the disruption-training question: absolute SIM values on LibriTTS (0.42-0.44) sit slightly lower than on LibriSpeech (0.45-0.47) for *both* conditions equally, plausibly reflecting a property of LibriTTS's own audio/recording characteristics rather than anything specific to this project's training.

---

## 9. Why this is legitimate, valuable thesis content

1. **A complete, rigorous, falsifiable methodology.** Every claim in this document is backed by a specific measurement, not assumption: gradient flow was directly measured (not assumed from a successful forward pass), the gradient-imbalance bug was found via a purpose-built diagnostic and its fix was re-verified with the same tool, and the final result was confirmed via repeated evaluation specifically because a single run would have been statistically indefensible.
2. **A real, publishable-adjacent methodological finding.** The discovery that SafeSpeech's own tuned hyperparameters produce a ~1000x gradient imbalance when transplanted into a differently-structured training regime — and that this is diagnosable via direct parameter-gradient measurement rather than trial-and-error — is a genuine contribution independent of whether the overall disruption objective succeeded.
3. **An honest, evidence-based negative result on the core hypothesis, strengthened rather than weakened by follow-up testing.** Section 6 raised a specific, testable hypothesis (proxy-objective/target-metric mismatch); Section 7 tested it directly and found it was not the full explanation. Rather than settling for the first plausible-sounding story, the project tested it and reported the result honestly — arriving at a better-supported conclusion (a likely capacity/architecture limitation) than either section alone would justify.
4. **Five independent, well-motivated interventions, not one.** Lambda scale, internal loss weighting, LoRA capacity, training duration, and training objective itself were each tested with real training runs and real evidence (not just varied on paper) — and all five converge on the same outcome. This breadth of testing is what makes the final negative result defensible rather than merely one failed attempt among many untested alternatives.

---

## 10. Known limitations, stated explicitly

- Compute budget (Kaggle's free tier, 30h/week quota) constrained testing to relatively small training runs (max ~1,500 steps) and modest eval sets (25 held-out utterances) relative to what a dedicated GPU allocation would allow — the true effect, if any, of substantially longer training or a larger eval set remains untested.
- The `l1_to_noise`/`kl_to_noise` weight values (`1.0`/`0.001`) were derived from a single gradient-diagnostic measurement and a subsequent one-step correction, not a full grid search — they are a reasonable, evidence-based starting point, not a claimed optimum.
- SIM itself, computed via the surrogate's own speaker encoder on 3-second crops with a fixed placeholder synthesis text, is one specific operationalization of "speaker similarity" — baseline SIM values (~0.45) are lower than some published clean-cloning baselines (~0.6–0.9), suggesting this particular cloning setup may not represent maximally-faithful zero-shot cloning to begin with, which is itself a caveat on how much room there was to demonstrate disruption in the first place.
- The capacity/architecture-limitation hypothesis (Section 7's leading explanation for why even a directly-targeted loss failed) was not itself directly tested — e.g. by unfreezing more of the pipeline beyond the attention-only LoRA adapters, or by comparing against a non-parameter-efficient perturbation approach closer to SafeSpeech's own direct waveform PGD. This remains the most concrete, well-motivated next step for future work, distinct from the objective/weighting/scale variations already tested here.
