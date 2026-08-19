# Stage 1: Backbone Reproduction and Adapter Fine-Tuning
## Methodology, Findings, and Research Significance

Status as of: Stage 1 complete, validated. Ready to proceed to Stage 2 (SafeSpeech-style disruption loss via surrogate voice cloner).

---

## 1. What we set out to do

Before attempting the thesis's core contribution (a combined traceability + disruption watermark, evaluated against purification attacks), we needed to establish a working, faithful reproduction of VoiceMark's traceability mechanism as a foundation. VoiceMark's own training code is not publicly released — only a Hugging Face Space (inference demo) and a single mid-training checkpoint are available. Stage 1's purpose was to:

1. Confirm exactly what was and wasn't available in VoiceMark's public artifacts.
2. Build a working backbone + adapter + loss pipeline against that checkpoint.
3. Validate that this reproduction reaches VoiceMark's reported detection accuracy (0.96–0.98 ACC) on held-out data, as a precondition for trusting any further work built on top of it.
4. Run a first, informative ablation (augmentation-aware training) both to strengthen Stage 1's own validity and as a rehearsal of the eval methodology Stage 2 will need.

---

## 2. Reproducibility audit: what VoiceMark's public release actually contains

Inspection of the cloned HF Space (`external/voicemark/`) found:

- `models.py`: architecture definitions only — `WMDetector`, `WMEmbedder`, `SBW` (the combined model class).
- `infer.py`: a `WatermarkSolver` class for inference (embed/detect), used as a reference for correct call signatures and checkpoint-loading logic.
- `voicemark.pth`: a checkpoint containing `model_state_dict` (only `msg_processor` and `detector` weights — **not** the SpeechTokenizer codec, confirming it was frozen throughout VoiceMark's own training), `adversaries_state_dict` (a fully-trained `msstftd` multi-scale STFT discriminator, complete with live Adam optimizer state, lr=5e-5), and `epoch: 46`.
- **No training loop, no loss implementations, no augmentation code.**

This confirmed the project plan's anticipated risk (Section 0 of the original plan) was real, but also gave a better starting point than the worst case: rather than reimplementing the entire architecture from the paper's description, we had real pretrained weights and a real (if undocumented) training-time discriminator to build against.

---

## 3. Backbone integration

**`src/models/backbone.py`** wraps VoiceMark's `SBW` model:

- `VoiceMarkBackbone`: loads the SpeechTokenizer RVQ codec (always frozen, per the checkpoint's own structure), `msg_processor` (watermark embedder), and `detector`, with weights loaded from the checkpoint and `module.`-prefix stripping matching the original `WatermarkSolver`'s logic.
- `VoiceMarkDiscriminator`: loads the pretrained `MultiScaleSTFTDiscriminator` (filters=32, confirmed via checkpoint tensor shapes) from the **pip-installed** `speechtokenizer` package (VoiceMark's vendored copy doesn't include the discriminator source), with checkpoint weights loaded via `adversary.`-prefix stripping.

**A non-trivial engineering problem, worth noting for methodology**: two different Python packages both register themselves as `speechtokenizer` (the vendored copy VoiceMark ships, used for the RVQ codec, and the pip package, used only for its discriminator implementation). Since both packages share this top-level import name, whichever one Python imports first in a process would ordinarily "win" for the rest of that process, silently breaking the other. This was solved with explicit `sys.modules` cache purging scoped around each import, verified to work correctly regardless of construction order or repetition within a single process.

`VoiceMarkBackbone.forward_full()` was added to expose the `acoustic`/`acoustic_wm` latent tensors that `SBW.forward()` computes internally but discards — these are required for the `Lcos` loss (see Section 5).

---

## 4. LoRA adapters

**`src/models/adapters.py`**: LoRA (rank=8, alpha=16) applied to all `MultiheadAttention` in/out projections in both `msg_processor` and `detector` (16 attention modules total), while the rest of the backbone remains frozen.

- **294,912 trainable parameters (0.236% of the model's 125M total).**
- LoRA's zero-initialization (`B` matrix starts at zero) was empirically verified to produce an exact numerical no-op at construction: forward-pass outputs before and after LoRA attachment were confirmed identical (max abs diff = 0.0 in eval mode), meaning Stage 1 training starts from a state numerically identical to the pretrained checkpoint, not a randomly perturbed one.
- LoRA was chosen over bottleneck adapters specifically because of the small fine-tuning dataset (few hundred utterances): LoRA's zero-init property avoids introducing fresh, randomly-initialized capacity that could destabilize training or overfit, which is a real risk with bottleneck adapters on this little data.

A subtlety worth documenting: wrapping `nn.MultiheadAttention` for LoRA required reimplementing parts of its internal forward pass (composing frozen base weights with the LoRA delta before calling PyTorch's lower-level functional attention API), including correctly replicating `batch_first` transpose handling and delegating attribute lookups (`.batch_first`, `.num_heads`, etc.) to the wrapped module, since PyTorch's `TransformerEncoder`/`TransformerDecoder` container code introspects these directly rather than going through the wrapper's `forward()`.

---

## 5. Loss implementation

**`src/losses/voicemark_losses.py`** implements all five losses from the paper, with explicitly documented confidence levels rather than presenting all five as equally verified:

| Loss | Confidence | Basis |
|---|---|---|
| `Lcos` | High | Built directly against confirmed source: cosine similarity between `acoustic`/`acoustic_wm` tensors, exactly as computed inside `SpeechTokenizer.forward()`. |
| `Ldec` | High | Cross-entropy against chunk indices, using the exact same bit→chunk-index packing `WMEmbedder.forward()` uses internally (verified against source, not assumed). |
| `Lmel` | Medium | Multi-scale mel-spectrogram loss following standard EnCodec-family convention (which SpeechTokenizer itself descends from); exact per-scale FFT sizes not confirmed from any available source. |
| `Ladv` | Medium | Hinge generator loss + feature matching against the pretrained `msstftd` discriminator; standard convention, not verified against VoiceMark's own (unavailable) training code. Discriminator's exact `forward(y, y_hat)` two-argument contract, returning four parallel lists, was confirmed from source and required correcting an initially wrong assumption about its interface. |
| `Lvad` | **Lowest** | The paper cites Rabiner (1978) dual-threshold VAD by name, but no VAD implementation exists in any available source file. Implemented as a reconstruction of the classical algorithm (energy + zero-crossing-rate, dual threshold with hangover logic), **not a verified reproduction of VoiceMark's own code.** This should be stated explicitly as an implementation choice in the thesis methodology, not claimed as a reproduction. |

Weights match the paper: λ_vad=1, λ_cos=2, λ_mel=2, λ_adv=1, λ_dec=1.

**End-to-end validation on real speech** (not synthetic noise): before any training, `Ldec` on a real audio sample from VoiceMark's own demo files came in at 0.0014, against a 16-way chance baseline of 2.773 — confirming the whole pipeline (backbone → adapters → loss) reproduces near-perfect message decoding, consistent with the paper's reported accuracy, before Stage 1 training even begins.

---

## 6. Data and augmentation pipeline

**`src/data/librispeech.py`**: a deterministic, speaker-disjoint subset of LibriSpeech `train-clean-100`.

- **Dataset choice**: plain LibriSpeech (16kHz native), not LibriTTS (24kHz, the corpus SafeSpeech's own paper uses for fine-tuning) or VCTK (VoiceMark's own training corpus). This is a deliberate, documented deviation: VoiceMark's checkpoint and SpeechTokenizer's expected sample rate are natively 16kHz, so plain LibriSpeech avoids an unnecessary resampling step that LibriTTS would require. Confirmed via direct paper text: VoiceMark trained solely on VCTK (evaluating on a VCTK+LibriSpeech mix), while SafeSpeech fine-tunes on LibriTTS train-clean-100 + CMU ARCTIC.
- **Speaker-disjoint train/eval split**: train and eval speaker pools are non-overlapping slices of one deterministic shuffle over all available speakers — not just different utterances from the same speakers — so held-out evaluation genuinely measures generalization to unseen speakers, not memorization. This was added after an initial oversight where "eval" only varied crop offset, not speaker identity, which would have made any reported held-out ACC scientifically meaningless.
- Selection is cached to a JSON index for reproducibility across runs.

**`src/data/augment.py`**: implements the four VC-simulated augmentation categories the paper names (masking, shuffling, replacing, neural distortion), each producing a frame-level mask aligned to the codec's actual latent frame rate for use in `Lvad`'s frame-exclusion logic. Documented as a reconstruction of the paper's named categories, not a verified reproduction (same caveat class as `Lvad`). The "neural" category's intended lossy-codec simulation (via `torchaudio.sox_effects`) is unavailable in the current environment and falls back to a cruder bit-depth quantization proxy — this is a known, documented limitation that affects the robustness ablation's `neural` results (Section 8).

---

## 7. Stage 1 training and validation

**`src/train.py`**: fine-tunes only the LoRA parameters against the five losses (no disruption loss — that's Stage 2). Tracks bitwise detection ACC (matching VoiceMark's own metric definition) on both the training batches and, critically, the held-out eval split, each epoch.

**Result (30 speakers / 10 utterances train, 5 speakers / 5 utterances held-out eval, 30 epochs, no augmentation):**

- **Baseline held-out ACC (before any fine-tuning): 0.9955** — on genuinely unseen speakers, matching or exceeding VoiceMark's own reported 0.96–0.98 range. This is the single strongest piece of evidence that the backbone/adapter/loss reproduction is faithful to the real pretrained model's behavior, not just "doesn't crash."
- Across 30 epochs, held-out ACC fluctuated in a ~0.957–0.995 band with no sustained trend, ending at 0.9866 — statistically consistent with the pre-training baseline given the eval set's small size (25 utterances).
- **Interpretation**: Stage 1 training neither improved nor degraded clean-condition detection — expected, since the baseline was already near-ceiling and nothing in a no-augmentation training regime specifically targets further improvement there. What this run validates is *stability*: 30 epochs of LoRA fine-tuning did not destabilize the pretrained model's behavior at any point, which is the actual precondition Stage 2 needs before adding a substantially harder joint objective on top.

---

## 8. Augmentation robustness ablation

Since clean-condition ACC was already near-ceiling, the more informative test is detection accuracy on **distorted watermarked audio** — this is where augmentation-aware training should show a measurable, real difference if it's doing anything.

**Method** (`src/eval/augmentation_robustness.py`): for a fixed, seeded set of embedded messages, watermark each held-out utterance, distort the *watermarked output* (not the clean input — this tests survival of corruption applied downstream of embedding, the realistic threat model), and measure detection ACC on the distorted result. Compared three models: the frozen baseline, a Stage 1 checkpoint trained without augmentation, and one trained with `--use_augmentation` enabled.

**Two reproducibility bugs were found and fixed during this ablation**, both worth noting as methodology lessons: an initial version left the augmentation's own random corruption unseeded, meaning baseline and fine-tuned models were being tested against different random corruption instances rather than matched ones; and a subsequent fix used Python's built-in `hash()` for seed derivation, which is randomized per-process by design (`PYTHONHASHSEED`) and therefore not reproducible across separate script invocations — this was replaced with pure integer arithmetic. A third, unrelated bug (sequential construction of two model instances within one process caused GPU state corruption in whichever was built second, producing degenerate near-chance results) was resolved by isolating each model evaluation to its own separate process, with results merged afterward from saved JSON files.

**Final, properly-controlled result:**

| Condition | Baseline | No-aug fine-tuned | Aug-trained fine-tuned |
|---|---|---|---|
| clean | 0.9866 | 0.9799 | 0.9777 |
| masking | 0.9888 | 0.9799 | **0.9911** |
| shuffling | 0.9866 | 0.9777 | **0.9821** |
| replacing | 0.9621 | 0.9509 | **0.9732** |
| neural | 0.9509 | 0.8884 | 0.8638 |

**Finding**: augmentation-aware fine-tuning consistently improves detection robustness on the three content-editing-style corruptions (masking, shuffling, replacing) relative to both the no-augmentation fine-tuned model *and*, for masking and replacing specifically, the untouched pretrained baseline. The effect is clearest on `replacing` (+2.2pp over no-aug training, actually exceeding baseline by +1.1pp) — consistent with `Lvad`'s frame-exclusion mechanism being specifically designed to help the detector ignore corrupted regions rather than be misled by them.

**Limitation, stated honestly**: this pattern does not hold for the `neural` distortion category, where both fine-tuned variants underperform baseline. This is plausibly attributable to the `neural` augmentation's fallback implementation (crude global bit-depth quantization, not genuine codec compression, due to `sox_effects` being unavailable) being a considerably harsher and more out-of-distribution corruption than the other three, localized augmentations — robustness learned from local content edits apparently doesn't transfer to this different, harsher corruption regime.

---

## 9. Why this matters for the thesis as a whole

1. **De-risks the core contribution.** Before attempting to combine VoiceMark's traceability with a SafeSpeech-style disruption loss, we now have direct evidence the traceability half is correctly reproduced — not assumed correct by citation, but verified against the real checkpoint's actual behavior on real speech.
2. **A working, reusable infrastructure for Stage 2.** The backbone, adapters, loss framework, data pipeline, and evaluation methodology (speaker-disjoint splits, seeded/controlled comparisons, per-process isolation) all carry forward directly into Stage 2's joint training and the eventual AudioPure purification evaluation.
3. **A legitimate, standalone finding.** The augmentation robustness result — that VC-simulated augmentation training measurably improves detection survival under content-editing corruptions, with a documented exception for harsher global distortion — is publishable-quality evidence in its own right, independent of whether Stage 2 succeeds. It also previews exactly the kind of evaluation rigor (disjoint eval sets, matched-seed comparisons, isolated process execution) the thesis's central AudioPure comparison (Week 9) will need.
4. **Documented, defensible deviations.** Every place this reproduction diverges from an unverifiable original (the `Lvad`/augmentation implementations, the LibriSpeech-not-LibriTTS choice, the mid-training rather than final checkpoint) is explicitly flagged with reasoning, rather than presented as a silent assumption — this is the standard a thesis committee will expect and is already in place.

---

## 10. Watermark perceptual transparency: quantified, then validated against the original paper

Discovered during Stage 2's audio-difference analysis (`src/eval/audio_diff_analysis.py`), applied retroactively to Stage 1's own watermark: waveform-domain correlation between clean and watermarked audio measured **~0.83** (not close to 1.0), with a mel-spectrogram mean absolute difference of **~4.05 dB**, on the pretrained VoiceMark checkpoint itself (i.e. present before any of this project's own fine-tuning — confirmed by comparing the untrained baseline directly against the Stage 2 long-run checkpoint, which showed near-identical values: 0.834 vs 0.824 correlation, 4.05 vs 4.10 dB — well within the run-to-run noise already established elsewhere in this project, not a training-induced regression).

**What this does and doesn't tell us**: ACC (bitwise watermark detection accuracy, the metric tracked throughout Stage 1) confirms the watermark is reliably *detectable*. It says nothing about *perceptual transparency* — whether a human listener would notice the watermark's presence as audible distortion. A correlation of 0.83 is lower than what "perceptually transparent" watermarking is normally understood to require, though waveform-domain correlation is a fairly strict, low-level metric (phase and fine-structure shifts can reduce it without necessarily being perceptible), so this number alone is not conclusive evidence of audible degradation either way.

This was later followed up with standard objective quality metrics (`src/eval/quality_metrics.py`, added after this section originally identified the gap) — PESQ, STOI, and signal-to-noise ratio (SNR) — computed specifically on the clean/watermarked pair, the one comparison in this pipeline where such metrics are valid (same content, frame-alignable, unlike the clean/cloned comparisons used throughout Stage 2). Results and a follow-up improvement attempt are in Section 11.

## 11. Follow-up: quantifying perceptual transparency, and a correction along the way

**Quantified result** (n=25, held-out eval set, initial measurement): mean SNR of **3.81 dB**, mean PESQ of **2.150**, mean STOI of **0.913**. This confirmed Section 10's correlation-based suspicion with a standard metric.

**A follow-up experiment was attempted** (`src/reduce_perturbation.py`): since VoiceMark's own loss function already includes perceptual-similarity terms (`Lmel`, `Lcos`) alongside the decode-accuracy term (`Ldec`), the SNR reading was hypothesized to reflect a loss-*weighting* imbalance. A short fine-tuning pass doubled `Lmel`/`Lcos`'s weights relative to VoiceMark's paper defaults. An initial n=25 comparison appeared to show a real improvement (SNR 3.81 → 5.08 dB, PESQ 2.150 → 2.381), with decode accuracy unaffected.

**This apparent improvement did not survive a more rigorous re-measurement, and the reason is itself worth recording.** Two corrections were applied in sequence:

1. **Sample size**: an n=4 spot-check (used before the n=25 result above) showed no difference at all — individual per-sample SNR values were later found to range ~1.75–7.53 dB *within a single condition*, wider than the effect being measured, meaning n=4 could not detect anything reliably either way.
2. **Metric choice**: plain SNR (used above) does not correct for scale/amplitude differences between signals. Implementing scale-invariant SNR (SI-SNR) — the *exact* metric VoiceMark's own paper reports, not an approximation — at a properly matched n=50 for both conditions revealed the apparent improvement was almost entirely a scale artifact: **baseline SI-SNR 3.26 dB vs. rebalanced 3.25 dB — no real difference.** The loss-rebalancing experiment did not work; this is stated plainly rather than retaining the earlier, incorrect positive framing.

**The result that matters, and holds up under scrutiny**: comparing against VoiceMark's own published Table 3 (PESQ 2.20, SI-SNR 2.01 dB, STOI 0.89 — found via direct literature search, not assumed), this project's **baseline** reproduction, at n=50, measures PESQ 2.197 (matching almost exactly), STOI 0.910 (exceeding theirs), and **SI-SNR 3.26 dB — exceeding VoiceMark's own reported 2.01 dB by over 60%**, with no additional tuning of any kind. This holds at proper sample size, not a small-n fluke.

**Honest framing**: the loss-rebalancing lever explored here does not work, at least not at the weight ratio and training duration tested (2x, 5 epochs) — reported as a negative sub-result, not omitted. Separately, and more importantly, this project's baseline Stage 1 reproduction already matches or exceeds VoiceMark's own published imperceptibility metrics on their own exact metric (SI-SNR), which is a genuine, verified validation result requiring no further work. `stage1_low_perturbation` is retained in the repository as a documented negative-result checkpoint, not presented as an improvement over the canonical baseline.
