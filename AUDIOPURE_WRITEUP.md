# AudioPure Purification Attack Evaluation
## Methodology, Findings, and Research Significance

Status: Complete. The thesis's clearest, most statistically conclusive result.

---

## 1. What we set out to do

Stage 1 validated that VoiceMark's watermark reliably survives detection under normal conditions and under a set of VC-simulated distortions (masking, shuffling, content replacement, quantization noise). Stage 2 investigated whether the same watermark could be trained to also disrupt zero-shot voice cloning. Neither stage, and neither of the source papers (VoiceMark or SafeSpeech) independently, tested the watermark against **AudioPure** — a diffusion-based adversarial purification defense (Wu et al., ICLR 2023) that noises audio partway through a diffusion process and then denoises it, a mechanism fundamentally different from the corruption types Stage 1's augmentation training targeted. This is the central, previously-untested question the whole project was designed to reach.

---

## 2. Locating and integrating AudioPure

**Official implementation**: `github.com/cychomatica/AudioPure`, vendored as a git submodule (`external/audiopure`). The repository's own README states only "More details to be updated..." — required the same source-first reverse-engineering approach used throughout this project, not documentation-reading.

**The purification mechanism** (`diffusion_models/diffwave_ddpm.py`, the `DiffWave` class): a complete, correct, waveform-native implementation — noise the input to `reverse_timestep` steps of a 200-step diffusion schedule, then iteratively denoise back down. A second candidate implementation in the same repo (`ImprovedDiffusion`, operating on mel-spectrograms) was found to contain **actual dead code** (`_reverse()` calls `p_sample_loop()` but never returns its result, meaning `forward()` would crash on `None` if ever executed) — confirmed via direct source inspection, not assumed, and excluded in favor of the working `DiffWave` path.

**Checkpoint discovery**: no pretrained weights were bundled in the main repository or attached as a GitHub Release (confirmed via the Releases API, which returned empty). The actual checkpoint was found referenced in a nested `DiffWave_Unconditional/README.md` (a file not visible from the repository's top level), pointing to a separate repository (`philsyn/DiffWave-unconditional`) hosting the trained weights directly in its file tree. Corroborated by an independent third-party paper explicitly citing use of "the official checkpoint" from this same source.

**Compatibility fixes required** (documented in `scripts/patch_audiopure.py`, applied idempotently every session since these are edits to a git submodule's files, which a fresh `git submodule update` always reverts to upstream):
- `dataset.py` imports `download_url`/`extract_archive` from `torchaudio.datasets.utils` — both removed in current `torchaudio` versions. Neither function is used by anything this project calls (only by `load_Qualcomm_keyword`, an unused data-loading utility); the import was removed rather than stubbed.
- Missing `__init__.py` files at three levels of the submodule's package structure, required for its relative imports to resolve at all.

---

## 3. Domain considerations, addressed directly rather than assumed away

The checkpoint (`WaveNet_Speech_Commands`, per its class name) is trained on SC09 — a dataset of 1-second isolated spoken digits (0-9) — a substantially different domain from this project's 3-second continuous LibriSpeech sentences. Two things were verified empirically before treating this as usable, rather than assuming either that it would work or that it wouldn't:

- **Architecture**: `WaveNet.py` is fully convolutional (dilated `Conv1d` residual blocks, no fixed-size pooling or time-axis fully-connected layers) — confirmed length-agnostic by direct test: a 48,000-sample (3-second) input produced a 48,000-sample output with no shape error, no chunking required.
- **Output sanity**: purifying real watermarked audio (not synthetic noise) produced properly-bounded, structurally sane output (comparable amplitude range to the input, no NaN, no explosion) — confirmed before any accuracy evaluation was run.
- **Methodological precedent**: SafeSpeech's own published AudioPure evaluation (which this project's plan explicitly cites as a comparison target) applies this same SC09-trained checkpoint to non-digit speech (LibriTTS/CMU ARCTIC) — using it here is the methodologically consistent choice for direct comparability to prior work, not a compromise.

---

## 4. Method

For each held-out, speaker-disjoint eval utterance: embed a 16-bit watermark (producing `recon_wm`), measure detection accuracy on the clean watermarked audio (a sanity check against Stage 1's own reported numbers), purify `recon_wm` through AudioPure (`reverse_timestep=25` of 200, AudioPure's own default), then re-measure detection accuracy on the purified result. The gap between pre- and post-purification accuracy is the headline metric. Evaluated on the same 25-utterance, speaker-disjoint eval set used throughout this project, across four checkpoints: pretrained VoiceMark (baseline), Stage 1 (traceability-only), Stage 1 with VC-augmentation training, and Stage 2 (disruption-trained, sim-mode).

**Practical note**: purification takes ~19-21 seconds per 3-second utterance on Kaggle's T4/P100 GPUs — confirmed via direct timing before committing to a full evaluation sweep, consistent with the original project plan's anticipated risk that diffusion-based purification would be the pipeline's slowest component.

---

## 5. Result

| Condition | ACC before | ACC after | Drop |
|---|---|---|---|
| Baseline (pretrained VoiceMark) | 0.9825 | 0.5050 | 0.4775 |
| Stage 1 (traceability fine-tuned) | 1.0000 | 0.4875 | 0.5125 |
| Stage 1 + VC-augmentation training | 0.9875 | 0.5025 | 0.4850 |
| Stage 2 (disruption fine-tuned, sim-mode) | 0.9950 | 0.5025 | 0.4925 |

**All four conditions converge to the chance floor.** Bitwise detection accuracy has a theoretical floor of 0.50 (each bit independently guessable); all four post-purification values (0.4875-0.5050) sit tightly clustered on top of it, with a total spread of just 0.0175 across four differently-trained models. This is not "weakened detection" — it is complete destruction of the watermark signal, indistinguishable from random guessing, and this outcome is consistent regardless of which training approach was applied.

**The augmentation-training result is the most specific and informative finding here.** Stage 1's own augmentation ablation (masking, shuffling, content replacement, quantization noise) demonstrated real, measurable, reproducible robustness gains against those particular corruption types. That same checkpoint shows **no benefit whatsoever** against AudioPure (0.5025, statistically indistinguishable from baseline's 0.5050) — a precise, mechanistic finding: robustness learned against those corruption categories does not transfer to a fundamentally different attack mechanism (diffusion-based reconstruction from a partially-noised latent state), rather than a vague "nothing helps."

---

## 6. Why this is the thesis's central contribution

1. **Directly answers the project's founding question.** Neither VoiceMark nor SafeSpeech test this specific combination (VoiceMark's watermark against AudioPure's purification). This project provides the first direct, quantified answer: the watermark does not survive.
2. **Statistically unambiguous, not requiring repeated-run averaging to interpret.** Unlike the Stage 2 SIM measurements (which needed three-run averaging to distinguish real effects from ~0.02 evaluation noise), this result's effect size (a ~0.48-0.51 drop to the chance floor) is roughly 25x larger than any noise floor observed elsewhere in this project — a single evaluation run here is already conclusive.
3. **Specific, not just categorical.** The finding isn't merely "purification works" — it's "purification works regardless of the traceability, distortion-robustness, or disruption training strategies tested in this thesis," with the augmentation-training null result specifically demonstrating *why* (attack-mechanism mismatch, not insufficient training).
4. **Built on a fully-verified pipeline.** Every component — the purification mechanism's correctness (verified against dead code in the alternative implementation), the checkpoint's authenticity (corroborated by independent third-party usage), the architecture's length-agnosticism (empirically tested, not assumed), and the domain-mismatch question (addressed via precedent and direct output verification) — was checked against real evidence before being relied upon, consistent with this project's approach throughout.

---

## 7. Known limitations, stated explicitly

- The purification checkpoint's SC09 training domain (isolated digits) versus this project's continuous-sentence evaluation data remains a domain gap, even though methodologically justified by precedent (Section 3) and empirically sane output (no artifacts, proper bounding). A checkpoint trained on matched-domain data, if one becomes available, would strengthen this result further, though the magnitude of the observed effect (destruction to the chance floor) leaves little room for a domain-matched result to look substantially different.
- Only `reverse_timestep=25` (AudioPure's own published default) was tested. A sweep across this parameter (weaker/stronger purification) was not performed given compute constraints — worth noting that since the result already sits at the chance floor at this setting, weaker purification is the more informative direction for future work (is there a threshold below which some watermark signal survives?), not stronger.
- As with all evaluations in this project, results are drawn from a 25-utterance, speaker-disjoint eval set — larger-scale evaluation would strengthen statistical confidence, though the effect size here is large enough that this is a lower priority than it was for Stage 2's more marginal findings.
