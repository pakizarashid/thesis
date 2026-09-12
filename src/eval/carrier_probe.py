"""
src/eval/carrier_probe.py

CARRIER-PROBE (2026-09-12 pre-registration, `carrier-fragility-pivot-2026-09-12.md`):
does VoiceMark's watermark-bearing RVQ residual (SpeechTokenizer layers 2-8,
`acoustic`/`acoustic_wm` in backbone.py's own terms) structurally survive each TTS
architecture's own reference-conditioning transformation? Pure forward-pass
measurement -- no PGD, no gradient descent, no training, no new dependency (reuses
SpeechTokenizer, already loaded by build_backbone in every script in this repo).

WHY THIS SHAPE, NOT A DIRECT acoustic-vs-acoustic COMPARISON:
`acoustic` (the PRE-watermark residual) is computed inside the vendored, uneditable
external/voicemark/models.py as `e - quantized_list[0]` -- the SUM across all 7 RVQ
layers, with no per-layer breakdown exposed anywhere this repo can reach without
modifying that vendored file. `acoustic_wm`, by contrast, IS decomposable per-layer
from this repo, because msg_processor is called once per layer and summed
(backbone.py's own documented internals, already exploited by
disruption_pgd_latent.py's LatentDeltaMsgProcessor -- same interception mechanism,
reused here via LayerRecorder, which RECORDS each per-layer call instead of
perturbing it). So the measurement embeds the SAME watermark message into BOTH the
source and the clone, through msg_processor, and compares the resulting per-layer
outputs. Since msg_processor is the same fixed, small, near-identity-preserving
function applied to both sides with an IDENTICAL message, its own contribution
approximately cancels in the comparison -- what's left is dominated by how similar
the underlying per-layer acoustic content (x) is between source and clone. This is
an indirect measurement of the same thing a direct acoustic-vs-acoustic comparison
would show, not an equivalent-by-construction one -- stated here so the method's
own assumption is visible, not buried.

MEASUREMENT: per architecture, per sample: embed the same message into the clean
reference AND the clone (same msg_processor, same message), record each of the 7
per-layer msg_processor outputs on both sides (RVQ layers 2-8), compute per-layer
cosine similarity, and also the pooled (summed, i.e. full acoustic_wm) similarity as
a cross-check. Aggregate over samples; compare the resulting per-architecture
aggregate against the already-measured ACC ladder:
    YourTTS 0.5337 | XTTS-v2 0.6119 | CosyVoice 0.7669 | MaskGCT 0.9137 | F5-TTS 0.9300

INPUT LAYOUT (matches the two conventions already in use in this repo -- either is
accepted, matched by glob so no renaming is needed):
    <dir>/sample{i}_reference.wav        the clean source utterance
    <dir>/sample{i}_clone.wav            OR
    <dir>/sample{i}_clone_{cloner}.wav   that utterance's zero-shot clone
(cloner_watermark_eval.py --save_clones_dir already writes the second form for
CosyVoice/MaskGCT/F5-TTS. If YourTTS/XTTS sample dirs from the five-architecture
ladder run aren't already on disk under either convention, regenerate a small n
(10-15 is enough) with --save_clones_dir / equivalent before running this script.)

DISCIPLINE: run --diagnostic first (1 sample pair, verbose shape/shape-mismatch
printing) before trusting an aggregate run -- same rule as every other script in
this repo. In particular this script has NEVER been run before, so the exact
tensor shape st_model expects (verified elsewhere only via the dataset loaders'
collate functions, which this script does NOT use, since audio comes from disk) is
UNVERIFIED -- the diagnostic prints shapes at every step specifically to catch a
mismatch before spending time on all 5 architectures.

Usage:
    # 1. Diagnostic, one architecture's sample dir:
    python src/eval/carrier_probe.py --sample_dir ./audio_samples/ladder_yourtts --diagnostic

    # 2. Once diagnostic looks sane, run per architecture:
    python src/eval/carrier_probe.py --sample_dir ./audio_samples/ladder_yourtts \\
        --cloner_label yourtts --output results/results_carrierprobe_yourtts.json
    # repeat for xtts / cosyvoice / maskgct / f5tts, then aggregate the 5 JSONs.
"""
import os
import re
import sys
import glob
import json
import argparse

import torch
import torch.nn.functional as F
import torchaudio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "losses"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from disruption_pgd import build_backbone, random_message

RVQ_LAYER_LABELS = [f"layer{k}" for k in range(2, 9)]  # 7 labels, RVQ layers 2-8


class LayerRecorder:
    """
    Same interception mechanism as disruption_pgd_latent.py's LatentDeltaMsgProcessor
    (st_model.forward() calls this once per RVQ layer in `subset`, summing the
    results into acoustic_wm) -- but RECORDS each per-layer output instead of
    perturbing it. Call .reset() before each fresh forward pass (mirrors that
    class's own .set_delta() reset convention).
    """

    def __init__(self, base_msg_processor):
        self.base = base_msg_processor
        self.calls = []

    def reset(self):
        self.calls = []

    def __call__(self, x, message):
        out = self.base(x, message)
        self.calls.append(out.detach())
        return out


def load_wav_16k(path: str, device: str) -> torch.Tensor:
    """Loads a WAV, mixes to mono, resamples to 16kHz, returns shape [1, T]."""
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav.to(device)


def layerwise_cosine(calls_a, calls_b, verbose=False):
    """
    calls_a/calls_b: lists of tensors, one per RVQ-layer call (expected length 7,
    verified by the caller). Returns (per_layer_sims: list[float], pooled_sim: float,
    frame_mismatch_rel: float) -- pooled is cosine similarity of the SUMMED
    (acoustic_wm-equivalent) tensors, the same aggregate quantity every other SIM
    measurement in this project reports.

    FRAME-COUNT MISMATCH: reference and clone are two DIFFERENT recordings (one
    human, one TTS-synthesized), so their time-axis length essentially never matches
    exactly even when they say the same words -- natural prosodic-timing variation.
    This crops both sides to the shorter one's frame count (same crop applied to
    every layer, since layer length is determined by audio duration, not by layer
    index) so cosine similarity is well-defined. A SMALL mismatch (a few percent) is
    expected and fine. A LARGE one (>25%) is a different problem: it means the clone
    was very likely synthesized from different words than the reference actually
    says, which this crop cannot fix -- see gen_samples_yourtts.py / gen_samples_xtts.py
    and cloner_watermark_eval.py's --gen_text_from_transcript, which address this at
    generation time by making the clone target the reference's own transcript.
    """
    assert len(calls_a) == len(calls_b), (
        f"layer-call-count mismatch: {len(calls_a)} vs {len(calls_b)} -- the "
        f"call-count assumption (7 calls, RVQ layers 2-8) this script inherits from "
        f"disruption_pgd_latent.py's LatentDeltaMsgProcessor did not hold here. Do "
        f"not trust any number from this run until this is resolved."
    )
    t_a = calls_a[0].shape[-1]
    t_b = calls_b[0].shape[-1]
    t_min = min(t_a, t_b)
    frame_mismatch_rel = abs(t_a - t_b) / max(t_a, t_b) if max(t_a, t_b) > 0 else 0.0
    if t_a != t_b and verbose:
        flag = "  !!! LARGE -- check the clone's gen_text actually matches the reference's own words." \
            if frame_mismatch_rel > 0.25 else ""
        print(f"  [layerwise_cosine] frame-count mismatch: reference={t_a} clone={t_b} "
              f"(cropping both to {t_min}). relative difference: {frame_mismatch_rel:.1%}{flag}")
    calls_a = [a[..., :t_min] for a in calls_a]
    calls_b = [b[..., :t_min] for b in calls_b]

    per_layer = []
    for a, b in zip(calls_a, calls_b):
        a_flat = a.flatten(start_dim=1) if a.dim() > 1 else a.unsqueeze(0)
        b_flat = b.flatten(start_dim=1) if b.dim() > 1 else b.unsqueeze(0)
        sim = F.cosine_similarity(a_flat, b_flat, dim=-1).mean().item()
        per_layer.append(sim)
    sum_a = torch.stack(calls_a, dim=0).sum(dim=0)
    sum_b = torch.stack(calls_b, dim=0).sum(dim=0)
    sum_a_flat = sum_a.flatten(start_dim=1) if sum_a.dim() > 1 else sum_a.unsqueeze(0)
    sum_b_flat = sum_b.flatten(start_dim=1) if sum_b.dim() > 1 else sum_b.unsqueeze(0)
    pooled = F.cosine_similarity(sum_a_flat, sum_b_flat, dim=-1).mean().item()
    return per_layer, pooled, frame_mismatch_rel


def probe_one_pair(backbone, reference_audio, clone_audio, message, device, verbose=False):
    st_model = backbone.model.st_model
    real_msg_processor = backbone.model.msg_processor

    rec_ref = LayerRecorder(real_msg_processor)
    rec_clone = LayerRecorder(real_msg_processor)

    with torch.no_grad():
        if verbose:
            print(f"  reference_audio.shape={tuple(reference_audio.shape)} "
                  f"clone_audio.shape={tuple(clone_audio.shape)} message.shape={tuple(message.shape)}")
        _o_ref, _owm_ref, _ac_ref, acoustic_wm_ref = st_model(
            reference_audio, msg_processor=rec_ref, message=message
        )
        _o_clone, _owm_clone, _ac_clone, acoustic_wm_clone = st_model(
            clone_audio, msg_processor=rec_clone, message=message
        )

    if verbose:
        print(f"  msg_processor call count: reference={len(rec_ref.calls)} clone={len(rec_clone.calls)} "
              f"(expect 7, RVQ layers 2-8, per backbone.py's documented internals)")
        if rec_ref.calls:
            print(f"  per-call shape: {tuple(rec_ref.calls[0].shape)}")

    per_layer_sim, pooled_sim, frame_mismatch_rel = layerwise_cosine(
        rec_ref.calls, rec_clone.calls, verbose=verbose
    )
    return per_layer_sim, pooled_sim, len(rec_ref.calls), frame_mismatch_rel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample_dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Omit to use pretrained VoiceMark (zero-init LoRA) -- matches the "
                        "protocol prereg-conditioning-bandwidth-2026-09-10.md used for the "
                        "five-architecture ladder this script's results are compared against.")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--cloner_label", type=str, default=None,
                   help="Name for this architecture in the output JSON, e.g. yourtts/xtts/"
                        "cosyvoice/maskgct/f5tts. Required unless --diagnostic.")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--diagnostic", action="store_true",
                   help="1 sample pair, verbose shape printing, exits without requiring "
                        "--cloner_label/--output. Run this FIRST -- this script has never "
                        "been run before; the diagnostic is what confirms st_model accepts "
                        "disk-loaded audio shaped the way this script loads it.")
    args = p.parse_args()

    if not args.diagnostic and (args.cloner_label is None or args.output is None):
        p.error("--cloner_label and --output are required unless --diagnostic is set")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = build_backbone(args.checkpoint, args.lora_r, args.lora_alpha, False, 32)

    refs = sorted(glob.glob(os.path.join(args.sample_dir, "sample*_reference.wav")))
    if not refs:
        raise SystemExit(f"No sample*_reference.wav found in {args.sample_dir}.")
    pairs = []
    for ref_path in refs:
        idx = re.search(r"sample(\d+)_reference\.wav$", os.path.basename(ref_path)).group(1)
        # Accept either sample{i}_clone.wav or sample{i}_clone_{cloner}.wav (the latter
        # is what cloner_watermark_eval.py --save_clones_dir actually writes).
        candidates = sorted(glob.glob(os.path.join(args.sample_dir, f"sample{idx}_clone*.wav")))
        if candidates:
            pairs.append((idx, ref_path, candidates[0]))
    print(f"[carrier_probe] found {len(pairs)} reference/clone pairs in {args.sample_dir}")
    if not pairs:
        raise SystemExit(f"No matching sample{{i}}_clone.wav for any reference in {args.sample_dir}.")

    if args.diagnostic:
        idx, ref_path, clone_path = pairs[0]
        print(f"\n{'=' * 60}\nDIAGNOSTIC: sample {idx}\n{'=' * 60}")
        reference_audio = load_wav_16k(ref_path, device).unsqueeze(0)  # [1, 1, T]
        clone_audio = load_wav_16k(clone_path, device).unsqueeze(0)
        message = random_message(16, 1, device, seed=123)
        per_layer_sim, pooled_sim, n_calls, frame_mismatch_rel = probe_one_pair(
            backbone, reference_audio, clone_audio, message, device, verbose=True
        )
        if n_calls != 7:
            print(f"\n  !!! WARNING: expected 7 msg_processor calls (RVQ layers 2-8), got "
                  f"{n_calls}. The call-count assumption this script inherits from "
                  f"disruption_pgd_latent.py does not hold for this checkpoint/backbone "
                  f"config -- do not trust per-layer numbers below until this is resolved.")
        print(f"\n  per-layer cosine similarity (layers 2-8): "
              f"{[round(s, 4) for s in per_layer_sim]}")
        print(f"  pooled (summed acoustic_wm) cosine similarity: {pooled_sim:.4f}")
        if frame_mismatch_rel > 0.25:
            print(f"\n  !!! frame-count mismatch was {frame_mismatch_rel:.1%} on this sample -- "
                  f"large enough that the clone likely says DIFFERENT words than the reference. "
                  f"If this repeats across samples, regenerate with --gen_text_from_transcript "
                  f"(cloner_watermark_eval.py) or check gen_samples_yourtts.py/gen_samples_xtts.py "
                  f"are using the reference's own transcript, not a fixed sentence, before trusting "
                  f"any similarity number from this architecture.")
        print("\n[carrier_probe] Diagnostic complete. If shapes/call-count looked sane, "
              "proceed to a full run (drop --diagnostic, add --cloner_label/--output).")
        return

    per_layer_all = []
    pooled_all = []
    frame_mismatch_all = []
    n_mismatch = 0
    n_large_frame_mismatch = 0
    for idx, ref_path, clone_path in pairs:
        reference_audio = load_wav_16k(ref_path, device).unsqueeze(0)
        clone_audio = load_wav_16k(clone_path, device).unsqueeze(0)
        message = random_message(16, 1, device, seed=123 + int(idx))
        per_layer_sim, pooled_sim, n_calls, frame_mismatch_rel = probe_one_pair(
            backbone, reference_audio, clone_audio, message, device, verbose=False
        )
        if n_calls != 7:
            n_mismatch += 1
            continue
        if frame_mismatch_rel > 0.25:
            n_large_frame_mismatch += 1
        per_layer_all.append(per_layer_sim)
        pooled_all.append(pooled_sim)
        frame_mismatch_all.append(frame_mismatch_rel)

    if n_mismatch:
        print(f"[carrier_probe] WARNING: {n_mismatch}/{len(pairs)} samples had an unexpected "
              f"msg_processor call count and were EXCLUDED from the aggregate below.")
    if n_large_frame_mismatch:
        print(f"[carrier_probe] WARNING: {n_large_frame_mismatch}/{len(per_layer_all)} samples "
              f"had a large (>25%) reference/clone frame-count mismatch -- likely the clone said "
              f"different words than the reference on those samples. They were still INCLUDED in "
              f"the aggregate below (cropped to the shorter length), but if this count is a large "
              f"fraction of n, treat the similarity numbers as unreliable until the clones are "
              f"regenerated with matched text (--gen_text_from_transcript).")

    n = len(per_layer_all)
    per_layer_mean = [sum(per_layer_all[i][k] for i in range(n)) / n for k in range(7)]
    pooled_mean = sum(pooled_all) / n

    print(f"\n{'=' * 70}\nCARRIER-PROBE: {args.cloner_label}  (n={n})\n{'=' * 70}")
    for label, sim in zip(RVQ_LAYER_LABELS, per_layer_mean):
        print(f"  {label:<10} mean cosine sim = {sim:.4f}")
    print(f"  {'pooled':<10} mean cosine sim = {pooled_mean:.4f}")

    result = {
        "label": "carrier_probe",
        "cloner": args.cloner_label,
        "sample_dir": args.sample_dir,
        "n": n,
        "n_excluded_mismatch": n_mismatch,
        "n_large_frame_mismatch": n_large_frame_mismatch,
        "mean_frame_mismatch_rel": sum(frame_mismatch_all) / n if n else None,
        "rvq_layer_labels": RVQ_LAYER_LABELS,
        "per_layer_mean_cosine_sim": per_layer_mean,
        "pooled_mean_cosine_sim": pooled_mean,
        "per_sample_per_layer": per_layer_all,
        "per_sample_pooled": pooled_all,
        "per_sample_frame_mismatch_rel": frame_mismatch_all,
    }
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[carrier_probe] Saved to {args.output}")


if __name__ == "__main__":
    main()
