"""
src/eval/far_eval.py

Implements VoiceMark's False Attribution Rate (FAR) metric, exactly as
described in the paper (arXiv 2505.21568): "FAR simulates real-world
multi-candidate identification by comparing the Hamming distance of each
decoded watermark to 100 candidates" (1 ground truth + 99 random).

THIS IS NOT THE SAME METRIC as false_positive_rate.py already in this repo.
That script measures whether the presence classifier fires on CLEAN,
never-watermarked audio (a detection-presence metric, single reference,
thresholded). FAR is a DIFFERENT question: given audio that WAS watermarked
and decoded, would the recovered bits be correctly attributed to the true
owner's message, or could they plausibly be mistaken for one of many other
candidate messages in circulation? This matters specifically for the
traceability use case (Stage 1's stated goal) -- ACC alone doesn't tell you
whether decoded bits are unique enough to identify the right source among
many candidates; FAR does.

Protocol (matches the paper's stated method):
  1. Embed a random 16-bit true message into each utterance (same convention
     as every other eval script in this project).
  2. Decode the FULL 16-bit message from the (optionally attacked/purified)
     audio -- not just aggregate bitwise accuracy, the actual decoded vector,
     since Hamming distance needs it.
  3. Sample n_candidates - 1 random 16-bit distractor messages per utterance
     (default 99, giving 100 total candidates as in the paper), resampling
     any distractor that collides exactly with the true message.
  4. Compute Hamming distance from the DECODED bits to all 100 candidates.
  5. A trial is a FALSE ATTRIBUTION if any distractor's Hamming distance is
     <= the true message's distance -- the conservative reading (a tie means
     the system cannot uniquely attribute the recovered bits to the right
     owner, so it counts against correct attribution, not for it).
  FAR = (# false-attribution trials) / (# total trials)

Optionally applies AudioPure purification before decoding (--apply_audiopure),
mirroring audiopure_eval.py, so FAR can be reported both under clean
conditions and under the purification attack that's this project's central
finding for ACC -- useful for showing whether purification also destroys
attribution-level identifiability, not just raw bit accuracy.

One model (and one denoiser, if used) per process -- same safe pattern as
every other eval script in this project (see save_audio_samples.py's own
note on GPU state corruption from sequential model construction).

Usage:
    python src/eval/far_eval.py --output results_baseline_far.json
    python src/eval/far_eval.py --checkpoint ./checkpoints/stage1_full_recalibrated_v3/recalibrated_final.pt --output results_stage1_far.json
    python src/eval/far_eval.py --checkpoint ./checkpoints/stage1_full_recalibrated_v3/recalibrated_final.pt \
        --apply_audiopure --repo_root . --output results_stage1_far_audiopure.json
"""

import os
import sys
import json
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

from backbone import VoiceMarkBackbone
from adapters import apply_lora_adapters
from librispeech import LibriSpeechSubset, collate_librispeech
from torch.utils.data import DataLoader


def build_backbone(lora_checkpoint_path: str = None, r: int = 8, alpha: int = 16):
    backbone = VoiceMarkBackbone()
    apply_lora_adapters(backbone, r=r, alpha=alpha)
    if lora_checkpoint_path is not None:
        ckpt = torch.load(lora_checkpoint_path, map_location="cpu", weights_only=False)
        backbone.model.load_state_dict(ckpt["lora_state_dict"], strict=False)
        print(f"[build_backbone] Loaded checkpoint {lora_checkpoint_path}")
    else:
        print("[build_backbone] Using baseline (LoRA zero-init, == pretrained VoiceMark)")
    return backbone


def build_audiopure_denoiser(repo_root: str, reverse_timestep: int = 25):
    """Same loader as audiopure_eval.py -- see that file for the vendored/patched
    import notes (two broken torchaudio.datasets.utils imports were removed)."""
    sys.path.insert(0, repo_root)
    from external.audiopure.diffusion_models.diffwave_ddpm import create_diffwave_model

    model_path = os.path.join(
        repo_root, "external/audiopure/diffusion_models/DiffWave_Unconditional/"
        "exp/ch256_T200_betaT0.02/logs/checkpoint/1000000.pkl"
    )
    config_path = os.path.join(
        repo_root, "external/audiopure/diffusion_models/DiffWave_Unconditional/config.json"
    )
    denoiser = create_diffwave_model(model_path=model_path, config_path=config_path,
                                      reverse_timestep=reverse_timestep)
    print(f"[build_audiopure_denoiser] Loaded DiffWave denoiser (reverse_timestep={reverse_timestep})")
    return denoiser


def decode_message(chunk_logits: torch.Tensor, nchunk_size: int = 4) -> torch.Tensor:
    """
    Decodes the FULL predicted 16-bit message from chunk_logits, using the
    EXACT same bit order voicemark_losses.py::bits_to_chunk_indices packs
    (chunk_val = sum(bits[bit_idx] << bit_idx for bit_idx in range(nchunk_size))),
    so decoded[:, i*nchunk_size + bit_idx] = (pred_chunk_val >> bit_idx) & 1.

    chunk_logits: [batch, nchunks, 2**nchunk_size]
    Returns: [batch, nbits] predicted binary message.
    """
    pred_chunks = torch.argmax(chunk_logits, dim=-1)  # [batch, nchunks]
    batch, nchunks = pred_chunks.shape
    bits = []
    for i in range(nchunks):
        chunk_val = pred_chunks[:, i]
        for bit_idx in range(nchunk_size):
            bits.append(((chunk_val >> bit_idx) & 1).unsqueeze(1))
    return torch.cat(bits, dim=1)  # [batch, nchunks * nchunk_size]


def sample_distractors(true_message: torch.Tensor, n_distractors: int, generator, device) -> torch.Tensor:
    """
    Samples n_distractors random 16-bit messages PER utterance, resampling
    any exact collision with that utterance's true message (a distractor
    identical to the true message isn't a meaningful distractor -- with
    16 bits this is rare, ~1/65536 per draw, but cheap to guard against).

    Returns: [batch, n_distractors, nbits]
    """
    batch, nbits = true_message.shape
    distractors = torch.randint(0, 2, (batch, n_distractors, nbits), generator=generator, device=device)
    for b in range(batch):
        for d in range(n_distractors):
            tries = 0
            while torch.equal(distractors[b, d], true_message[b]) and tries < 10:
                distractors[b, d] = torch.randint(0, 2, (nbits,), generator=generator, device=device)
                tries += 1
    return distractors


def compute_far(decoded: torch.Tensor, true_message: torch.Tensor, distractors: torch.Tensor):
    """
    decoded, true_message: [batch, nbits]
    distractors: [batch, n_distractors, nbits]

    Returns per-utterance (false_attribution: [batch] bool, true_dist: [batch],
    min_distractor_dist: [batch]).
    """
    true_dist = (decoded != true_message).float().sum(dim=1)  # [batch]
    distractor_dist = (decoded.unsqueeze(1) != distractors).float().sum(dim=2)  # [batch, n_distractors]
    min_distractor_dist = distractor_dist.min(dim=1).values  # [batch]
    false_attribution = (min_distractor_dist <= true_dist)  # ties count as false attribution
    return false_attribution, true_dist, min_distractor_dist


def run_far_eval(backbone, eval_loader, device, denoiser=None, n_candidates: int = 100, seed: int = 321) -> dict:
    backbone.model.eval()
    n_distractors = n_candidates - 1
    fa_flags, true_dists, min_distractor_dists = [], [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            clean_audio = batch["waveform"].to(device)
            gen = torch.Generator(device=device).manual_seed(seed + batch_idx)
            message = torch.randint(0, 2, (clean_audio.shape[0], 16), generator=gen, device=device)

            out = backbone.forward_full(clean_audio, message)
            recon_wm = out["recon_wm"]

            audio_to_decode = recon_wm
            if denoiser is not None:
                audio_to_decode = denoiser(recon_wm)

            detect_feat = backbone.model.st_model.forward_feature(audio_to_decode)
            _, chunk_logits = backbone.model.detector(detect_feat)
            decoded = decode_message(chunk_logits)

            distractors = sample_distractors(message, n_distractors, gen, device)
            fa, true_dist, min_dist = compute_far(decoded, message, distractors)

            fa_flags.extend(fa.cpu().tolist())
            true_dists.extend(true_dist.cpu().tolist())
            min_distractor_dists.extend(min_dist.cpu().tolist())

            print(f"  utterance {batch_idx}: true_dist={true_dist.float().mean().item():.2f}, "
                  f"min_distractor_dist={min_dist.float().mean().item():.2f}, "
                  f"false_attribution={fa.float().mean().item():.2f}")

    far = sum(fa_flags) / len(fa_flags) if fa_flags else float("nan")
    return {
        "far": far,
        "n_trials": len(fa_flags),
        "n_candidates": n_candidates,
        "mean_true_hamming_dist": sum(true_dists) / len(true_dists),
        "mean_min_distractor_hamming_dist": sum(min_distractor_dists) / len(min_distractor_dists),
        "false_attribution_flags": fa_flags,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--data_root", type=str, default="./data/librispeech")
    p.add_argument("--n_speakers", type=int, default=10)
    p.add_argument("--utterances_per_speaker", type=int, default=10)
    p.add_argument("--n_eval_speakers", type=int, default=5)
    p.add_argument("--eval_utterances_per_speaker", type=int, default=5)
    p.add_argument("--crop_seconds", type=float, default=3.0)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--n_candidates", type=int, default=100, help="1 true + (n_candidates - 1) random distractors, matching VoiceMark's own 100.")
    p.add_argument("--seed", type=int, default=321)
    p.add_argument("--apply_audiopure", action="store_true",
                    help="Purify recon_wm through AudioPure's DiffWave denoiser before decoding, "
                         "to report FAR under purification alongside the existing ACC number.")
    p.add_argument("--repo_root", type=str, default=".", help="Needed only if --apply_audiopure is set.")
    p.add_argument("--reverse_timestep", type=int, default=25)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    eval_ds = LibriSpeechSubset(
        root=args.data_root, n_speakers=args.n_speakers,
        utterances_per_speaker=args.utterances_per_speaker,
        n_eval_speakers=args.n_eval_speakers,
        eval_utterances_per_speaker=args.eval_utterances_per_speaker,
        sample_rate=16000, crop_seconds=args.crop_seconds, split="eval",
    )
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_librispeech)

    label = "baseline" if args.checkpoint is None else args.checkpoint
    condition = " + AudioPure purification" if args.apply_audiopure else " (clean)"
    print(f"\n{'=' * 60}\nEvaluating FAR: {label}{condition}\n{'=' * 60}")

    backbone = build_backbone(lora_checkpoint_path=args.checkpoint, r=args.lora_r, alpha=args.lora_alpha)
    denoiser = build_audiopure_denoiser(args.repo_root, reverse_timestep=args.reverse_timestep) if args.apply_audiopure else None

    results = run_far_eval(backbone, eval_loader, device, denoiser=denoiser,
                            n_candidates=args.n_candidates, seed=args.seed)

    print(f"\n{'=' * 60}")
    print(f"FAR ({results['n_candidates']} candidates, n={results['n_trials']}): {results['far']:.4f}")
    print(f"Mean true-message Hamming distance:      {results['mean_true_hamming_dist']:.3f} / 16 bits")
    print(f"Mean closest-distractor Hamming distance: {results['mean_min_distractor_hamming_dist']:.3f} / 16 bits")

    with open(args.output, "w") as f:
        json.dump({
            "label": label, "checkpoint": args.checkpoint,
            "condition": "audiopure" if args.apply_audiopure else "clean",
            "results": {
                "far": results["far"],
                "n_trials": results["n_trials"],
                "n_candidates": results["n_candidates"],
                "mean_true_hamming_dist": results["mean_true_hamming_dist"],
                "mean_min_distractor_hamming_dist": results["mean_min_distractor_hamming_dist"],
            },
        }, f, indent=2)
    print(f"[main] Saved results to {args.output}")


if __name__ == "__main__":
    main()
