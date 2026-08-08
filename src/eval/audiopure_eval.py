"""
src/eval/audiopure_eval.py

THE central test of the thesis: does VoiceMark's watermark survive AudioPure's
diffusion-based purification attack? Neither VoiceMark nor SafeSpeech test
this specific combination -- this is the untested claim the whole project
builds toward.

Method: for each held-out eval utterance, embed a watermark (producing
recon_wm), then purify recon_wm through AudioPure's DiffWave denoiser, then
re-run detection on the PURIFIED audio. Reports bitwise ACC both BEFORE and
AFTER purification -- the gap between them is the actual headline number.

One model per process (same safe pattern as every other eval script in this
project). Output JSON matches augmentation_robustness.py's schema so
compare_results.py-style tooling works, but with its own results dict shape
(acc_before, acc_after -- not sim-style keys).

Usage:
    python src/eval/audiopure_eval.py --output results_baseline_audiopure.json
    python src/eval/audiopure_eval.py --checkpoint ./checkpoints/stage1_full/stage1_epoch29.pt --output results_stage1_audiopure.json
    python src/eval/audiopure_eval.py --checkpoint ./checkpoints/stage2_sim_longrun/stage2_epoch29.pt --output results_stage2_sim_audiopure.json
"""

import os
import sys
import json
import argparse
import time
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
        missing, unexpected = backbone.model.load_state_dict(ckpt["lora_state_dict"], strict=False)
        print(f"[build_backbone] Loaded checkpoint {lora_checkpoint_path} (epoch {ckpt.get('epoch')})")
    else:
        print("[build_backbone] Using baseline (LoRA zero-init, == pretrained VoiceMark)")
    return backbone


def build_audiopure_denoiser(repo_root: str, reverse_timestep: int = 25):
    """
    Loads AudioPure's DiffWave denoiser. Imports the vendored, ALREADY-PATCHED
    copy at external/audiopure (two broken torchaudio.datasets.utils imports
    were removed -- see this project's own debugging notes; if you re-clone
    the submodule fresh, those patches need to be reapplied before this import
    will succeed).
    """
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
    print(f"[build_audiopure_denoiser] Loaded DiffWave denoiser "
          f"({sum(p.numel() for p in denoiser.parameters()):,} params, "
          f"reverse_timestep={reverse_timestep})")
    return denoiser


def compute_detection_accuracy(chunk_logits: torch.Tensor, message: torch.Tensor, nchunk_size: int = 4) -> float:
    pred_chunks = torch.argmax(chunk_logits, dim=-1)
    batch, nchunks = pred_chunks.shape
    correct_bits, total_bits = 0, 0
    for i in range(nchunks):
        true_chunk = message[:, i * nchunk_size:(i + 1) * nchunk_size]
        pred_val = pred_chunks[:, i]
        for bit_idx in range(nchunk_size):
            pred_bit = (pred_val >> bit_idx) & 1
            correct_bits += (pred_bit == true_chunk[:, bit_idx]).sum().item()
            total_bits += batch
    return correct_bits / total_bits


def run_audiopure_eval(backbone, denoiser, eval_loader, device, seed: int = 123) -> dict:
    backbone.model.eval()
    acc_before_list = []
    acc_after_list = []
    purify_times = []

    for batch_idx, batch in enumerate(eval_loader):
        clean_audio = batch["waveform"].to(device)
        gen = torch.Generator(device=device).manual_seed(seed + batch_idx)
        message = torch.randint(0, 2, (clean_audio.shape[0], 16), generator=gen, device=device)

        with torch.no_grad():
            out = backbone.forward_full(clean_audio, message)
            recon_wm = out["recon_wm"]

            # Detection BEFORE purification (sanity baseline -- should match
            # Stage 1's own reported ACC for this checkpoint)
            detect_feat_before = backbone.model.st_model.forward_feature(recon_wm)
            _, chunk_logits_before = backbone.model.detector(detect_feat_before)
            acc_before = compute_detection_accuracy(chunk_logits_before, message)

            # Purify, then detect AFTER -- the actual headline test
            t0 = time.time()
            purified = denoiser(recon_wm)
            purify_times.append(time.time() - t0)

            detect_feat_after = backbone.model.st_model.forward_feature(purified)
            _, chunk_logits_after = backbone.model.detector(detect_feat_after)
            acc_after = compute_detection_accuracy(chunk_logits_after, message)

        acc_before_list.append(acc_before)
        acc_after_list.append(acc_after)
        print(f"  utterance {batch_idx}: acc_before={acc_before:.4f} acc_after={acc_after:.4f} "
              f"(purify_time={purify_times[-1]:.1f}s)")

    return {
        "acc_before_mean": sum(acc_before_list) / len(acc_before_list),
        "acc_after_mean": sum(acc_after_list) / len(acc_after_list),
        "acc_before_values": acc_before_list,
        "acc_after_values": acc_after_list,
        "mean_purify_time_s": sum(purify_times) / len(purify_times),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--repo_root", type=str, default=".", help="Path to repo root (contains external/audiopure)")
    p.add_argument("--reverse_timestep", type=int, default=25,
                    help="AudioPure's own default -- higher = stronger purification but slower.")
    p.add_argument("--data_root", type=str, default="./data/librispeech")
    p.add_argument("--n_speakers", type=int, default=10)
    p.add_argument("--utterances_per_speaker", type=int, default=10)
    p.add_argument("--n_eval_speakers", type=int, default=5)
    p.add_argument("--eval_utterances_per_speaker", type=int, default=5)
    p.add_argument("--crop_seconds", type=float, default=3.0)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
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
    print(f"\n{'=' * 60}\nEvaluating: {label}\n{'=' * 60}")

    backbone = build_backbone(lora_checkpoint_path=args.checkpoint, r=args.lora_r, alpha=args.lora_alpha)
    denoiser = build_audiopure_denoiser(args.repo_root, reverse_timestep=args.reverse_timestep)

    results = run_audiopure_eval(backbone, denoiser, eval_loader, device)

    print(f"\n{'=' * 60}")
    print(f"ACC before purification: {results['acc_before_mean']:.4f}")
    print(f"ACC after purification:  {results['acc_after_mean']:.4f}")
    print(f"ACC drop: {results['acc_before_mean'] - results['acc_after_mean']:.4f}")
    print(f"Mean purification time: {results['mean_purify_time_s']:.2f}s/utterance")

    with open(args.output, "w") as f:
        json.dump({
            "label": label, "checkpoint": args.checkpoint,
            "results": {
                "acc_before": results["acc_before_mean"],
                "acc_after": results["acc_after_mean"],
                "acc_drop": results["acc_before_mean"] - results["acc_after_mean"],
            },
        }, f, indent=2)
    print(f"[main] Saved results to {args.output}")


if __name__ == "__main__":
    main()
