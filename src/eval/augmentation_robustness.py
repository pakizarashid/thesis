"""
src/eval/augmentation_robustness.py

Evaluates detection ACC under distortion for ONE model per invocation
(baseline OR a specific checkpoint) -- deliberately NOT both in one process.
An earlier version constructed two backbones sequentially in the same process
(load baseline -> del -> torch.cuda.empty_cache() -> load checkpoint), which
produced corrupted/degenerate results for whichever model was built second
(observed as identical near-chance accuracy across all conditions). Running
one model per process sidesteps that entirely and is more robust.

Usage:
    # Baseline (LoRA at zero-init, == pretrained VoiceMark)
    python src/eval/augmentation_robustness.py --output results_baseline.json

    # A trained checkpoint
    python src/eval/augmentation_robustness.py --checkpoint ./checkpoints/stage1_full/stage1_epoch29.pt --output results_noaug.json
    python src/eval/augmentation_robustness.py --checkpoint ./checkpoints/stage1_aug/stage1_epoch29.pt --output results_aug.json

    # Then compare any set of result files:
    python src/eval/compare_results.py results_baseline.json results_noaug.json results_aug.json
"""

import os
import sys
import json
import argparse
import random as py_random
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

from backbone import VoiceMarkBackbone
from adapters import apply_lora_adapters
from librispeech import LibriSpeechSubset, collate_librispeech
from augment import AUGMENTATION_FNS
from torch.utils.data import DataLoader


def build_backbone(lora_checkpoint_path: str = None, r: int = 8, alpha: int = 16):
    backbone = VoiceMarkBackbone()
    apply_lora_adapters(backbone, r=r, alpha=alpha)

    if lora_checkpoint_path is not None:
        ckpt = torch.load(lora_checkpoint_path, map_location="cpu", weights_only=False)
        lora_state = ckpt["lora_state_dict"]
        missing, unexpected = backbone.model.load_state_dict(lora_state, strict=False)
        non_lora_missing = [k for k in missing if "_lora." in k]
        if non_lora_missing:
            print(f"[build_backbone] WARNING - {len(non_lora_missing)} LoRA keys missing "
                  f"from checkpoint (possible r/alpha mismatch): {non_lora_missing[:3]}...")
        if unexpected:
            print(f"[build_backbone] WARNING - unexpected keys in checkpoint: {unexpected[:3]}...")
        avg_acc = ckpt.get("avg_acc")
        avg_acc_str = f"{avg_acc:.4f}" if avg_acc is not None else "N/A"
        print(f"[build_backbone] Loaded LoRA weights from {lora_checkpoint_path} "
              f"(epoch {ckpt.get('epoch')}, train-time avg_acc={avg_acc_str})")
    else:
        print("[build_backbone] Using baseline (LoRA at zero-init, == pretrained VoiceMark)")

    return backbone


def compute_detection_accuracy(chunk_logits: torch.Tensor, message: torch.Tensor, nchunk_size: int = 4) -> float:
    pred_chunks = torch.argmax(chunk_logits, dim=-1)
    batch, nchunks = pred_chunks.shape
    correct_bits = 0
    total_bits = 0
    for i in range(nchunks):
        true_chunk = message[:, i * nchunk_size:(i + 1) * nchunk_size]
        pred_val = pred_chunks[:, i]
        for bit_idx in range(nchunk_size):
            pred_bit = (pred_val >> bit_idx) & 1
            true_bit = true_chunk[:, bit_idx]
            correct_bits += (pred_bit == true_bit).sum().item()
            total_bits += batch
    return correct_bits / total_bits


def apply_augmentation_to_batch(waveform_batch: torch.Tensor, aug_name: str, downsample_rate: int) -> torch.Tensor:
    device = waveform_batch.device
    fn = AUGMENTATION_FNS[aug_name]
    out = []
    for i in range(waveform_batch.shape[0]):
        wav_cpu = waveform_batch[i].detach().cpu()
        n_frames = wav_cpu.shape[-1] // downsample_rate
        if aug_name == "replacing":
            other_idx = (i + 1) % waveform_batch.shape[0]
            other_wav = waveform_batch[other_idx].detach().cpu()
            aug_wav, _ = fn(wav_cpu, downsample_rate, n_frames, other_waveform=other_wav)
        else:
            aug_wav, _ = fn(wav_cpu, downsample_rate, n_frames)
        out.append(aug_wav)
    return torch.stack(out).to(device)


def run_robustness_eval(backbone, eval_loader, device, augmentation_names, seed: int = 123):
    backbone.model.eval()
    downsample_rate = backbone.model.st_model.downsample_rate
    results = {name: [] for name in ["clean"] + list(augmentation_names)}

    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            clean_audio = batch["waveform"].to(device)

            gen = torch.Generator(device=device).manual_seed(seed + batch_idx)
            message = torch.randint(0, 2, (clean_audio.shape[0], 16), generator=gen, device=device)

            out = backbone.forward_full(clean_audio, message)
            recon_wm = out["recon_wm"]

            detect_feat = backbone.model.st_model.forward_feature(recon_wm)
            _, chunk_logits = backbone.model.detector(detect_feat)
            results["clean"].append(compute_detection_accuracy(chunk_logits, message))

            for aug_idx, aug_name in enumerate(augmentation_names):
                py_random.seed(seed * 100000 + batch_idx * 100 + aug_idx)
                distorted = apply_augmentation_to_batch(recon_wm, aug_name, downsample_rate)
                detect_feat_d = backbone.model.st_model.forward_feature(distorted)
                _, chunk_logits_d = backbone.model.detector(detect_feat_d)
                results[aug_name].append(compute_detection_accuracy(chunk_logits_d, message))

    return {name: sum(vals) / len(vals) for name, vals in results.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None,
                    help="Path to a train.py Stage 1 checkpoint. If omitted, evaluates baseline (LoRA zero-init).")
    p.add_argument("--output", type=str, required=True, help="Where to save results as JSON.")
    p.add_argument("--data_root", type=str, default="./data/librispeech")
    p.add_argument("--n_speakers", type=int, default=30)
    p.add_argument("--utterances_per_speaker", type=int, default=10)
    p.add_argument("--n_eval_speakers", type=int, default=5)
    p.add_argument("--eval_utterances_per_speaker", type=int, default=5)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--crop_seconds", type=float, default=3.0)
    p.add_argument("--batch_size", type=int, default=4)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    eval_ds = LibriSpeechSubset(
        root=args.data_root, n_speakers=args.n_speakers,
        utterances_per_speaker=args.utterances_per_speaker,
        n_eval_speakers=args.n_eval_speakers,
        eval_utterances_per_speaker=args.eval_utterances_per_speaker,
        sample_rate=16000, crop_seconds=args.crop_seconds, split="eval",
    )
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_librispeech, drop_last=False)

    aug_names = list(AUGMENTATION_FNS.keys())

    label = "baseline" if args.checkpoint is None else args.checkpoint
    print(f"\n{'=' * 60}\nEvaluating: {label}\n{'=' * 60}")
    backbone = build_backbone(lora_checkpoint_path=args.checkpoint, r=args.lora_r, alpha=args.lora_alpha)
    results = run_robustness_eval(backbone, eval_loader, device, aug_names)

    print(f"\n{'Condition':<15} {'ACC':>10}")
    for name in ["clean"] + aug_names:
        print(f"{name:<15} {results[name]:>10.4f}")

    with open(args.output, "w") as f:
        json.dump({"label": label, "checkpoint": args.checkpoint, "results": results}, f, indent=2)
    print(f"\n[main] Saved results to {args.output}")


if __name__ == "__main__":
    main()
