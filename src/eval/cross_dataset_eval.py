"""
src/eval/cross_dataset_eval.py

Tests whether checkpoints trained ONLY on LibriSpeech generalize to VCTK --
audio from a completely different corpus (different speakers, different
recording conditions, and notably VoiceMark's OWN original training domain,
making this also a meaningful comparison point to their paper). No training
happens here -- same checkpoints, same weights, evaluation only, since the
whole point is testing generalization, not creating a VCTK-adapted model.

Usage:
    python src/eval/cross_dataset_eval.py --output results/vctk_baseline.json
    python src/eval/cross_dataset_eval.py --checkpoint ./checkpoints/stage1_full_recalibrated_v3/recalibrated_final.pt --output results/vctk_stage1_v3.json
"""

import os
import sys
import json
import argparse
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

from backbone import VoiceMarkBackbone
from adapters import apply_lora_adapters
from vctk import VCTKSubset, collate_vctk


def build_backbone(lora_checkpoint_path: str = None, r: int = 8, alpha: int = 16):
    backbone = VoiceMarkBackbone()
    apply_lora_adapters(backbone, r=r, alpha=alpha)
    if lora_checkpoint_path is not None:
        ckpt = torch.load(lora_checkpoint_path, map_location="cpu", weights_only=False)
        backbone.model.load_state_dict(ckpt["lora_state_dict"], strict=False)
        print(f"[build_backbone] Loaded checkpoint {lora_checkpoint_path} "
              f"(trained on LibriSpeech ONLY -- this eval tests generalization to VCTK)")
    else:
        print("[build_backbone] Using baseline (LoRA zero-init)")
    return backbone


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--vctk_root", type=str,
                    default="/kaggle/input/datasets/pratt3000/vctk-corpus/VCTK-Corpus/VCTK-Corpus")
    p.add_argument("--n_speakers", type=int, default=10)
    p.add_argument("--utterances_per_speaker", type=int, default=10)
    p.add_argument("--n_eval_speakers", type=int, default=5)
    p.add_argument("--eval_utterances_per_speaker", type=int, default=5)
    p.add_argument("--crop_seconds", type=float, default=3.0)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    eval_ds = VCTKSubset(
        root=args.vctk_root, n_speakers=args.n_speakers,
        utterances_per_speaker=args.utterances_per_speaker,
        n_eval_speakers=args.n_eval_speakers,
        eval_utterances_per_speaker=args.eval_utterances_per_speaker,
        sample_rate=16000, crop_seconds=args.crop_seconds, split="eval",
    )
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_vctk)

    label = "baseline" if args.checkpoint is None else args.checkpoint
    print(f"\n{'=' * 60}\nCross-dataset (VCTK) evaluation: {label}\n{'=' * 60}")

    backbone = build_backbone(lora_checkpoint_path=args.checkpoint, r=args.lora_r, alpha=args.lora_alpha)
    backbone.model.eval()

    accs = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            clean_audio = batch["waveform"].to(device)
            gen = torch.Generator(device=device).manual_seed(args.seed + batch_idx)
            message = torch.randint(0, 2, (clean_audio.shape[0], 16), generator=gen, device=device)

            out = backbone.forward_full(clean_audio, message)
            recon_wm = out["recon_wm"]
            detect_feat = backbone.model.st_model.forward_feature(recon_wm)
            _, chunk_logits = backbone.model.detector(detect_feat)

            acc = compute_detection_accuracy(chunk_logits, message)
            accs.append(acc)
            print(f"  batch {batch_idx}: acc={acc:.4f} speakers={batch['speaker_id']}")

    mean_acc = sum(accs) / len(accs)
    print(f"\nVCTK cross-dataset detection accuracy: {mean_acc:.4f}")
    print(f"(Compare against this checkpoint's LibriSpeech held-out accuracy to assess generalization)")

    with open(args.output, "w") as f:
        json.dump({
            "label": label, "checkpoint": args.checkpoint,
            "results": {"vctk_detection_acc": mean_acc, "batch_accs": accs},
        }, f, indent=2)
    print(f"[main] Saved results to {args.output}")


if __name__ == "__main__":
    main()
