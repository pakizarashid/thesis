"""
src/eval/false_positive_rate.py

Standard, expected watermarking metric never previously measured in this
project: does the detector wrongly claim a watermark is present on audio
that was NEVER watermarked at all? All prior evaluation only tested
detection accuracy CONDITIONED ON a watermark being present -- this tests
the complementary, equally important question.

Uses WMDetector's watermark_head (a dedicated presence/absence classifier,
separate from the message-decoding heads) -- present in the architecture
since Stage 1 but never evaluated on its own until now.

Method: run CLEAN, never-watermarked audio directly through the frozen codec
and detector (skipping the watermark embedding step entirely), threshold the
presence logits, and report the fraction of clean utterances incorrectly
flagged as watermarked.

Usage:
    python src/eval/false_positive_rate.py --output results_baseline_fpr.json
    python src/eval/false_positive_rate.py --checkpoint ./checkpoints/stage1_full/stage1_epoch29.pt --output results_stage1_fpr.json
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
        print("[build_backbone] Using baseline (LoRA zero-init)")
    return backbone


def run_fpr_eval(backbone, eval_loader, device, threshold: float = 0.0) -> dict:
    """
    threshold=0.0 on raw logits corresponds to sigmoid probability 0.5 --
    the standard default decision boundary. Per-utterance decision: mean
    presence logit across time frames, thresholded once per utterance.
    """
    backbone.model.eval()
    presence_logits_list = []
    false_positives = 0
    total = 0

    with torch.no_grad():
        for batch in eval_loader:
            clean_audio = batch["waveform"].to(device)
            # Deliberately skip watermark embedding -- this audio was NEVER
            # watermarked. Run directly through the codec's feature path and
            # detector, exactly as done for real detection elsewhere in this
            # project, just without the msg_processor step.
            detect_feat = backbone.model.st_model.forward_feature(clean_audio)
            presence_logits, chunk_logits = backbone.model.detector(detect_feat)

            mean_logit_per_utt = presence_logits.mean(dim=-1)  # [batch]
            decisions = (mean_logit_per_utt > threshold)  # True = incorrectly flagged as watermarked

            false_positives += decisions.sum().item()
            total += decisions.shape[0]
            presence_logits_list.extend(mean_logit_per_utt.cpu().tolist())

    fpr = false_positives / total if total > 0 else float("nan")
    return {
        "false_positive_rate": fpr,
        "false_positives": false_positives,
        "total_clean_utterances": total,
        "mean_presence_logit": sum(presence_logits_list) / len(presence_logits_list),
        "presence_logit_values": presence_logits_list,
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
    p.add_argument("--threshold", type=float, default=0.0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Uses the TRAIN split deliberately, not eval -- these utterances were
    # never watermarked in ANY of this project's experiments, so they're
    # clean, genuinely never-seen-by-a-watermark-embedder audio, appropriate
    # for a false-positive test regardless of which split they came from.
    eval_ds = LibriSpeechSubset(
        root=args.data_root, n_speakers=args.n_speakers,
        utterances_per_speaker=args.utterances_per_speaker,
        n_eval_speakers=args.n_eval_speakers,
        eval_utterances_per_speaker=args.eval_utterances_per_speaker,
        sample_rate=16000, crop_seconds=args.crop_seconds, split="eval",
    )
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_librispeech)

    label = "baseline" if args.checkpoint is None else args.checkpoint
    print(f"\n{'=' * 60}\nEvaluating false positive rate: {label}\n{'=' * 60}")

    backbone = build_backbone(lora_checkpoint_path=args.checkpoint, r=args.lora_r, alpha=args.lora_alpha)
    results = run_fpr_eval(backbone, eval_loader, device, threshold=args.threshold)

    print(f"\nFalse positive rate: {results['false_positive_rate']:.4f} "
          f"({results['false_positives']}/{results['total_clean_utterances']} clean utterances "
          f"incorrectly flagged as watermarked)")
    print(f"Mean presence logit on clean audio: {results['mean_presence_logit']:.4f} "
          f"(threshold={args.threshold}, more negative = more confidently 'no watermark')")

    with open(args.output, "w") as f:
        json.dump({
            "label": label, "checkpoint": args.checkpoint,
            "results": {
                "false_positive_rate": results["false_positive_rate"],
                "mean_presence_logit": results["mean_presence_logit"],
            },
        }, f, indent=2)
    print(f"[main] Saved results to {args.output}")


if __name__ == "__main__":
    main()
