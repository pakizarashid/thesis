"""
src/eval/disruption_effectiveness_capacity.py

Capacity-aware variant of disruption_effectiveness.py. The plain script's
build_backbone() calls apply_lora_adapters(backbone, r=r, alpha=alpha) with no
include_ffn/ffn_r -- it always builds the canonical attention-only, rank-8
backbone regardless of what checkpoint you point it at. Pointing it at a
stage2_capacity_ffn checkpoint (trained with include_ffn=True, ffn_r=32 on
msg_processor, per train_stage2_capacity.py) crashes with a shape-mismatch
RuntimeError -- the checkpoint's tensors are rank-32, the freshly-built
backbone's are rank-8, and strict=False only tolerates missing/extra KEYS, not
shape mismatches on matching keys. Same class of bug gradient_diagnostic.py
had (fixed by gradient_diagnostic_capacity.py) -- this is the eval-side
equivalent, needed for the SAME reason: measuring SIM on the capacity
checkpoint you just trained.

Only two things differ from disruption_effectiveness.py: build_backbone()
takes capacity_lora_r and builds the exact architecture
train_stage2_capacity.py trains, and a --capacity_lora_r CLI flag was added.
Everything else (SIM computation, pivotal distance, output schema, dataset
handling) is unchanged and produces the identical JSON shape, so it still
drops straight into aggregate_results.py.

Usage:
    python src/eval/disruption_effectiveness_capacity.py \\
        --checkpoint ./checkpoints/stage2_capacity_ffn/stage2_capacity_epoch29.pt \\
        --capacity_lora_r 32 --output results/results_capacity_ffn_sim_run1.json
"""

import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "losses"))

from backbone import VoiceMarkBackbone
from adapters import apply_lora_adapters
from surrogate_vc import load_yourtts_surrogate
from safespeech_losses import SafeSpeechMelSpectrogram, compute_pivotal_disruption_loss
from librispeech import LibriSpeechSubset, collate_librispeech
from vctk import VCTKSubset, collate_vctk
from libritts import LibriTTSSubset, collate_libritts
from torch.utils.data import DataLoader


def build_backbone(lora_checkpoint_path: str, r: int, alpha: int, capacity_lora_r: int):
    backbone = VoiceMarkBackbone()
    # Exactly train_stage2_capacity.py's apply_lora_adapters call -- msg_processor
    # gets capacity_lora_r on BOTH its attention and its (new) feedforward LoRA;
    # detector keeps the canonical r (irrelevant to recon_wm / SIM either way).
    apply_lora_adapters(
        backbone, r=r, alpha=alpha, targets=("msg_processor", "detector"),
        include_ffn=True, ffn_r=capacity_lora_r, ffn_targets=("msg_processor",),
    )

    if lora_checkpoint_path is not None:
        ckpt = torch.load(lora_checkpoint_path, map_location="cpu", weights_only=False)
        lora_state = ckpt["lora_state_dict"]
        missing, unexpected = backbone.model.load_state_dict(lora_state, strict=False)
        non_lora_missing = [k for k in missing if "_lora." in k]
        if non_lora_missing:
            print(f"[build_backbone] WARNING - {len(non_lora_missing)} LoRA keys missing: {non_lora_missing[:3]}... "
                  f"(if this checkpoint was trained with a DIFFERENT --capacity_lora_r than {capacity_lora_r}, "
                  f"you'll get a shape-mismatch crash instead of a clean load -- pass the matching value.)")
        avg_acc = ckpt.get("avg_acc")
        avg_acc_str = f"{avg_acc:.4f}" if avg_acc is not None else "N/A"
        print(f"[build_backbone] Loaded LoRA weights from {lora_checkpoint_path} "
              f"(epoch {ckpt.get('epoch')}, train-time avg_acc={avg_acc_str}, capacity_lora_r={capacity_lora_r})")
    else:
        print("[build_backbone] No checkpoint given -- baseline zero-init at this (expanded) architecture. "
              "NOTE: for a true baseline comparison, use plain disruption_effectiveness.py --checkpoint None "
              "instead (canonical r=8 backbone) -- zero-init behaves identically to the canonical baseline "
              "since LoRA's B matrix starts at zero regardless of rank, but there's no reason to build the "
              "larger architecture just to evaluate a no-op.")

    return backbone


def random_message(nbits: int, batch_size: int, device, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.randint(0, 2, (batch_size, nbits), generator=gen, device=device)


def compute_sim(surrogate, clean_audio: torch.Tensor, cloned_output: torch.Tensor) -> float:
    with torch.no_grad():
        emb_clean = surrogate.compute_speaker_embedding(clean_audio)
        emb_cloned = surrogate.compute_speaker_embedding(cloned_output)
        sim = F.cosine_similarity(emb_clean, emb_cloned, dim=-1)
    return sim.mean().item()


def run_disruption_eval(backbone, surrogate, eval_loader, device, mel_fn,
                         text: str, seed: int = 123) -> dict:
    backbone.model.eval()
    sims = []
    pivotal_distances = []

    for batch_idx, batch in enumerate(eval_loader):
        clean_audio = batch["waveform"].to(device)
        message = random_message(16, clean_audio.shape[0], device, seed=seed + batch_idx)

        with torch.no_grad():
            out = backbone.forward_full(clean_audio, message)
            recon_wm = out["recon_wm"]
            cloned_output = surrogate.clone_voice(recon_wm, text=text)

            sim = compute_sim(surrogate, clean_audio, cloned_output)
            pivotal = compute_pivotal_disruption_loss(mel_fn, clean_audio, cloned_output).item()

        sims.append(sim)
        pivotal_distances.append(pivotal)

    return {
        "sim_mean": sum(sims) / len(sims),
        "sim_values": sims,
        "pivotal_distance_mean": sum(pivotal_distances) / len(pivotal_distances),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None,
                    help="Path to a stage2_capacity_ffn checkpoint. Required in practice -- see the "
                         "note in build_backbone() about baseline comparisons.")
    p.add_argument("--capacity_lora_r", type=int, default=32,
                    help="MUST match the --capacity_lora_r the checkpoint was trained with "
                         "(train_stage2_capacity.py), or loading will crash on a shape mismatch.")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--dataset", type=str, default="librispeech", choices=["librispeech", "vctk", "libritts"])
    p.add_argument("--vctk_root", type=str,
                    default="/kaggle/input/datasets/pratt3000/vctk-corpus/VCTK-Corpus/VCTK-Corpus")
    p.add_argument("--libritts_root", type=str,
                    default="/kaggle/input/datasets/pratt3000/libritts/LibriTTS")
    p.add_argument("--data_root", type=str, default="./data/librispeech")
    p.add_argument("--n_speakers", type=int, default=10)
    p.add_argument("--utterances_per_speaker", type=int, default=10)
    p.add_argument("--n_eval_speakers", type=int, default=5)
    p.add_argument("--eval_utterances_per_speaker", type=int, default=5)
    p.add_argument("--crop_seconds", type=float, default=3.0)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--surrogate_sample_rate", type=int, default=16000)
    p.add_argument("--surrogate_text", type=str, default="This is a test sentence for voice cloning.")
    p.add_argument("--lora_r", type=int, default=8, help="detector's rank -- unchanged from canonical.")
    p.add_argument("--lora_alpha", type=int, default=16)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset == "librispeech":
        eval_ds = LibriSpeechSubset(
            root=args.data_root, n_speakers=args.n_speakers,
            utterances_per_speaker=args.utterances_per_speaker,
            n_eval_speakers=args.n_eval_speakers,
            eval_utterances_per_speaker=args.eval_utterances_per_speaker,
            sample_rate=16000, crop_seconds=args.crop_seconds, split="eval",
        )
        collate_fn = collate_librispeech
    elif args.dataset == "vctk":
        eval_ds = VCTKSubset(
            root=args.vctk_root, n_speakers=args.n_speakers,
            utterances_per_speaker=args.utterances_per_speaker,
            n_eval_speakers=args.n_eval_speakers,
            eval_utterances_per_speaker=args.eval_utterances_per_speaker,
            sample_rate=16000, crop_seconds=args.crop_seconds, split="eval",
        )
        collate_fn = collate_vctk
    else:
        eval_ds = LibriTTSSubset(
            root=args.libritts_root, n_speakers=args.n_speakers,
            utterances_per_speaker=args.utterances_per_speaker,
            n_eval_speakers=args.n_eval_speakers,
            eval_utterances_per_speaker=args.eval_utterances_per_speaker,
            sample_rate=16000, crop_seconds=args.crop_seconds, split="eval",
        )
        collate_fn = collate_libritts

    print(f"[main] Evaluating on dataset: {args.dataset}")
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, drop_last=False)

    label = "baseline_capacity_arch" if args.checkpoint is None else args.checkpoint
    print(f"\n{'=' * 60}\nEvaluating: {label} (capacity_lora_r={args.capacity_lora_r})\n{'=' * 60}")

    backbone = build_backbone(lora_checkpoint_path=args.checkpoint, r=args.lora_r,
                               alpha=args.lora_alpha, capacity_lora_r=args.capacity_lora_r)
    print("[main] Loading YourTTS surrogate (frozen)...")
    surrogate = load_yourtts_surrogate(device=device)
    mel_fn = SafeSpeechMelSpectrogram(sampling_rate=args.surrogate_sample_rate).to(device)

    results = run_disruption_eval(backbone, surrogate, eval_loader, device, mel_fn, args.surrogate_text)

    print(f"\nSIM (speaker similarity, LOWER = more disrupted): {results['sim_mean']:.4f}")
    print(f"Pivotal mel distance (HIGHER = more disrupted): {results['pivotal_distance_mean']:.4f}")

    with open(args.output, "w") as f:
        json.dump({
            "label": label, "checkpoint": args.checkpoint, "dataset": args.dataset,
            "capacity_lora_r": args.capacity_lora_r,
            "results": {"sim": results["sim_mean"], "pivotal_distance": results["pivotal_distance_mean"]},
        }, f, indent=2)
    print(f"[main] Saved results to {args.output}")


if __name__ == "__main__":
    main()
