"""
src/eval/disruption_effectiveness.py

Measures the actual disruption effectiveness metric both VoiceMark and
SafeSpeech report as their headline evidence: speaker similarity (SIM) between
the original speaker and the surrogate's zero-shot clone of the watermarked
audio. LOWER SIM = more effective disruption (the cloned voice sounds less
like the original speaker).

Also reports pivotal mel-distance (the raw training signal) for reference, but
SIM is the metric that actually matters for the thesis's claims -- it's
directly comparable to both papers' own reported numbers.

One model per process (same safe pattern as augmentation_robustness.py -- an
earlier attempt to load two backbones sequentially in one process corrupted
GPU state and produced degenerate results for whichever loaded second).
Outputs the same JSON schema as augmentation_robustness.py, so your existing
compare_results.py works unmodified.

Usage (run each in a separate process/kernel restart):
    python src/eval/disruption_effectiveness.py --output results_baseline_sim.json
    python src/eval/disruption_effectiveness.py --checkpoint ./checkpoints/stage1_full/stage1_epoch29.pt --output results_stage1_sim.json
    python src/eval/disruption_effectiveness.py --checkpoint ./checkpoints/stage2_real/stage2_epoch4.pt --output results_stage2_sim.json

    python src/eval/compare_results.py results_baseline_sim.json results_stage1_sim.json results_stage2_sim.json
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
            print(f"[build_backbone] WARNING - {len(non_lora_missing)} LoRA keys missing: {non_lora_missing[:3]}...")
        print(f"[build_backbone] Loaded LoRA weights from {lora_checkpoint_path} "
              f"(epoch {ckpt.get('epoch')}, train-time avg_acc={ckpt.get('avg_acc'):.4f})")
    else:
        print("[build_backbone] Using baseline (LoRA at zero-init, == pretrained VoiceMark)")

    return backbone


def random_message(nbits: int, batch_size: int, device, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.randint(0, 2, (batch_size, nbits), generator=gen, device=device)


def compute_sim(surrogate, clean_audio: torch.Tensor, cloned_output: torch.Tensor) -> float:
    """
    Speaker similarity: cosine similarity between the surrogate's own speaker
    embedding of the ORIGINAL clean audio and of the CLONED output. This is
    the same speaker encoder used to condition cloning in the first place, so
    it directly measures whether the clone actually sounds like the original
    speaker from the surrogate's own perspective -- matching how VoiceMark/
    SafeSpeech report their SIM metric.
    """
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
                    help="Path to a train.py/train_stage2.py checkpoint. If omitted, evaluates baseline (LoRA zero-init).")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--data_root", type=str, default="./data/librispeech")
    p.add_argument("--n_speakers", type=int, default=10)
    p.add_argument("--utterances_per_speaker", type=int, default=10)
    p.add_argument("--n_eval_speakers", type=int, default=5)
    p.add_argument("--eval_utterances_per_speaker", type=int, default=5)
    p.add_argument("--crop_seconds", type=float, default=3.0)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--surrogate_sample_rate", type=int, default=16000)
    p.add_argument("--surrogate_text", type=str, default="This is a test sentence for voice cloning.")
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
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_librispeech, drop_last=False)

    label = "baseline" if args.checkpoint is None else args.checkpoint
    print(f"\n{'=' * 60}\nEvaluating: {label}\n{'=' * 60}")

    backbone = build_backbone(lora_checkpoint_path=args.checkpoint, r=args.lora_r, alpha=args.lora_alpha)
    print("[main] Loading YourTTS surrogate (frozen)...")
    surrogate = load_yourtts_surrogate(device=device)
    mel_fn = SafeSpeechMelSpectrogram(sampling_rate=args.surrogate_sample_rate).to(device)

    results = run_disruption_eval(backbone, surrogate, eval_loader, device, mel_fn, args.surrogate_text)

    print(f"\nSIM (speaker similarity, LOWER = more disrupted): {results['sim_mean']:.4f}")
    print(f"Pivotal mel distance (HIGHER = more disrupted): {results['pivotal_distance_mean']:.4f}")

    with open(args.output, "w") as f:
        json.dump({
            "label": label, "checkpoint": args.checkpoint,
            "results": {"sim": results["sim_mean"], "pivotal_distance": results["pivotal_distance_mean"]},
        }, f, indent=2)
    print(f"[main] Saved results to {args.output}")


if __name__ == "__main__":
    main()
