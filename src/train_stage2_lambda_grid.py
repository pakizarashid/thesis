"""
src/train_stage2_lambda_grid.py

Small lambda grid search: loads backbone + adapters + discriminator +
surrogate + dataset ONCE, then for each candidate lambda_disrupt_max, resets
the LoRA weights back to the Stage 1 checkpoint (undoing any drift from the
previous candidate), runs a short training burst, and records:
    - avg detection ACC over the burst (should stay high -- traceability
      shouldn't collapse)
    - delta_pivotal = last-step pivotal_disruption - first-step pivotal_disruption
      (want this POSITIVE and as large as possible -- it means the cloned
      output is becoming MORE dissimilar from the original speaker over
      training, i.e. disruption is actually working, not just present in the
      loss function)

This is a cheap PROXY for real disruption effectiveness (mel-distance trend),
not a substitute for the full WER/SIM evaluation against the eval set that
should follow once a good lambda is chosen -- see the project plan's Week 7
milestone for that fuller evaluation.

Usage:
    python src/train_stage2_lambda_grid.py \
      --stage1_checkpoint ./checkpoints/stage1_full/stage1_epoch29.pt \
      --n_speakers 3 --utterances_per_speaker 3 \
      --n_eval_speakers 2 --eval_utterances_per_speaker 2 \
      --steps_per_candidate 15 \
      --lambda_candidates 0.0001 0.0003 0.001 0.003 0.01
"""

import os
import sys
import copy
import argparse
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "losses"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))

from backbone import VoiceMarkBackbone, VoiceMarkDiscriminator
from adapters import apply_lora_adapters
from surrogate_vc import load_yourtts_surrogate
from voicemark_losses import VoiceMarkLosses
from safespeech_losses import SafeSpeechDisruptionLoss
from librispeech import LibriSpeechSubset, collate_librispeech

from train_stage2 import (
    random_message, compute_detection_accuracy, load_stage1_checkpoint, lambda_schedule,
)


def run_one_candidate(backbone, discriminator, surrogate, voicemark_losses_fn,
                       disruption_loss_fn, train_loader, device, args, lambda_max):
    """Resets LoRA weights to the Stage 1 checkpoint, runs a short training
    burst at this lambda, returns summary metrics."""
    load_stage1_checkpoint(backbone, args.stage1_checkpoint)
    # NOTE: st_model contains cuDNN-backed LSTM layers (SLSTM), which refuse to
    # run backward() in eval() mode (a hard cuDNN constraint, not a choice) --
    # so we must stay in train() mode. To still get comparable dropout
    # behavior across candidates, seed torch's global RNG identically instead;
    # this makes dropout's random masks identical across candidates even
    # though dropout remains functionally active.
    backbone.model.train()
    torch.manual_seed(args.data_seed)

    trainable_params = [p for p in backbone.model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    accs = []
    pivotal_values = []
    step = 0
    # Fixed seed so every candidate sees the IDENTICAL sequence of batches --
    # otherwise differences in delta_pivotal could just reflect different
    # random utterance orderings rather than a real lambda effect.
    seeded_generator = torch.Generator().manual_seed(args.data_seed)
    data_iter = iter(DataLoader(
        train_loader.dataset, batch_size=train_loader.batch_size, shuffle=True,
        collate_fn=collate_librispeech, drop_last=True, generator=seeded_generator,
    ))

    while step < args.steps_per_candidate:
        try:
            batch = next(data_iter)
        except StopIteration:
            # Re-seed with a derived-but-still-deterministic seed for the next
            # pass through the data, so wraparounds are also reproducible
            # across candidates (not falling back to unseeded shuffling).
            wraparound_generator = torch.Generator().manual_seed(args.data_seed + step)
            data_iter = iter(DataLoader(
                train_loader.dataset, batch_size=train_loader.batch_size, shuffle=True,
                collate_fn=collate_librispeech, drop_last=True, generator=wraparound_generator,
            ))
            batch = next(data_iter)

        clean_audio = batch["waveform"].to(device)
        message = random_message(16, clean_audio.shape[0], device)
        current_lambda = lambda_schedule(step, args.ramp_steps, lambda_max)

        out = backbone.forward_full(clean_audio, message)
        recon_wm = out["recon_wm"]
        detect_feat = backbone.model.st_model.forward_feature(recon_wm)
        logits, chunk_logits = backbone.model.detector(detect_feat)

        vm_loss_dict = voicemark_losses_fn(
            recon_wm=recon_wm, clean_audio=clean_audio,
            acoustic=out["acoustic"], acoustic_wm=out["acoustic_wm"],
            detection_logits=logits, chunk_logits=chunk_logits,
            message=message, discriminator=discriminator,
        )

        cloned_output = surrogate.clone_voice(recon_wm, text=args.surrogate_text)
        disrupt_loss_dict = disruption_loss_fn(clean_audio, cloned_output)

        total_loss = vm_loss_dict["total"] + current_lambda * disrupt_loss_dict["total"]

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        accs.append(compute_detection_accuracy(chunk_logits.detach(), message))
        pivotal_values.append(disrupt_loss_dict["pivotal_disruption"].item())
        step += 1

    avg_acc = sum(accs) / len(accs)
    delta_pivotal = pivotal_values[-1] - pivotal_values[0]
    return {
        "lambda_max": lambda_max,
        "avg_acc": avg_acc,
        "final_acc": accs[-1],
        "pivotal_first": pivotal_values[0],
        "pivotal_last": pivotal_values[-1],
        "delta_pivotal": delta_pivotal,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_checkpoint", type=str, required=True)
    p.add_argument("--lambda_candidates", type=float, nargs="+",
                    default=[0.0001, 0.0003, 0.001, 0.003, 0.01])
    p.add_argument("--steps_per_candidate", type=int, default=15)
    p.add_argument("--data_seed", type=int, default=42,
                    help="Fixed seed for data ordering, identical across all lambda candidates, "
                         "so delta_pivotal differences reflect lambda's effect, not random data order.")
    p.add_argument("--ramp_steps", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--data_root", type=str, default="./data/librispeech")
    p.add_argument("--n_speakers", type=int, default=3)
    p.add_argument("--utterances_per_speaker", type=int, default=3)
    p.add_argument("--n_eval_speakers", type=int, default=2)
    p.add_argument("--eval_utterances_per_speaker", type=int, default=2)
    p.add_argument("--crop_seconds", type=float, default=3.0)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--surrogate_sample_rate", type=int, default=16000)
    p.add_argument("--surrogate_text", type=str, default="This is a test sentence for voice cloning.")
    p.add_argument("--no_discriminator", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[grid] Using device: {device}")

    print("[grid] Loading backbone + adapters + discriminator (once, reused across candidates)...")
    backbone = VoiceMarkBackbone()
    apply_lora_adapters(backbone, r=args.lora_r, alpha=args.lora_alpha)
    discriminator = VoiceMarkDiscriminator() if not args.no_discriminator else None

    print("[grid] Loading YourTTS surrogate (once, frozen, reused across candidates)...")
    surrogate = load_yourtts_surrogate(device=device)

    print("[grid] Loading dataset (once, reused across candidates)...")
    train_ds = LibriSpeechSubset(
        root=args.data_root, n_speakers=args.n_speakers,
        utterances_per_speaker=args.utterances_per_speaker,
        n_eval_speakers=args.n_eval_speakers,
        eval_utterances_per_speaker=args.eval_utterances_per_speaker,
        sample_rate=16000, crop_seconds=args.crop_seconds, split="train",
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_librispeech, drop_last=True)

    voicemark_losses_fn = VoiceMarkLosses(sample_rate=16000).to(device)
    disruption_loss_fn = SafeSpeechDisruptionLoss(sampling_rate=args.surrogate_sample_rate).to(device)

    results = []
    for lambda_max in args.lambda_candidates:
        print(f"\n{'=' * 60}\nCandidate: lambda_disrupt_max = {lambda_max}\n{'=' * 60}")
        result = run_one_candidate(
            backbone, discriminator, surrogate, voicemark_losses_fn,
            disruption_loss_fn, train_loader, device, args, lambda_max,
        )
        print(f"  avg_acc={result['avg_acc']:.4f}  final_acc={result['final_acc']:.4f}  "
              f"pivotal: {result['pivotal_first']:.4f} -> {result['pivotal_last']:.4f}  "
              f"delta={result['delta_pivotal']:+.4f}")
        results.append(result)

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    print(f"{'lambda':>10} {'avg_acc':>10} {'final_acc':>10} {'delta_pivotal':>15}")
    for r in results:
        print(f"{r['lambda_max']:>10} {r['avg_acc']:>10.4f} {r['final_acc']:>10.4f} {r['delta_pivotal']:>+15.4f}")

    # Simple selection heuristic: highest delta_pivotal among candidates that
    # keep avg_acc above 0.90 -- adjust this threshold based on how much
    # traceability degradation you're willing to accept for more disruption.
    viable = [r for r in results if r["avg_acc"] >= 0.90]
    if viable:
        best = max(viable, key=lambda r: r["delta_pivotal"])
        print(f"\nSuggested starting point (highest delta_pivotal with avg_acc >= 0.90): "
              f"lambda_disrupt_max = {best['lambda_max']}")
    else:
        print("\nWARNING - no candidate kept avg_acc >= 0.90. All candidates degraded "
              "detection meaningfully in just this short burst -- consider smaller "
              "lambda values or a longer ramp_steps schedule.")


if __name__ == "__main__":
    main()
