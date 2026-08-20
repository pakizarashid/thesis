"""
src/train_stage2_capacity.py

Stage 2 variant built to directly test the leading, previously-untested
explanation from STAGE2_WRITEUP.md Section 7/9/10: that five converging
negative results on the disruption objective reflect a genuine CAPACITY
limitation of the attention-only LoRA adapters (294,912 params), not a
tuning artifact.

WHAT'S DIFFERENT FROM train_stage2.py:
  1. Uses apply_lora_adapters(..., include_ffn=True, ffn_targets=("msg_processor",))
     -- adds LoRA to msg_processor's transformer feedforward Linear layers
     (linear1/linear2 inside each TransformerDecoderLayer), which the
     original attention-only wrapping never touched at all. In a standard
     PyTorch transformer decoder layer the FFN block typically has MORE
     parameters than the attention projections, so this is a substantial,
     previously-untested capacity increase specifically on the ONE submodule
     that actually produces recon_wm (msg_processor) -- detector's FFN is
     deliberately left untouched, since detector plays no role in the
     disruption loss's gradient path (it only reads recon_wm after the fact
     for ACC/FPR, it never shapes it).
  2. --capacity_lora_r (default 32, i.e. 4x the original r=8) controls the
     rank used for msg_processor specifically. detector keeps the original
     --lora_r (default 8) so its param count -- and therefore its
     traceability behavior -- stays comparable to the canonical checkpoints,
     isolating capacity as the ONLY variable changed for this experiment.
  3. Stage 1 checkpoint loading is now PARTIAL by necessity: a Stage 1
     checkpoint was trained with attention-only LoRA, so it has no weights
     for the new FFN LoRA modules. load_stage1_checkpoint_partial() below
     loads whatever DOES match (attention deltas) and leaves the new FFN
     deltas at their zero-init (mathematically a no-op at start, same
     property the original LoRA init already relies on) rather than
     rejecting the whole checkpoint on a shape/key mismatch like the
     original stricter loader does.

Everything else (loss composition, lambda ramp, disrupt_mode='sim' default,
logging, checkpoint format) is unchanged from train_stage2.py -- this script
isolates ONE variable (capacity on the recon_wm-producing submodule) rather
than re-testing lambda/weighting/objective again, which STAGE2_WRITEUP.md
Section 9 already covers.

RUN THIS BEFORE COMMITTING GPU HOURS: this file was written and syntax-
checked but NOT run against your actual data/GPU pipeline (that requires
your Kaggle environment). Do a short --epochs 1 --n_speakers 5 smoke test
first, and re-run gradient_diagnostic.py-style checks on the new FFN
parameters specifically before trusting a long run, exactly as
STAGE2_WRITEUP.md Section 4/7 did for every previous lever.

Usage:
    python src/train_stage2_capacity.py \\
        --stage1_checkpoint ./checkpoints/stage1_full_recalibrated_v3/recalibrated_final.pt \\
        --capacity_lora_r 32 --epochs 30 --disrupt_mode sim \\
        --checkpoint_dir ./checkpoints/stage2_capacity_ffn
"""

import os
import sys
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
from safespeech_losses import SafeSpeechDisruptionLoss, compute_sim_disruption_loss
from librispeech import LibriSpeechSubset, collate_librispeech

# Reuse everything not specific to capacity from train_stage2.py rather than
# forking logic that would drift; import them directly.
from train_stage2 import (
    random_message, compute_detection_accuracy, lambda_schedule,
)


def load_stage1_checkpoint_partial(backbone, checkpoint_path: str = None):
    """
    Like train_stage2.load_stage1_checkpoint, but tolerant of the new FFN
    LoRA keys having no counterpart in a Stage 1 checkpoint trained before
    this script existed. Loads only keys that exist AND shape-match; leaves
    everything else (the new FFN deltas) at zero-init.
    """
    if checkpoint_path is None:
        print("[load_stage1_checkpoint_partial] No checkpoint provided -- "
              "starting entirely from LoRA zero-init.")
        return

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    lora_state = ckpt["lora_state_dict"]
    current_sd = backbone.model.state_dict()

    to_load = {}
    skipped_shape = []
    skipped_missing = []
    for k, v in lora_state.items():
        if k not in current_sd:
            skipped_missing.append(k)
        elif current_sd[k].shape != v.shape:
            skipped_shape.append(k)
        else:
            to_load[k] = v

    missing, unexpected = backbone.model.load_state_dict(to_load, strict=False)
    print(f"[load_stage1_checkpoint_partial] Loaded {len(to_load)}/{len(lora_state)} "
          f"tensors from {checkpoint_path} (epoch {ckpt.get('epoch')}). "
          f"{len(skipped_shape)} shape-mismatched, {len(skipped_missing)} not present "
          f"in current model (expected for detector's attention if --lora_r differs, "
          f"and for any FFN keys if this is an older attention-only checkpoint -- "
          f"those stay at zero-init, a mathematical no-op at construction).")


def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_checkpoint", type=str, default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lora_r", type=int, default=8,
                    help="Rank for detector (kept at the original/canonical value -- "
                         "detector's capacity is irrelevant to the disruption loss).")
    p.add_argument("--capacity_lora_r", type=int, default=32,
                    help="Rank for msg_processor (attention AND the new FFN LoRA modules). "
                         "Default 4x the canonical r=8, matching the scale already tried "
                         "(and found insufficient) for attention-only capacity in "
                         "STAGE2_WRITEUP.md Section 5 -- the point of this script is "
                         "testing whether FFN capacity specifically, not just more "
                         "attention rank, makes the difference.")
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--data_root", type=str, default="./data/librispeech")
    p.add_argument("--n_speakers", type=int, default=30)
    p.add_argument("--utterances_per_speaker", type=int, default=10)
    p.add_argument("--n_eval_speakers", type=int, default=5)
    p.add_argument("--eval_utterances_per_speaker", type=int, default=5)
    p.add_argument("--crop_seconds", type=float, default=3.0)
    p.add_argument("--lambda_vad", type=float, default=1.0)
    p.add_argument("--lambda_cos", type=float, default=2.0)
    p.add_argument("--lambda_mel", type=float, default=2.0)
    p.add_argument("--lambda_adv", type=float, default=1.0)
    p.add_argument("--lambda_dec", type=float, default=1.0)
    p.add_argument("--lambda_disrupt_max", type=float, default=4.45,
                    help="Default matches STAGE2_WRITEUP.md Section 7's gradient-matched "
                         "starting point for sim-mode -- re-derive with gradient_diagnostic.py "
                         "if you change disrupt_mode, capacity_lora_r, or anything upstream "
                         "of the disruption loss, per that section's own methodology.")
    p.add_argument("--disrupt_mode", type=str, default="sim", choices=["mel", "sim"])
    p.add_argument("--lambda_ramp_steps", type=int, default=200)
    p.add_argument("--disrupt_lambda_mel", type=float, default=2.0)
    p.add_argument("--disrupt_weight_l1", type=float, default=1.0)
    p.add_argument("--disrupt_weight_kl", type=float, default=0.001)
    p.add_argument("--surrogate_sample_rate", type=int, default=16000)
    p.add_argument("--surrogate_text", type=str, default="This is a test sentence for voice cloning.")
    p.add_argument("--no_discriminator", action="store_true")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--checkpoint_every", type=int, default=5)
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints/stage2_capacity_ffn")
    return p


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_stage2_capacity] Using device: {device}")

    print("[train_stage2_capacity] Loading backbone + FFN-capacity adapters + discriminator...")
    backbone = VoiceMarkBackbone()
    apply_lora_adapters(
        backbone, r=args.lora_r, alpha=args.lora_alpha,
        targets=("msg_processor", "detector"),
        include_ffn=True, ffn_r=args.capacity_lora_r, ffn_targets=("msg_processor",),
    )
    load_stage1_checkpoint_partial(backbone, args.stage1_checkpoint)
    discriminator = VoiceMarkDiscriminator() if not args.no_discriminator else None

    print("[train_stage2_capacity] Loading YourTTS surrogate (frozen)...")
    surrogate = load_yourtts_surrogate(device=device)

    print("[train_stage2_capacity] Loading dataset...")
    train_ds = LibriSpeechSubset(
        root=args.data_root, n_speakers=args.n_speakers,
        utterances_per_speaker=args.utterances_per_speaker,
        n_eval_speakers=args.n_eval_speakers,
        eval_utterances_per_speaker=args.eval_utterances_per_speaker,
        sample_rate=16000, crop_seconds=args.crop_seconds, split="train",
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_librispeech, drop_last=True)

    voicemark_losses_fn = VoiceMarkLosses(
        sample_rate=16000, lambda_vad=args.lambda_vad, lambda_cos=args.lambda_cos,
        lambda_mel=args.lambda_mel, lambda_adv=args.lambda_adv, lambda_dec=args.lambda_dec,
    ).to(device)
    disruption_loss_fn = SafeSpeechDisruptionLoss(
        sampling_rate=args.surrogate_sample_rate, lambda_mel=args.disrupt_lambda_mel,
        weight_l1=args.disrupt_weight_l1, weight_kl=args.disrupt_weight_kl,
    ).to(device)

    trainable_params = [p for p in backbone.model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"[train_stage2_capacity] Total trainable params: {n_trainable:,} "
          f"(canonical Stage 2 was 294,912 -- compare against this number in your writeup)")
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    global_step = 0
    for epoch in range(args.epochs):
        backbone.model.train()
        epoch_acc = []

        for batch in train_loader:
            clean_audio = batch["waveform"].to(device)
            message = random_message(16, clean_audio.shape[0], device)
            current_lambda = lambda_schedule(global_step, args.lambda_ramp_steps, args.lambda_disrupt_max)

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

            if current_lambda > 0:
                cloned_output = surrogate.clone_voice(recon_wm, text=args.surrogate_text)
                if args.disrupt_mode == "sim":
                    emb_clean = surrogate.compute_speaker_embedding(clean_audio)
                    emb_cloned = surrogate.compute_speaker_embedding(cloned_output)
                    sim_loss = compute_sim_disruption_loss(emb_clean, emb_cloned)
                    disrupt_loss_dict = {"total": sim_loss, "sim_disruption": sim_loss.detach()}
                else:
                    disrupt_loss_dict = disruption_loss_fn(clean_audio, cloned_output)
            else:
                disrupt_loss_dict = {"total": torch.tensor(0.0, device=device)}

            total_loss = vm_loss_dict["total"] + current_lambda * disrupt_loss_dict["total"]

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            acc = compute_detection_accuracy(chunk_logits.detach(), message)
            epoch_acc.append(acc)
            global_step += 1
            if global_step % args.log_every == 0:
                recent_acc = sum(epoch_acc[-args.log_every:]) / len(epoch_acc[-args.log_every:])
                print(f"[epoch {epoch} step {global_step}] capacity_r={args.capacity_lora_r} "
                      f"lambda={current_lambda:.4f} total={total_loss.item():.4f} acc={recent_acc:.4f}")

        epoch_avg_acc = sum(epoch_acc) / len(epoch_acc)
        print(f"=== Epoch {epoch} complete. Train-batch avg ACC: {epoch_avg_acc:.4f} ===")

        if (epoch + 1) % args.checkpoint_every == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(args.checkpoint_dir, f"stage2_capacity_epoch{epoch}.pt")
            lora_state = {k: v for k, v in backbone.model.state_dict().items() if "_lora." in k}
            torch.save({
                "epoch": epoch, "lora_state_dict": lora_state,
                "lora_r": args.lora_r, "capacity_lora_r": args.capacity_lora_r,
                "lora_alpha": args.lora_alpha, "avg_acc": epoch_avg_acc, "global_step": global_step,
            }, ckpt_path)
            print(f"[train_stage2_capacity] Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    train(args)
