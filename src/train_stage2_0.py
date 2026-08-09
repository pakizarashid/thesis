"""
src/train_stage2.py

Stage 2 training: joint objective combining VoiceMark's five losses (Stage 1)
with SafeSpeech's disruption loss, per the project plan:
    L = (Lvad + Lcos + Lmel + Ladv + Ldec) + lambda * SafeSpeech_disruption_loss
with a lambda RAMP (starts at 0, ramps to lambda_max over the first N steps)
rather than a fixed constant from step one -- per the plan's Section 3
reasoning: the disruption loss reaches the shared LoRA parameters through many
more hops (through the full surrogate cloner) than the VoiceMark losses do
(1-2 hops from the RVQ latents), so introducing it at full strength
immediately risks destabilizing a Stage 1 checkpoint that already reproduces
the paper's reported ACC.

RESUMES from a Stage 1 checkpoint (--stage1_checkpoint) by default -- this is
deliberate, not optional: Stage 1 already validated that this checkpoint
reproduces VoiceMark's accuracy; starting Stage 2 from scratch would throw
that validation away.

CONFIDENCE NOTE: verify the surrogate's actual output sample rate against
SafeSpeechDisruptionLoss's sampling_rate argument before trusting numbers from
this script -- see the accompanying chat message for how to check.
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
from safespeech_losses import SafeSpeechDisruptionLoss
from librispeech import LibriSpeechSubset, collate_librispeech


def random_message(nbits: int, batch_size: int, device) -> torch.Tensor:
    return torch.randint(0, 2, (batch_size, nbits), device=device)


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


def load_stage1_checkpoint(backbone, checkpoint_path: str = None):
    """
    Loads a Stage 1 checkpoint if provided AND its LoRA rank matches the
    current run's --lora_r. If checkpoint_path is None, or the checkpoint's
    saved tensor shapes don't match the current rank (e.g. testing a
    different --lora_r than Stage 1 was trained with), falls back to LoRA
    zero-init (mathematically equivalent to pretrained VoiceMark -- see
    apply_lora_adapters's own zero-init property, confirmed earlier in this
    project). PyTorch's load_state_dict(strict=False) tolerates missing/extra
    KEYS but NOT shape mismatches on matching keys -- it would raise a
    RuntimeError, not silently skip, without this explicit handling.
    """
    if checkpoint_path is None:
        print("[load_stage1_checkpoint] No checkpoint provided -- starting from "
              "LoRA zero-init (== pretrained VoiceMark).")
        return

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    lora_state = ckpt["lora_state_dict"]

    current_shapes = {k: v.shape for k, v in backbone.model.state_dict().items() if "_lora." in k}
    shape_mismatch = any(
        k in current_shapes and current_shapes[k] != v.shape
        for k, v in lora_state.items()
    )
    if shape_mismatch:
        print(f"[load_stage1_checkpoint] WARNING - checkpoint's LoRA tensor shapes don't match "
              f"the current run's --lora_r/--lora_alpha (checkpoint was likely trained with a "
              f"different rank). Falling back to LoRA zero-init instead of crashing on the "
              f"shape mismatch. If you need to resume from this exact checkpoint, use matching "
              f"--lora_r/--lora_alpha values.")
        return

    missing, unexpected = backbone.model.load_state_dict(lora_state, strict=False)
    non_lora_missing = [k for k in missing if "_lora." in k]
    if non_lora_missing:
        print(f"[load_stage1_checkpoint] WARNING - {len(non_lora_missing)} LoRA keys missing: "
              f"{non_lora_missing[:3]}...")
    print(f"[load_stage1_checkpoint] Resumed from {checkpoint_path} "
          f"(Stage 1 epoch {ckpt.get('epoch')}, train-time avg_acc={ckpt.get('avg_acc'):.4f})")


def lambda_schedule(step: int, ramp_steps: int, lambda_max: float) -> float:
    """Linear ramp from 0 to lambda_max over ramp_steps, then held constant."""
    if ramp_steps <= 0:
        return lambda_max
    return lambda_max * min(1.0, step / ramp_steps)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_stage2] Using device: {device}")

    print("[train_stage2] Loading backbone + adapters + discriminator...")
    backbone = VoiceMarkBackbone()
    apply_lora_adapters(backbone, r=args.lora_r, alpha=args.lora_alpha)
    load_stage1_checkpoint(backbone, args.stage1_checkpoint)
    discriminator = VoiceMarkDiscriminator() if not args.no_discriminator else None

    print("[train_stage2] Loading YourTTS surrogate (frozen)...")
    surrogate = load_yourtts_surrogate(device=device)

    print("[train_stage2] Loading dataset...")
    train_ds = LibriSpeechSubset(
        root=args.data_root, n_speakers=args.n_speakers,
        utterances_per_speaker=args.utterances_per_speaker,
        n_eval_speakers=args.n_eval_speakers,
        eval_utterances_per_speaker=args.eval_utterances_per_speaker,
        sample_rate=16000, crop_seconds=args.crop_seconds, split="train",
    )
    eval_ds = LibriSpeechSubset(
        root=args.data_root, n_speakers=args.n_speakers,
        utterances_per_speaker=args.utterances_per_speaker,
        n_eval_speakers=args.n_eval_speakers,
        eval_utterances_per_speaker=args.eval_utterances_per_speaker,
        sample_rate=16000, crop_seconds=args.crop_seconds, split="eval",
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_librispeech, drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_librispeech, drop_last=False)

    voicemark_losses_fn = VoiceMarkLosses(
        sample_rate=16000, lambda_vad=args.lambda_vad, lambda_cos=args.lambda_cos,
        lambda_mel=args.lambda_mel, lambda_adv=args.lambda_adv, lambda_dec=args.lambda_dec,
    ).to(device)
    disruption_loss_fn = SafeSpeechDisruptionLoss(
        sampling_rate=args.surrogate_sample_rate, lambda_mel=args.disrupt_lambda_mel,
        weight_l1=args.disrupt_weight_l1, weight_kl=args.disrupt_weight_kl,
    ).to(device)

    trainable_params = [p for p in backbone.model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    global_step = 0
    for epoch in range(args.epochs):
        backbone.model.train()
        epoch_metrics = {}
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

            # Disruption loss only engages once lambda ramps above ~0 -- skip
            # the (expensive) surrogate forward pass entirely at step 0 if
            # lambda is exactly 0, saving compute during the ramp's start.
            if current_lambda > 0:
                cloned_output = surrogate.clone_voice(recon_wm, text=args.surrogate_text)
                disrupt_loss_dict = disruption_loss_fn(clean_audio, cloned_output)
            else:
                disrupt_loss_dict = {"total": torch.tensor(0.0, device=device),
                                      "pivotal_disruption": torch.tensor(0.0, device=device),
                                      "l1_to_noise": torch.tensor(0.0, device=device),
                                      "kl_to_noise": torch.tensor(0.0, device=device)}

            total_loss = vm_loss_dict["total"] + current_lambda * disrupt_loss_dict["total"]

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            acc = compute_detection_accuracy(chunk_logits.detach(), message)
            epoch_acc.append(acc)
            for k, v in vm_loss_dict.items():
                if k != "total":
                    epoch_metrics.setdefault(f"vm_{k}", []).append(v.item())
            for k, v in disrupt_loss_dict.items():
                if k != "total":
                    epoch_metrics.setdefault(f"disrupt_{k}", []).append(
                        v.item() if torch.is_tensor(v) else v)

            global_step += 1
            if global_step % args.log_every == 0:
                recent_acc = sum(epoch_acc[-args.log_every:]) / len(epoch_acc[-args.log_every:])
                # Per-component breakdown, NOT just the combined disrupt_total --
                # the combined number can stay flat while pivotal_disruption
                # (the term that actually drives SIM) moves in either direction,
                # masked by l1/kl offsetting it. This was a real blind spot in
                # earlier runs; always check pivotal_disruption's own trend
                # directly, not just the aggregate.
                pivotal_val = disrupt_loss_dict["pivotal_disruption"]
                l1_val = disrupt_loss_dict["l1_to_noise"]
                kl_val = disrupt_loss_dict["kl_to_noise"]
                print(f"[epoch {epoch} step {global_step}] lambda={current_lambda:.4f} "
                      f"total={total_loss.item():.4f} acc={recent_acc:.4f} "
                      f"vm_total={vm_loss_dict['total'].item():.4f} "
                      f"disrupt_total={disrupt_loss_dict['total'].item() if torch.is_tensor(disrupt_loss_dict['total']) else disrupt_loss_dict['total']:.4f} "
                      f"| pivotal={pivotal_val.item() if torch.is_tensor(pivotal_val) else pivotal_val:.4f} "
                      f"l1_noise={l1_val.item() if torch.is_tensor(l1_val) else l1_val:.4f} "
                      f"kl_noise={kl_val.item() if torch.is_tensor(kl_val) else kl_val:.4f}")

        epoch_avg_acc = sum(epoch_acc) / len(epoch_acc)
        print(f"=== Epoch {epoch} complete. Train-batch avg ACC: {epoch_avg_acc:.4f} ===")

        if (epoch + 1) % args.checkpoint_every == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(args.checkpoint_dir, f"stage2_epoch{epoch}.pt")
            lora_state = {k: v for k, v in backbone.model.state_dict().items() if "_lora." in k}
            torch.save({
                "epoch": epoch, "lora_state_dict": lora_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
                "avg_acc": epoch_avg_acc, "global_step": global_step,
            }, ckpt_path)
            print(f"[train_stage2] Saved checkpoint to {ckpt_path}")


def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_checkpoint", type=str, default=None,
                    help="Path to a Stage 1 checkpoint to resume from. Optional -- if omitted, "
                         "or if its LoRA shapes don't match --lora_r/--lora_alpha, falls back "
                         "to LoRA zero-init (== pretrained VoiceMark) with a warning.")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=2,
                    help="Smaller default than Stage 1 -- the surrogate forward pass is far more expensive per step.")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lora_r", type=int, default=8)
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
    p.add_argument("--lambda_disrupt_max", type=float, default=1.0,
                    help="Outer lambda multiplying the whole SafeSpeech disruption loss, per the plan's "
                         "L = VoiceMark losses + lambda * SafeSpeech losses.")
    p.add_argument("--lambda_ramp_steps", type=int, default=200,
                    help="Steps to linearly ramp lambda_disrupt from 0 to lambda_disrupt_max.")
    p.add_argument("--disrupt_lambda_mel", type=float, default=2.0,
                    help="Inner weight on SafeSpeechDisruptionLoss's own pivotal term. "
                         "See safespeech_losses.py docstring for how this default was derived.")
    p.add_argument("--disrupt_weight_l1", type=float, default=1.0,
                    help="Weight on l1_to_noise. Lowered from SafeSpeech's original 10 -- "
                         "gradient_diagnostic.py showed l1's raw gradient was already "
                         "comparable to pivotal_disruption's, so x10 over-weighted it.")
    p.add_argument("--disrupt_weight_kl", type=float, default=0.001,
                    help="Weight on kl_to_noise. Lowered further from an initial 0.01 attempt -- "
                         "gradient_diagnostic.py showed kl's raw gradient at ~325, roughly "
                         "1000x pivotal's, requiring this much reduction. Re-run "
                         "gradient_diagnostic.py after any change here before a long run.")
    p.add_argument("--surrogate_sample_rate", type=int, default=16000,
                    help="VERIFY this matches your surrogate's actual output sample rate before trusting results.")
    p.add_argument("--surrogate_text", type=str, default="This is a test sentence for voice cloning.",
                    help="Fixed placeholder text for the surrogate's synthesis -- content doesn't matter for disruption.")
    p.add_argument("--no_discriminator", action="store_true")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--checkpoint_every", type=int, default=5)
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints/stage2")
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    train(args)
