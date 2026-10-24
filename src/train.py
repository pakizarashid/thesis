"""
src/train.py

Stage 1 training: fine-tune LoRA adapters on msg_processor/detector against
VoiceMark's five losses ONLY (Lvad, Lcos, Lmel, Ladv, Ldec). No disruption loss
yet -- that's Stage 2, which requires the YourTTS surrogate (not yet built).

Purpose of this stage (per the project plan): sanity-check that this
reimplementation reproduces VoiceMark's reported ACC (0.96-0.98) on your
LibriSpeech subset BEFORE adding the harder Stage 2 objective on top. If ACC
doesn't approach that range here, something in the backbone/adapter/loss
wiring needs fixing before Stage 2 is worth attempting.

Usage:
    python src/train.py --epochs 30 --batch_size 4 --lr 5e-5
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
from voicemark_losses import VoiceMarkLosses
from librispeech import LibriSpeechSubset, collate_librispeech
from augment import apply_random_augmentation  # legacy path only (--legacy_augmentation)
from augment_voicemark import add_augmentation_args, build_augmenter, samples_to_frames


def random_message(nbits: int, batch_size: int, device) -> torch.Tensor:
    return torch.randint(0, 2, (batch_size, nbits), device=device)


def compute_detection_accuracy(chunk_logits: torch.Tensor, message: torch.Tensor, nchunk_size: int = 4) -> float:
    """
    Bitwise accuracy, matching VoiceMark's own ACC metric definition (ratio of
    correctly decoded bits to total bits) -- reconstructs bits from predicted
    chunk indices and compares against ground truth bit-by-bit, not just
    chunk-level exact match (which would be a stricter, different metric).
    """
    pred_chunks = torch.argmax(chunk_logits, dim=-1)  # [batch, nchunks]
    batch, nchunks = pred_chunks.shape

    correct_bits = 0
    total_bits = 0
    for i in range(nchunks):
        true_chunk = message[:, i * nchunk_size:(i + 1) * nchunk_size]  # [batch, nchunk_size]
        pred_val = pred_chunks[:, i]  # [batch]
        for bit_idx in range(nchunk_size):
            pred_bit = (pred_val >> bit_idx) & 1
            true_bit = true_chunk[:, bit_idx]
            correct_bits += (pred_bit == true_bit).sum().item()
            total_bits += batch
    return correct_bits / total_bits


def run_eval(backbone, eval_loader, device, augmenter=None) -> float:
    """Runs the detector on the held-out eval split, no gradient. Without an
    augmenter this measures clean-condition detection ability; with one it
    measures ACC on augmented audio (a cheap proxy for cloning robustness,
    reported next to the clean number every epoch). Fixed seeds so epochs are
    comparable. Returns average bitwise ACC across the eval set."""
    import random as _random
    backbone.model.eval()
    accs = []
    rng_state = (torch.get_rng_state(), _random.getstate(),
                 torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    torch.manual_seed(1234)  # also seeds CUDA
    _random.seed(1234)
    with torch.no_grad():
        for batch in eval_loader:
            clean_audio = batch["waveform"].to(device)
            message = random_message(16, clean_audio.shape[0], device)
            out = backbone.forward_full(clean_audio, message)
            det_input = out["recon_wm"]
            if augmenter is not None:
                det_input = augmenter(det_input, clean_audio)["audio"]
            detect_feat = backbone.model.st_model.forward_feature(det_input)
            _, chunk_logits = backbone.model.detector(detect_feat)
            accs.append(compute_detection_accuracy(chunk_logits, message))
    torch.set_rng_state(rng_state[0])
    _random.setstate(rng_state[1])
    if rng_state[2] is not None:
        torch.cuda.set_rng_state_all(rng_state[2])
    backbone.model.train()
    return sum(accs) / len(accs)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader_gen = None
    if args.seed is not None:
        # 2026-10-24: makes data order, messages and augmentation draws repeatable
        # (GPU kernels are still not bit-deterministic). None = old unseeded behaviour.
        import random as _rnd
        import numpy as _np
        torch.manual_seed(args.seed)
        _rnd.seed(args.seed)
        _np.random.seed(args.seed)
        loader_gen = torch.Generator().manual_seed(args.seed)
    print(f"[train] Using device: {device}")

    print("[train] Loading backbone + adapters + discriminator...")
    backbone = VoiceMarkBackbone()
    apply_lora_adapters(backbone, r=args.lora_r, alpha=args.lora_alpha)
    discriminator = VoiceMarkDiscriminator() if not args.no_discriminator else None

    print("[train] Loading datasets (speaker-disjoint train/eval split)...")
    train_ds = LibriSpeechSubset(
        root=args.data_root,
        n_speakers=args.n_speakers,
        utterances_per_speaker=args.utterances_per_speaker,
        n_eval_speakers=args.n_eval_speakers,
        eval_utterances_per_speaker=args.eval_utterances_per_speaker,
        sample_rate=16000,
        crop_seconds=args.crop_seconds,
        split="train",
    )
    eval_ds = LibriSpeechSubset(
        root=args.data_root,
        n_speakers=args.n_speakers,
        utterances_per_speaker=args.utterances_per_speaker,
        n_eval_speakers=args.n_eval_speakers,
        eval_utterances_per_speaker=args.eval_utterances_per_speaker,
        sample_rate=16000,
        crop_seconds=args.crop_seconds,
        split="eval",
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_librispeech, drop_last=True, generator=loader_gen,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_librispeech, drop_last=False,
    )

    losses_fn = VoiceMarkLosses(
        sample_rate=16000,
        lambda_vad=args.lambda_vad, lambda_cos=args.lambda_cos,
        lambda_mel=args.lambda_mel, lambda_adv=args.lambda_adv,
        lambda_dec=args.lambda_dec,
    ).to(device)

    # Only LoRA parameters are trainable (backbone frozen by construction --
    # see apply_lora_adapters's own sanity check for non-LoRA trainable params)
    trainable_params = [p for p in backbone.model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    disc_optimizer = discriminator.build_optimizer(lr=args.disc_lr) if discriminator is not None else None

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("[train] Running PRE-TRAINING baseline eval (LoRA at zero-init -- "
          "should reflect pretrained VoiceMark's own detection ability on unseen speakers)...")
    baseline_acc = run_eval(backbone, eval_loader, device)
    print(f"=== Baseline held-out ACC (before any Stage 1 fine-tuning): {baseline_acc:.4f} ===")

    augmenter = None
    if args.use_augmentation and not args.legacy_augmentation:
        augmenter = build_augmenter(args, sample_rate=16000)
        print(f"[train] VoiceMark-faithful augmentation: mode={args.aug_mode} "
              f"list={args.aug_list} p_identity={args.aug_p_identity}")
        aug_eval_acc0 = run_eval(backbone, eval_loader, device, augmenter=augmenter)
        print(f"=== Baseline held-out ACC on AUGMENTED audio: {aug_eval_acc0:.4f} ===")
    elif args.use_augmentation:
        print("[train] WARNING: --legacy_augmentation reproduces the old behaviour in which the "
              "detector never sees augmented audio. Use only to reproduce old checkpoints.")

    global_step = 0
    for epoch in range(args.epochs):
        backbone.model.train()
        epoch_losses = {}
        epoch_acc = []
        aug_counts = {}

        for batch in train_loader:
            clean_audio = batch["waveform"].to(device)  # [B, 1, T]
            message = random_message(16, clean_audio.shape[0], device)

            out = backbone.forward_full(clean_audio, message)
            augmentation_mask = None
            vad_waveform = None
            det_input = out["recon_wm"]

            if args.use_augmentation and not args.legacy_augmentation:
                # 2026-10-24 fix: the detector is trained on the AUGMENTED
                # watermarked audio (VoiceMark Sec. 2.3: "it learns from
                # augmented audio x~"). Lmel/Lcos/Ladv stay on the clean
                # watermarked audio. VAD labels come from the clean audio after
                # the same structural edits, with masked and replaced frames
                # forced to 0 (Sec. 2.5).
                aug = augmenter(out["recon_wm"], clean_audio)
                det_input = aug["audio"]
                vad_waveform = aug["vad_ref"]
                for n in aug["names"]:
                    aug_counts[n] = aug_counts.get(n, 0) + 1

            detect_feat = backbone.model.st_model.forward_feature(det_input)
            logits, chunk_logits = backbone.model.detector(detect_feat)

            if args.use_augmentation and not args.legacy_augmentation:
                augmentation_mask = samples_to_frames(aug["nowm_mask"], logits.shape[1])
            elif args.use_augmentation and args.legacy_augmentation:
                # Pre-2026-10-24 behaviour, kept ONLY to reproduce old
                # checkpoints: the augmented waveform is discarded and only the
                # VAD labels change, so the detector never sees augmented audio.
                downsample_rate = backbone.model.st_model.downsample_rate
                n_frames_est = clean_audio.shape[-1] // downsample_rate
                masks = []
                for i in range(clean_audio.shape[0]):
                    _, frame_mask, _ = apply_random_augmentation(
                        clean_audio[i].cpu(), downsample_rate, n_frames_est
                    )
                    masks.append(frame_mask)
                augmentation_mask = torch.stack(masks).to(device)

            loss_dict = losses_fn(
                recon_wm=out["recon_wm"], clean_audio=clean_audio,
                acoustic=out["acoustic"], acoustic_wm=out["acoustic_wm"],
                detection_logits=logits, chunk_logits=chunk_logits,
                message=message, discriminator=discriminator,
                augmentation_mask=augmentation_mask,
                vad_waveform=vad_waveform,
            )

            optimizer.zero_grad()
            loss_dict["total"].backward()
            optimizer.step()

            # --- Discriminator update (separate step, standard GAN pattern) ---
            if discriminator is not None and args.train_discriminator:
                from voicemark_losses import compute_ladv_discriminator
                d_loss = compute_ladv_discriminator(discriminator, out["recon_wm"], clean_audio)
                disc_optimizer.zero_grad()
                d_loss.backward()
                disc_optimizer.step()

            acc = compute_detection_accuracy(chunk_logits.detach(), message)
            epoch_acc.append(acc)

            for k, v in loss_dict.items():
                if k == "total":
                    continue
                epoch_losses.setdefault(k, []).append(v.item())

            global_step += 1
            if global_step % args.log_every == 0:
                avg_losses = {k: sum(v[-args.log_every:]) / len(v[-args.log_every:]) for k, v in epoch_losses.items()}
                recent_acc = sum(epoch_acc[-args.log_every:]) / len(epoch_acc[-args.log_every:])
                print(f"[epoch {epoch} step {global_step}] total={loss_dict['total'].item():.4f} "
                      f"acc={recent_acc:.4f} | " +
                      " ".join(f"{k}={v:.4f}" for k, v in avg_losses.items()))

        epoch_avg_acc = sum(epoch_acc) / len(epoch_acc)
        held_out_acc = run_eval(backbone, eval_loader, device)
        print(f"=== Epoch {epoch} complete. Train-batch avg ACC: {epoch_avg_acc:.4f} | "
              f"Held-out eval ACC: {held_out_acc:.4f} (baseline was {baseline_acc:.4f}) ===")
        if augmenter is not None:
            aug_acc = run_eval(backbone, eval_loader, device, augmenter=augmenter)
            print(f"    held-out ACC on augmented audio: {aug_acc:.4f} (baseline {aug_eval_acc0:.4f}) "
                  f"| train augmentation counts: {dict(sorted(aug_counts.items()))}")

        if (epoch + 1) % args.checkpoint_every == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(args.checkpoint_dir, f"stage1_epoch{epoch}.pt")
            # Save ONLY the LoRA parameters + optimizer state -- the frozen
            # base weights are always reloadable from the original VoiceMark
            # checkpoint, no need to duplicate 125M frozen params per checkpoint.
            lora_state = {
                k: v for k, v in backbone.model.state_dict().items()
                if "_lora." in k
            }
            torch.save({
                "epoch": epoch,
                "lora_state_dict": lora_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "avg_acc": epoch_avg_acc,
                # 2026-10-24: store the full command line, so a checkpoint
                # always records how it was trained (the stage1_scaleup_aug
                # checkpoint could not be traced back to its augmentation).
                "train_args": vars(args),
            }, ckpt_path)
            print(f"[train] Saved checkpoint to {ckpt_path}")


def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)  # matches paper's reported epoch count
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)  # matches paper's stated Adam lr
    p.add_argument("--disc_lr", type=float, default=5e-5)
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
    p.add_argument("--use_augmentation", action="store_true",
                    help="Train the detector on VoiceMark-style augmented audio "
                         "(src/data/augment_voicemark.py). See --aug_* options.")
    p.add_argument("--seed", type=int, default=None,
                    help="Seed for data order, messages and augmentation draws. None = old behaviour.")
    p.add_argument("--legacy_augmentation", action="store_true",
                    help="Reproduce the pre-2026-10-24 behaviour (augmented audio discarded, "
                         "only VAD labels changed). For reproducing old checkpoints only.")
    p.add_argument("--no_discriminator", action="store_true",
                    help="Skip Ladv entirely (useful for a fast first smoke "
                         "test of the other four losses).")
    p.add_argument("--train_discriminator", action="store_true",
                    help="Also update the discriminator's own weights each "
                         "step. Default OFF: the loaded discriminator is "
                         "already well-trained (epoch 46 checkpoint); "
                         "fine-tuning it further on your small subset risks "
                         "destabilizing it. Turn on only if Ladv plateaus.")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--checkpoint_every", type=int, default=5)
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    add_augmentation_args(p)
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    train(args)