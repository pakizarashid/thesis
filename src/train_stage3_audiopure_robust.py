"""
src/train_stage3_audiopure_robust.py

Stage 3: closes the still-open AudioPure purification gap that Step 8's PGD
hybrid does NOT address. PGD (disruption_pgd.py) makes cloning fail after
watermarking; it says nothing about whether the watermark itself survives an
adversary running AudioPure's diffusion-based purification on the watermarked
audio BEFORE anyone tries to clone or verify it (audiopure_eval.py's own
headline test: ACC before/after purification). Those are two independent
attack surfaces on two independent defenses (traceability vs. disruption) --
fixing one does not touch the other, which is exactly why this is a separate
stage rather than a knob on disruption_pgd.py.

MECHANISM: freeze msg_processor COMPLETELY (including its already-trained
stage1/stage2 LoRA delta -- recon_wm generation must stay EXACTLY as
calibrated, since the PGD epsilon sweep, PESQ/STOI numbers, and every other
result in this project assume this specific msg_processor state). LoRA-
fine-tune ONLY detector's existing adapter, continuing training on top of
whatever the base checkpoint already learned, but now against AudioPure-
PURIFIED audio as well as clean audio -- i.e. treat AudioPure purification as
a training-time augmentation, the same conceptual move augment.py already
makes for masking/shuffling/replacing (see that file's docstring), just with
a diffusion denoiser standing in for a VC pipeline's distortion. detector is
never asked to shape recon_wm (it doesn't -- msg_processor already produced
it before purification runs), so there is no risk of this stage degrading
imperceptibility or the disruption numbers already banked in Step 8.

WHY THIS DOESN'T NEED BACKWARD THROUGH THE DIFFUSION MODEL: purification only
ever appears as an INPUT transform (recon_wm -> purified), computed entirely
under torch.no_grad() -- exactly like audiopure_eval.py's own eval loop.
Gradient only needs to reach detector's LoRA parameters via detector's own
forward pass on the (already-computed, already-detached) purified or clean
tensor. This also means the cuDNN "RNN backward in eval mode" restriction
disruption_pgd.py had to work around (torch.backends.cudnn.flags) does NOT
apply here: st_model.forward_feature() runs entirely inside torch.no_grad()
below (no backward is ever requested through it), so its internal RNN never
gets asked to do the thing cuDNN refuses. Only detector's own (transformer-
based, not RNN-based -- see adapters.py's docstring) forward needs backward,
and that's unaffected by backbone.model.eval() mode.

TRAINING-TIME AUGMENTATION MIX: --purify_prob (default 0.7) controls what
fraction of TRAINING batches get AudioPure-purified before the detector loss
is computed; the rest see plain recon_wm. This is a batch-level coin flip
(not per-sample) purely for implementation simplicity -- DiffWave purification
is not cheap (reverse_timestep diffusion steps per batch), and running it on
every single batch would make this stage very slow for no clear benefit over
a high-probability mix. The remaining clean-condition batches exist
specifically so this script optimizes BOTH numbers at once (purified ACC via
the augmented batches, clean-condition ACC via the un-augmented ones) rather
than letting clean-condition detection quietly regress while purified-
condition detection improves -- --eval_every reports both explicitly every
validation pass so that regression would be caught immediately, not assumed
away.

COST WARNING: AudioPure's DiffWave denoiser runs `reverse_timestep` diffusion
sampling steps per forward call (25 by default, matching audiopure_eval.py) --
this is substantially slower per training step than Stage 1/2's plain forward
passes. Budget accordingly: start with a short --epochs (5-10, not 30),
small --n_speakers, and --purify_prob lower than 0.7 for the first smoke run
if wall-clock time on your Kaggle GPU quota is a concern. --eval_every runs a
SMALL validation pass (--n_eval_batches, default 5) for the same reason --
this script does not re-run full audiopure_eval.py-scale evaluation (n=100)
during training; do that separately with audiopure_eval.py itself once a
checkpoint here looks promising.

RUN THIS BEFORE COMMITTING GPU HOURS: written and syntax-checked but NOT run
against your actual Kaggle GPU pipeline (that requires diffusion_models/
DiffWave_Unconditional's checkpoint + config to be present under
external/audiopure/, same as audiopure_eval.py needs). Run --diagnostic
first: 1 batch, 1 step, purified AND clean losses both printed, confirms (a)
AudioPure's denoiser loads and runs, (b) gradient actually reaches detector's
LoRA params, before burning GPU hours on a full run that could be silently
broken at either point.

Usage:
    # 1. Cheap sanity check first (loads denoiser, 1 batch, 1 step, no save):
    python src/train_stage3_audiopure_robust.py \\
        --stage1_checkpoint ./checkpoints/stage1_full_recalibrated_v3/recalibrated_final.pt \\
        --diagnostic

    # 2. Short smoke run once the diagnostic looks sane:
    python src/train_stage3_audiopure_robust.py \\
        --stage1_checkpoint ./checkpoints/stage1_full_recalibrated_v3/recalibrated_final.pt \\
        --epochs 5 --n_speakers 10 --purify_prob 0.7 \\
        --checkpoint_dir ./checkpoints/stage3_audiopure_robust

    # 3. Then validate the resulting checkpoint with the EXISTING eval script
    #    (unmodified -- this stage's checkpoints are byte-compatible with it):
    python src/eval/audiopure_eval.py \\
        --checkpoint ./checkpoints/stage3_audiopure_robust/stage3_epoch4.pt \\
        --output results/results_audiopure_stage3_run1.json
    #    Compare acc_after against the pre-Stage-3 baseline's acc_after for the
    #    same checkpoint family -- that delta is Stage 3's actual headline number.
"""

import os
import sys
import random
import argparse
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "losses"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "eval"))

from backbone import VoiceMarkBackbone
from adapters import apply_lora_adapters
from voicemark_losses import compute_ldec
from librispeech import LibriSpeechSubset, collate_librispeech
from vctk import VCTKSubset, collate_vctk
from libritts import LibriTTSSubset, collate_libritts
from train_stage2 import random_message, compute_detection_accuracy
from audiopure_eval import build_audiopure_denoiser


def build_backbone_stage3(checkpoint_path: str, r: int, alpha: int, include_ffn: bool, capacity_lora_r: int):
    """
    Reconstructs the SAME LoRA architecture the base checkpoint was trained
    with (both msg_processor and detector wrapped -- matching
    disruption_pgd.py's build_backbone / train_stage2*.py's own construction),
    loads the full lora_state_dict so msg_processor's trained delta is
    restored EXACTLY, then explicitly freezes msg_processor's LoRA params
    (they loaded as trainable-by-construction; this call turns that off).
    detector's LoRA params are left trainable -- they are the ONLY thing this
    script ever updates.

    Loading msg_processor's LoRA into a wrapper and then freezing it (rather
    than never wrapping it at all) matters: if msg_processor were left
    unwrapped, the checkpoint's msg_processor._lora.* keys would silently
    fail to load (no matching submodule), reverting msg_processor to its
    zero-LoRA pretrained state -- a SILENT change to recon_wm that would
    invalidate every PGD/PESQ/STOI number already banked against the actual
    checkpoint's recon_wm. Wrap-then-freeze avoids that trap entirely.
    """
    backbone = VoiceMarkBackbone()
    if include_ffn:
        apply_lora_adapters(
            backbone, r=r, alpha=alpha, targets=("msg_processor", "detector"),
            include_ffn=True, ffn_r=capacity_lora_r, ffn_targets=("msg_processor",),
        )
    else:
        apply_lora_adapters(backbone, r=r, alpha=alpha)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    lora_state = ckpt["lora_state_dict"]
    missing, unexpected = backbone.model.load_state_dict(lora_state, strict=False)
    non_lora_missing = [k for k in missing if "_lora." in k]
    if non_lora_missing:
        print(f"[build_backbone_stage3] WARNING - {len(non_lora_missing)} LoRA keys missing: "
              f"{non_lora_missing[:3]}... (checkpoint/--include_ffn/--capacity_lora_r mismatch?)")
    print(f"[build_backbone_stage3] Loaded base checkpoint {checkpoint_path} "
          f"(epoch {ckpt.get('epoch')}, avg_acc={ckpt.get('avg_acc')}).")

    backbone.model.eval()

    n_frozen = 0
    for name, p in backbone.model.msg_processor.named_parameters():
        if p.requires_grad:
            p.requires_grad_(False)
            n_frozen += 1
    print(f"[build_backbone_stage3] Froze {n_frozen} msg_processor params (including its "
          f"trained LoRA delta) -- recon_wm generation is now IDENTICAL to the base "
          f"checkpoint for the rest of this run, no matter what detector's loss does.")

    trainable = [n for n, p in backbone.model.named_parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for n, p in backbone.model.named_parameters() if p.requires_grad)
    non_detector_trainable = [n for n in trainable if not n.startswith("detector.")]
    if non_detector_trainable:
        print(f"[build_backbone_stage3] WARNING - {len(non_detector_trainable)} trainable params "
              f"outside detector: {non_detector_trainable[:5]} (expected ONLY detector.*_lora.* here)")
    print(f"[build_backbone_stage3] Trainable params (detector LoRA only): {n_trainable:,}")

    return backbone


def detect_acc_and_loss(backbone, wm_audio: torch.Tensor, message: torch.Tensor, need_grad: bool):
    """
    Shared by both the training step and validation. When need_grad=False
    (validation, or computing recon_wm itself), the WHOLE forward -- including
    detector -- runs under no_grad, matching audiopure_eval.py's pattern
    exactly. When need_grad=True (a training step), only st_model.forward_feature
    runs under no_grad (it has no trainable params and we never need gradient
    w.r.t. its input here -- see module docstring); detector's forward runs
    with grad enabled so its LoRA params receive gradient from wm_loss.
    """
    if need_grad:
        with torch.no_grad():
            detect_feat = backbone.model.st_model.forward_feature(wm_audio)
        _logits, chunk_logits = backbone.model.detector(detect_feat)
        wm_loss = compute_ldec(chunk_logits, message)
        acc = compute_detection_accuracy(chunk_logits.detach(), message)
        return wm_loss, acc
    else:
        with torch.no_grad():
            detect_feat = backbone.model.st_model.forward_feature(wm_audio)
            _logits, chunk_logits = backbone.model.detector(detect_feat)
            acc = compute_detection_accuracy(chunk_logits, message)
        return None, acc


def validate(backbone, denoiser, eval_loader, device, n_eval_batches: int, seed: int = 999):
    """
    Small validation pass -- reports BOTH clean-condition and purified-
    condition detection ACC, so a Stage-3 run that quietly trades clean
    accuracy for purified accuracy (or vice versa) is caught immediately
    rather than assumed away. NOT a substitute for a full audiopure_eval.py
    run at n=100 -- see module docstring's COST WARNING.
    """
    backbone.model.eval()
    acc_clean, acc_purified = [], []
    for batch_idx, batch in enumerate(eval_loader):
        if batch_idx >= n_eval_batches:
            break
        clean_audio = batch["waveform"].to(device)
        gen = torch.Generator(device=device).manual_seed(seed + batch_idx)
        message = torch.randint(0, 2, (clean_audio.shape[0], 16), generator=gen, device=device)

        with torch.no_grad():
            out = backbone.forward_full(clean_audio, message)
            recon_wm = out["recon_wm"]
            purified = denoiser(recon_wm)

        _, a_clean = detect_acc_and_loss(backbone, recon_wm, message, need_grad=False)
        _, a_pur = detect_acc_and_loss(backbone, purified, message, need_grad=False)
        acc_clean.append(a_clean)
        acc_purified.append(a_pur)

    return sum(acc_clean) / len(acc_clean), sum(acc_purified) / len(acc_purified)


def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_checkpoint", type=str, required=True,
                    help="Base checkpoint to continue from (e.g. stage1_full_recalibrated_v3's "
                         "recalibrated_final.pt). Required -- this script fine-tunes an EXISTING "
                         "detector adapter, it does not train one from scratch.")
    p.add_argument("--include_ffn", action="store_true",
                    help="Set if --stage1_checkpoint is a stage2_capacity_ffn checkpoint. Leave "
                         "unset for stage1 / plain stage2 checkpoints (recommended base, per Step "
                         "8's finding that FFN capacity contributes nothing beyond PGD alone).")
    p.add_argument("--capacity_lora_r", type=int, default=32)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)

    p.add_argument("--repo_root", type=str, default=".", help="Path to repo root (contains external/audiopure)")
    p.add_argument("--reverse_timestep", type=int, default=25,
                    help="AudioPure's own default, matching audiopure_eval.py -- keep these equal "
                         "so training-time purification matches eval-time purification strength.")
    p.add_argument("--purify_prob", type=float, default=0.7,
                    help="Fraction of TRAINING batches purified before the detector loss (rest see "
                         "plain recon_wm) -- batch-level coin flip, see module docstring. 0.7 biases "
                         "toward the harder condition while still exercising clean-condition ACC "
                         "often enough to catch regression there.")

    p.add_argument("--epochs", type=int, default=8,
                    help="Default far lower than Stage 1/2's 30 -- see COST WARNING in module "
                         "docstring. Scale up only after a short run confirms the mechanism works.")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)

    p.add_argument("--dataset", type=str, default="librispeech", choices=["librispeech", "vctk", "libritts"])
    p.add_argument("--vctk_root", type=str,
                    default="/kaggle/input/datasets/pratt3000/vctk-corpus/VCTK-Corpus/VCTK-Corpus")
    p.add_argument("--libritts_root", type=str,
                    default="/kaggle/input/datasets/pratt3000/libritts/LibriTTS")
    p.add_argument("--data_root", type=str, default="./data/librispeech")
    p.add_argument("--n_speakers", type=int, default=15)
    p.add_argument("--utterances_per_speaker", type=int, default=10)
    p.add_argument("--n_eval_speakers", type=int, default=5)
    p.add_argument("--eval_utterances_per_speaker", type=int, default=5)
    p.add_argument("--crop_seconds", type=float, default=3.0)

    p.add_argument("--diagnostic", action="store_true",
                    help="Smoke-test mode: loads the denoiser, runs 1 batch/1 step for BOTH the "
                         "purified and clean paths, prints wm_loss/acc/grad-norm for each, exits "
                         "without saving. Run this FIRST, same discipline as gradient_diagnostic.py "
                         "before train_stage2*.py's full runs.")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=200,
                    help="Run validate() every this many training steps (not epochs -- AudioPure "
                         "purification makes each epoch expensive, so step-level checking catches "
                         "problems sooner).")
    p.add_argument("--n_eval_batches", type=int, default=5,
                    help="Small on purpose -- see COST WARNING. Use audiopure_eval.py separately "
                         "for a real n=100-scale number.")
    p.add_argument("--checkpoint_every", type=int, default=2)
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints/stage3_audiopure_robust")
    return p


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_stage3] Using device: {device}")

    backbone = build_backbone_stage3(
        args.stage1_checkpoint, args.lora_r, args.lora_alpha, args.include_ffn, args.capacity_lora_r,
    )
    print("[train_stage3] Loading AudioPure denoiser (frozen, forward-only -- never backprop through it)...")
    denoiser = build_audiopure_denoiser(args.repo_root, reverse_timestep=args.reverse_timestep)

    if args.dataset == "librispeech":
        ds_kwargs = dict(root=args.data_root)
        DS, collate_fn = LibriSpeechSubset, collate_librispeech
    elif args.dataset == "vctk":
        ds_kwargs = dict(root=args.vctk_root)
        DS, collate_fn = VCTKSubset, collate_vctk
    else:
        ds_kwargs = dict(root=args.libritts_root)
        DS, collate_fn = LibriTTSSubset, collate_libritts

    common_kwargs = dict(
        n_speakers=args.n_speakers, utterances_per_speaker=args.utterances_per_speaker,
        n_eval_speakers=args.n_eval_speakers, eval_utterances_per_speaker=args.eval_utterances_per_speaker,
        sample_rate=16000, crop_seconds=args.crop_seconds,
    )
    train_ds = DS(split="train", **ds_kwargs, **common_kwargs)
    eval_ds = DS(split="eval", **ds_kwargs, **common_kwargs)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_fn, drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    trainable_params = [p for p in backbone.model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    if args.diagnostic:
        print("\n" + "=" * 60 + "\nDIAGNOSTIC MODE: 1 batch, purified path then clean path\n" + "=" * 60)
        batch = next(iter(train_loader))
        clean_audio = batch["waveform"].to(device)
        message = random_message(16, clean_audio.shape[0], device)
        with torch.no_grad():
            out = backbone.forward_full(clean_audio, message)
            recon_wm = out["recon_wm"]
            purified = denoiser(recon_wm)

        for label, wm_audio in [("purified", purified), ("clean", recon_wm)]:
            optimizer.zero_grad()
            wm_loss, acc = detect_acc_and_loss(backbone, wm_audio, message, need_grad=True)
            wm_loss.backward()
            grad_norm = sum(p.grad.norm().item() ** 2 for p in trainable_params if p.grad is not None) ** 0.5
            print(f"  [{label}] wm_loss={wm_loss.item():.4f} acc={acc:.4f} "
                  f"detector_lora_grad_norm={grad_norm:.4e}")
            if grad_norm == 0.0:
                print(f"  [train_stage3] WARNING: zero gradient reached detector's LoRA params on "
                      f"the {label} path -- do not trust a full run until this is nonzero.")
        print("\n[train_stage3] Diagnostic complete. If both grad norms above were nonzero and "
              "accuracies weren't already at floor/ceiling in a suspicious way, proceed to a full "
              "run (drop --diagnostic).")
        return

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    global_step = 0
    for epoch in range(args.epochs):
        backbone.model.eval()  # msg_processor/detector's frozen submodules stay in eval; LoRA
                                 # params still receive gradient regardless of this flag (see
                                 # module docstring -- no RNN backward-in-eval issue here).
        epoch_acc_purified, epoch_acc_clean = [], []

        for batch in train_loader:
            clean_audio = batch["waveform"].to(device)
            message = random_message(16, clean_audio.shape[0], device)

            with torch.no_grad():
                out = backbone.forward_full(clean_audio, message)
                recon_wm = out["recon_wm"]

            do_purify = random.random() < args.purify_prob
            if do_purify:
                with torch.no_grad():
                    wm_audio = denoiser(recon_wm)
            else:
                wm_audio = recon_wm

            optimizer.zero_grad()
            wm_loss, acc = detect_acc_and_loss(backbone, wm_audio, message, need_grad=True)
            wm_loss.backward()
            optimizer.step()

            (epoch_acc_purified if do_purify else epoch_acc_clean).append(acc)
            global_step += 1

            if global_step % args.log_every == 0:
                cond = "purified" if do_purify else "clean"
                print(f"[epoch {epoch} step {global_step}] cond={cond} wm_loss={wm_loss.item():.4f} acc={acc:.4f}")

            if global_step % args.eval_every == 0:
                acc_clean_val, acc_pur_val = validate(backbone, denoiser, eval_loader, device, args.n_eval_batches)
                print(f"  [validate @ step {global_step}] acc_clean={acc_clean_val:.4f} "
                      f"acc_purified={acc_pur_val:.4f}")
                backbone.model.eval()  # validate() doesn't change train/eval, but be explicit before resuming

        pur_str = f"{sum(epoch_acc_purified)/len(epoch_acc_purified):.4f}" if epoch_acc_purified else "n/a"
        clean_str = f"{sum(epoch_acc_clean)/len(epoch_acc_clean):.4f}" if epoch_acc_clean else "n/a"
        print(f"=== Epoch {epoch} complete. Train-batch avg ACC: purified={pur_str} clean={clean_str} ===")

        if (epoch + 1) % args.checkpoint_every == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(args.checkpoint_dir, f"stage3_epoch{epoch}.pt")
            # Save the FULL lora_state_dict (msg_processor's frozen delta included, unchanged from
            # the base checkpoint, plus detector's updated delta) -- keeps this checkpoint byte-
            # compatible with disruption_pgd.py / audiopure_eval.py's existing build_backbone()
            # functions, no format changes needed downstream.
            lora_state = {k: v for k, v in backbone.model.state_dict().items() if "_lora." in k}
            torch.save({
                "epoch": epoch, "lora_state_dict": lora_state,
                "base_checkpoint": args.stage1_checkpoint,
                "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
                "include_ffn": args.include_ffn, "capacity_lora_r": args.capacity_lora_r if args.include_ffn else None,
                "purify_prob": args.purify_prob, "reverse_timestep": args.reverse_timestep,
                "global_step": global_step,
            }, ckpt_path)
            print(f"[train_stage3] Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    train(args)
