"""
src/train_route2_clone_aware.py

ROUTE 2: train the watermark detector to recover the message from a CLONE,
by putting a real zero-shot voice cloner inside the training loop.

WHY THIS, AND WHY IT IS NOT A REPEAT OF STAGE 3
------------------------------------------------
VoiceMark's own paper is explicit that it never does this. Quoting it directly:
"During inference, it processes audio synthesized by zero-shot VC models, while
during training, it learns from augmented audio x~ without exposure to any VC
models." Their cloning robustness comes entirely from a proxy -- masking,
shuffling, replacing and neural-codec distortion (this repo reimplements all
four in src/data/augment.py). They report >95% ACC on cloned audio that way
(Table 1: CosyVoice 0.964, F5-TTS 0.979, MaskGCT 0.957).

So training against an ACTUAL differentiable cloner is precisely the thing
VoiceMark does not do, and is the concrete methodological difference this
script tests. It is also the audio analogue of Dual Defense (IEEE TIFS 2024),
which resolves the same watermark-vs-adversarial-disruption tension in face
swapping by putting the FaceSwap model in the training loop and adding an
explicit loss for decoding the watermark out of the DISRUPTED output.

Stage 3 of this project already tried detector-only fine-tuning and plateaued
at chance -- but against AudioPure, where the post-purification watermark ACC
was exactly at chance (~0.50). That is an information-theoretic dead end: if
the transformed audio carries zero mutual information about the message, no
amount of detector training can recover it. Cloning is a different situation,
and this is the load-bearing distinction: watermark ACC on an XTTS clone
measured 0.5587-0.5687 at n=100, p ~ 2e-8 against chance. A real, if faint,
channel demonstrably exists. This script exists to amplify a signal that has
been measured to be there, not to conjure one that is not.

WHAT IS TRAINABLE
-----------------
Default: the DETECTOR's LoRA only (the reader adapts). In this mode the clone
is generated under no_grad and detached, so nothing backpropagates through
YourTTS -- much faster and far more stable, and sufficient, because the
gradient the detector needs comes from its own forward pass on the clone.

--train_msg_processor additionally unfreezes the embedder's LoRA, so the
WRITER adapts too -- learning where to place bits that survive cloning, which
is the fuller Dual Defense recipe. That mode REQUIRES gradients to flow back
through the cloner, so the clone is generated with grad enabled (slower, more
memory). Known limitation inherited from surrogate_vc.py: gradients do not
flow through YourTTS's duration/rhythm path (a hard torch.ceil(), structural
to VITS), only through the flow/decoder conditioning.

THE LOSS
--------
    L = lambda_clean * Ldec(recon_wm)  +  lambda_clone * Ldec(clone(recon_wm))

The first term is not optional. Without it the detector is free to trade away
its clean-audio accuracy (currently ~0.99) for clone accuracy, which would be
a worse system overall -- attribution of the distributed protected file itself
matters as much as attribution of a clone. Clean ACC is printed every epoch
precisely so that regression is visible immediately, not discovered later.

Staged training, following Dual Defense's own scheduling (their decoder only
activates after epoch 30): --clone_warmup_epochs trains on the clean term
alone first, so the detector is not chasing a hard objective from a cold start.

EVALUATION DISCIPLINE
---------------------
Train against YourTTS (differentiable, in-loop). Evaluate on XTTS, which is
held out entirely -- a different architecture (GPT-autoregressive vs
VITS/flow), non-differentiable, never seen by training. Baseline to beat,
already measured on this exact eval: 0.5587-0.5687 watermark ACC on the XTTS
clone. Run src/eval/xtts_transfer_eval.py against the checkpoint this script
writes.

Note the run-to-run noise floor: two runs of the identical unprotected
condition returned 0.5687 and 0.5587, because XTTS decoding is stochastic.
Differences below ~0.02 on that eval are not interpretable.

BEFORE TRUSTING ANY RUN: --diagnostic. It verifies the cloner produces sane
output, the loss computes, gradients actually reach the trainable LoRA
parameters (a silent zero-gradient bug would train nothing while looking
perfectly healthy), and reports the pre-training baseline.

Usage:
    # 1. ALWAYS first -- seconds:
    python src/train_route2_clone_aware.py \\
        --checkpoint ./checkpoints/stage1_final_scaleup_recalibrated/recalibrated_final.pt \\
        --diagnostic

    # 2. Detector-only run (fast path -- start here):
    python src/train_route2_clone_aware.py \\
        --checkpoint ./checkpoints/stage1_final_scaleup_recalibrated/recalibrated_final.pt \\
        --epochs 20 --clone_warmup_epochs 2 \\
        --checkpoint_dir ./checkpoints/route2_clone_aware \\
        --output results/results_route2_training.json

    # 3. Then evaluate on the HELD-OUT cloner:
    python src/eval/xtts_transfer_eval.py \\
        --checkpoint ./checkpoints/route2_clone_aware/route2_final.pt \\
        --n_speakers 60 --n_eval_speakers 20 --eval_utterances_per_speaker 5 \\
        --output results/results_xtts_route2_n100.json
"""

import os
import sys
import json
import random
import re
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "losses"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "eval"))

from backbone import VoiceMarkBackbone
from adapters import apply_lora_adapters
from surrogate_vc import load_yourtts_surrogate
from librispeech import LibriSpeechSubset, collate_librispeech
from voicemark_losses import (compute_ldec, compute_lcos, compute_lvad,
                              MultiScaleMelLoss, compute_ladv_generator)
from augment_voicemark import add_augmentation_args, build_augmenter, samples_to_frames
from disruption_pgd import random_message, detect_acc


def build_trainable_backbone(checkpoint_path, lora_r, lora_alpha, train_msg_processor, msgproc_lora_r=None):
    """
    Loads the Stage-1 checkpoint, then unfreezes ONLY what this run trains.
    Everything else -- codec, base weights, and (by default) msg_processor --
    stays frozen, exactly as every other script in this project assumes.

    msgproc_lora_r (2026-09-14, quality-tradeoff patch): optional lower rank
    applied ONLY to msg_processor's LoRA adapters, independent of detector's
    rank (stays at lora_r). Motivated by the finding that neither --epochs nor
    --lambda_clone changed --train_msg_processor's PESQ/SIM cost at all (both
    landed statistically identical to the untouched run) -- pointing at LoRA
    capacity itself, not training duration or loss weighting, as the lever.
    """
    backbone = VoiceMarkBackbone()
    target_ranks = {"msg_processor": msgproc_lora_r} if msgproc_lora_r else None
    apply_lora_adapters(backbone, r=lora_r, alpha=lora_alpha, target_ranks=target_ranks)

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["lora_state_dict"]
        # 2026-09-14 msgproc_lora_r patch: the init checkpoint's LoRA tensors
        # were saved at their ORIGINAL rank (e.g. Stage 1's msg_processor LoRA
        # is rank 8). If msgproc_lora_r asks for a different rank, those specific
        # tensors can't be loaded -- LoRA A/B factors at different ranks aren't
        # reshapeable into each other (no clean truncation), so msg_processor's
        # adapters fall back to LoRA zero-init instead (the same starting point
        # any first --train_msg_processor run has before msg_processor is ever
        # trained). strict=False alone does NOT handle this -- it only tolerates
        # missing/extra key NAMES, not a shape mismatch on a key present in both
        # -- so mismatched keys are dropped explicitly first, loudly, before
        # load_state_dict runs.
        own_state = backbone.model.state_dict()
        skipped = [k for k in state_dict
                   if k in own_state and state_dict[k].shape != own_state[k].shape]
        if skipped:
            print(f"[build] msgproc_lora_r={msgproc_lora_r}: {len(skipped)} checkpoint "
                  f"tensor(s) don't match the new rank and will NOT be warm-started "
                  f"(LoRA zero-init instead):")
            for k in skipped:
                print(f"    {k}: checkpoint {tuple(state_dict[k].shape)} "
                      f"vs model {tuple(own_state[k].shape)}")
            state_dict = {k: v for k, v in state_dict.items() if k not in skipped}
        backbone.model.load_state_dict(state_dict, strict=False)
        print(f"[build] Loaded LoRA weights from {checkpoint_path} "
              f"(epoch {ckpt.get('epoch')}, train-time avg_acc={ckpt.get('avg_acc')})")
    else:
        print("[build] Starting from LoRA zero-init (== pretrained VoiceMark)")

    backbone.freeze_backbone()
    targets = ["detector"] + (["msg_processor"] if train_msg_processor else [])

    # NOTE: deliberately NOT backbone.unfreeze(targets) -- that unfreezes EVERY
    # parameter in those submodules, including the frozen pretrained VoiceMark
    # weights (its own docstring says it is for full fine-tuning / ablations).
    # This project's entire premise is that only LoRA adapters ever train, so
    # unfreeze by parameter name instead: LoRA params only, in the target
    # submodules only.
    trainable, names = [], []
    for name, prm in backbone.model.named_parameters():
        if "_lora." in name and any(name.startswith(t) or f".{t}." in name for t in targets):
            prm.requires_grad = True
            trainable.append(prm)
            names.append(name)

    if not trainable:
        raise RuntimeError(
            f"No LoRA parameters matched targets {targets}. Parameter naming may have changed -- "
            f"expected names containing '_lora.' under a '{targets[0]}' submodule. "
            f"Refusing to run: training would silently update nothing."
        )

    # Belt and braces: nothing OUTSIDE the LoRA adapters may be trainable, or we
    # would be fine-tuning frozen base weights and invalidating every comparison
    # against the Stage-1 baseline.
    leaked = [n for n, prm in backbone.model.named_parameters()
              if prm.requires_grad and "_lora." not in n]
    if leaked:
        raise RuntimeError(
            f"{len(leaked)} non-LoRA parameter(s) are trainable, e.g. {leaked[:3]}. "
            f"Refusing to run: this would fine-tune frozen base weights."
        )

    n_params = sum(p.numel() for p in trainable)
    print(f"[build] Trainable: {targets} LoRA only -- {len(trainable)} tensors, {n_params:,} params")
    print(f"[build] e.g. {names[0]}")
    return backbone, trainable


def decode_logits(backbone, audio):
    """Run the detector on a waveform, returning its chunk logits."""
    feat = backbone.model.st_model.forward_feature(audio)
    _logits, chunk_logits = backbone.model.detector(feat)
    return chunk_logits


def decode_both(backbone, audio):
    """Detector on a waveform -> (per-frame presence logits, chunk logits)."""
    feat = backbone.model.st_model.forward_feature(audio)
    return backbone.model.detector(feat)


class ExtraLosses:
    """
    2026-10-24 additions, all OFF by default (every weight 0), so a run without
    the new flags reproduces the old Route 2 objective exactly.

    Quality terms (VoiceMark's own, on the watermarked audio vs the clean
    original; only meaningful with --train_msg_processor, because only the
    embedder changes the audio):
        lambda_mel * Lmel(recon_wm, clean)   multi-scale mel, as in Stage 1
        lambda_cos * Lcos(acoustic, acoustic_wm)
        lambda_adv * (Ladv + Lfm)            frozen pretrained msstftd discriminator
    Robustness terms (VoiceMark's VC-simulated augmentation, in addition to the
    real YourTTS clone):
        lambda_aug * Ldec(augment(recon_wm))
        lambda_vad * Lvad on the same augmented audio
    Paper weights for reference: mel 2, cos 2, adv 1, vad 1, dec 1.
    """

    def __init__(self, args, device):
        self.args = args
        self.mel = MultiScaleMelLoss(sample_rate=16000).to(device) if args.lambda_mel > 0 else None
        self.disc = None
        if args.lambda_adv > 0:
            from backbone import VoiceMarkDiscriminator
            self.disc = VoiceMarkDiscriminator(device=device)
            for prm in self.disc.parameters():
                prm.requires_grad_(False)
            self.disc.eval()
        self.augmenter = None
        if args.lambda_aug > 0 or args.lambda_vad > 0:
            self.augmenter = build_augmenter(args, sample_rate=16000)
        self.active = any(getattr(args, k) > 0 for k in
                          ("lambda_mel", "lambda_cos", "lambda_adv", "lambda_aug", "lambda_vad"))

    def __call__(self, backbone, out, clean_audio, message):
        a = self.args
        terms, total = {}, 0.0
        recon_wm = out["recon_wm"]
        if a.lambda_mel > 0:
            terms["lmel"] = self.mel(recon_wm, clean_audio)
            total = total + a.lambda_mel * terms["lmel"]
        if a.lambda_cos > 0:
            terms["lcos"] = compute_lcos(out["acoustic"], out["acoustic_wm"])
            total = total + a.lambda_cos * terms["lcos"]
        if a.lambda_adv > 0:
            adv, fm = compute_ladv_generator(self.disc, recon_wm, clean_audio)
            terms["ladv"], terms["lfm"] = adv, fm
            total = total + a.lambda_adv * (adv + fm)
        if self.augmenter is not None:
            aug = self.augmenter(recon_wm, clean_audio)
            presence_logits, chunk_logits = decode_both(backbone, aug["audio"])
            if a.lambda_aug > 0:
                terms["ldec_aug"] = compute_ldec(chunk_logits, message)
                total = total + a.lambda_aug * terms["ldec_aug"]
            if a.lambda_vad > 0:
                mask = samples_to_frames(aug["nowm_mask"], presence_logits.shape[1])
                terms["lvad"] = compute_lvad(presence_logits, aug["vad_ref"], 16000, mask)
                total = total + a.lambda_vad * terms["lvad"]
        return total, terms


def normalize_text(t):
    """Same normalisation as src/eval/clone_eval_v2.py (keep the two identical)."""
    t = " ".join(str(t).strip().split())
    if not t:
        return t
    if t.isupper():
        t = t.lower()
    t = t[0].upper() + t[1:]
    if t[-1] not in ".?!":
        t += "."
    return t


def build_train_text_pool(train_ds, min_words, max_words):
    """
    2026-10-24 text protocol. Clone texts for training are transcripts of the
    TRAINING utterances only (LibriSpeech train split of this subset), 6-20 words,
    so no evaluation text (held-out speakers) can be seen in training.
    """
    pool = set()
    for idx in train_ds._subset_indices:
        fileid = train_ds._full_dataset._walker[idx]
        spk, chap, utt = fileid.split("-")
        t = normalize_text(train_ds._get_transcript(spk, chap, utt))
        if min_words <= len(re.findall(r"[a-z0-9']+", t.lower())) <= max_words:
            pool.add(t)
    return sorted(pool)


def make_clone(surrogate, audio, text, differentiable):
    """
    Generate a clone. When only the DETECTOR trains, no gradient needs to flow
    back through the cloner -- the clone is just input data -- so it is
    generated under no_grad and detached. That is dramatically faster and
    avoids holding YourTTS's graph in memory for every batch.
    """
    if differentiable:
        return surrogate.clone_voice(audio, text=text)
    with torch.no_grad():
        return surrogate.clone_voice(audio, text=text).detach()


def evaluate(backbone, surrogate, eval_loader, device, text, n_batches, differentiable=False):
    """Clean ACC (must not regress) and clone ACC (the target metric)."""
    backbone.model.eval()
    accs_clean, accs_clone = [], []
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= n_batches:
                break
            clean_audio = batch["waveform"].to(device)
            message = random_message(16, clean_audio.shape[0], device, seed=9000 + i)
            out = backbone.forward_full(clean_audio, message)
            recon_wm = out["recon_wm"]
            accs_clean.append(detect_acc(backbone, recon_wm, message))
            cloned = make_clone(surrogate, recon_wm, text, differentiable=False)
            accs_clone.append(detect_acc(backbone, cloned, message))
    backbone.model.train()
    return (sum(accs_clean) / len(accs_clean), sum(accs_clone) / len(accs_clone))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Stage-1 checkpoint to start from. Which one matters a lot -- see "
                        "the note about stage1_aug vs stage1_final_scaleup_recalibrated.")
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints/route2_clone_aware")
    p.add_argument("--output", type=str, default=None, help="Per-epoch metrics JSON.")
    p.add_argument("--diagnostic", action="store_true",
                   help="One batch, verbose, with gradient-flow verification. Always run first.")

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-5, help="Matches the paper's stated Adam lr.")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lambda_clean", type=float, default=1.0,
                   help="Weight on decoding from clean watermarked audio. Do NOT set to 0 -- "
                        "clean ACC (~0.99) is a real capability and must not be traded away.")
    p.add_argument("--lambda_clone", type=float, default=1.0,
                   help="Weight on decoding from the CLONE. This is the new objective.")
    p.add_argument("--clone_warmup_epochs", type=int, default=2,
                   help="Epochs of clean-only training before the clone term switches on "
                        "(Dual Defense stages its decoder activation the same way).")
    p.add_argument("--train_msg_processor", action="store_true",
                   help="Also train the EMBEDDER, so bit placement adapts to survive cloning "
                        "(fuller Dual Defense recipe). Requires backprop through the cloner: "
                        "slower and more memory.")

    p.add_argument("--surrogate_text", type=str, default="This is a test sentence for voice cloning.",
                   help="Clone text when --clone_text fixed (the pre-2026-10-24 protocol), and the "
                        "text of the in-loop monitoring eval in every mode.")
    p.add_argument("--clone_text", choices=("fixed", "sampled"), default="fixed",
                   help="'sampled' (2026-10-24): each training step clones with a different "
                        "sentence drawn from the TRAINING transcripts, so the watermark cannot "
                        "specialise to one sentence. 'fixed' reproduces the old runs.")
    p.add_argument("--text_min_words", type=int, default=6)
    p.add_argument("--text_max_words", type=int, default=14,
                   help="14 keeps the differentiable YourTTS clone (backprop through it) within T4 memory.")
    p.add_argument("--seed", type=int, default=None,
                   help="Fixes data order, messages, clone-text order and augmentation draws. "
                        "None reproduces the old (unseeded) behaviour. Seed 0 uses the same "
                        "message seeds as the old runs.")
    p.add_argument("--eval_batches", type=int, default=10)
    p.add_argument("--data_root", type=str, default="./data/librispeech")
    p.add_argument("--n_speakers", type=int, default=60)
    p.add_argument("--utterances_per_speaker", type=int, default=15)
    p.add_argument("--n_eval_speakers", type=int, default=20)
    p.add_argument("--eval_utterances_per_speaker", type=int, default=5)
    p.add_argument("--crop_seconds", type=float, default=3.0)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--msgproc_lora_r", type=int, default=None,
                    help="Lower LoRA rank for msg_processor ONLY (detector stays at "
                         "--lora_r). Quality-tradeoff test: --epochs and --lambda_clone "
                         "both failed to change --train_msg_processor's PESQ/SIM cost at "
                         "all, suggesting the cost comes from LoRA capacity itself, not "
                         "training duration or loss weighting. Only takes effect with "
                         "--train_msg_processor. Try e.g. 2 or 4 (full rank default is 8). "
                         "MUST be passed again, matching, to any eval script loading the "
                         "resulting checkpoint (e.g. xtts_transfer_eval.py --msgproc_lora_r).")
    # 2026-10-24: VoiceMark quality losses and augmentation (see ExtraLosses).
    p.add_argument("--lambda_mel", type=float, default=0.0,
                   help="VoiceMark multi-scale mel loss on recon_wm vs clean (paper weight 2).")
    p.add_argument("--lambda_cos", type=float, default=0.0,
                   help="VoiceMark latent cosine loss (paper weight 2).")
    p.add_argument("--lambda_adv", type=float, default=0.0,
                   help="VoiceMark adversarial + feature-matching loss, frozen discriminator (paper weight 1).")
    p.add_argument("--lambda_aug", type=float, default=0.0,
                   help="Ldec on VoiceMark-augmented recon_wm (augment_voicemark.py).")
    p.add_argument("--lambda_vad", type=float, default=0.0,
                   help="Lvad on the same augmented audio (paper weight 1).")
    add_augmentation_args(p)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    differentiable_clone = args.train_msg_processor
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    msg_seed_offset = 0 if args.seed is None else args.seed * 1_000_000

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
    # own generator for the shuffle, so extra random draws by the quality/augmentation
    # terms cannot change the data order between arms that share a seed
    loader_gen = torch.Generator().manual_seed(args.seed) if args.seed is not None else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_librispeech, drop_last=True, generator=loader_gen)

    text_pool, text_rng = None, random.Random(0 if args.seed is None else args.seed)
    if args.clone_text == "sampled":
        text_pool = build_train_text_pool(train_ds, args.text_min_words, args.text_max_words)
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        with open(os.path.join(args.checkpoint_dir, "clone_text_pool.json"), "w") as f:
            json.dump(text_pool, f, indent=0)
        print(f"[main] clone text: sampled per step from {len(text_pool)} training transcripts "
              f"({args.text_min_words}-{args.text_max_words} words); pool saved to "
              f"{args.checkpoint_dir}/clone_text_pool.json")
    else:
        print(f"[main] clone text: fixed sentence {args.surrogate_text!r}")

    def step_text():
        return text_rng.choice(text_pool) if text_pool else args.surrogate_text
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_librispeech)

    backbone, trainable = build_trainable_backbone(
        args.checkpoint, args.lora_r, args.lora_alpha, args.train_msg_processor,
        msgproc_lora_r=args.msgproc_lora_r)
    backbone.model.to(device)
    optimizer = torch.optim.Adam(trainable, lr=args.lr)

    extra = ExtraLosses(args, device)
    quality_on = args.lambda_mel > 0 or args.lambda_cos > 0 or args.lambda_adv > 0
    if quality_on and not args.train_msg_processor:
        print("[main] WARNING: quality losses have no effect without --train_msg_processor "
              "(the detector does not change the audio).")

    print("[main] Loading YourTTS surrogate (the in-loop cloner; frozen)...")
    surrogate = load_yourtts_surrogate(device=device)

    # ---------------- diagnostic ----------------
    if args.diagnostic:
        print("\n" + "=" * 60 + "\nDIAGNOSTIC MODE\n" + "=" * 60)
        batch = next(iter(train_loader))
        clean_audio = batch["waveform"].to(device)
        message = random_message(16, clean_audio.shape[0], device, seed=123)

        out = backbone.forward_full(clean_audio, message)
        recon_wm = out["recon_wm"]
        print(f"\n--- 1. Shapes ---")
        print(f"  clean_audio={tuple(clean_audio.shape)}  recon_wm={tuple(recon_wm.shape)}")

        diag_text = step_text()
        cloned = make_clone(surrogate, recon_wm, diag_text, differentiable_clone)
        print(f"  clone={tuple(cloned.shape)}  (differentiable={differentiable_clone})  text={diag_text!r}")

        acc_clean = detect_acc(backbone, recon_wm, message)
        acc_clone = detect_acc(backbone, cloned, message)
        print(f"\n--- 2. Pre-training baseline (this run must improve on the second number) ---")
        print(f"  ACC on clean watermarked audio: {acc_clean:.4f}  (expect ~0.99; must NOT regress)")
        print(f"  ACC on the clone:               {acc_clone:.4f}  (expect ~0.5-0.6 -- the target)")

        loss_clean = compute_ldec(decode_logits(backbone, recon_wm), message)
        loss_clone = compute_ldec(decode_logits(backbone, cloned), message)
        loss = args.lambda_clean * loss_clean + args.lambda_clone * loss_clone
        extra_total, extra_terms = extra(backbone, out, clean_audio, message) if extra.active else (0.0, {})
        loss = loss + extra_total
        print(f"\n--- 3. Loss ---")
        print(f"  Ldec(clean)={loss_clean.item():.4f}  Ldec(clone)={loss_clone.item():.4f}  total={loss.item():.4f}")
        if extra_terms:
            print("  extra terms: " + "  ".join(f"{k}={v.item():.4f}" for k, v in extra_terms.items()))

        optimizer.zero_grad()
        loss.backward()
        n_with_grad = sum(1 for prm in trainable if prm.grad is not None and prm.grad.abs().sum() > 0)
        total_gn = sum(prm.grad.norm().item() for prm in trainable if prm.grad is not None)
        print(f"\n--- 4. Gradient flow (the silent-failure check) ---")
        print(f"  {n_with_grad}/{len(trainable)} trainable tensors received a NONZERO gradient")
        print(f"  total grad norm = {total_gn:.6f}")
        if n_with_grad == 0 or total_gn == 0.0:
            print("  *** STOP: no gradient reaches the trainable parameters. Training would run "
                  "for hours and change nothing. Fix this before any real run. ***")
        else:
            print("  -> gradients are flowing; training will actually update the detector.")

        print("\n[main] Diagnostic complete. If gradients flowed and the baseline numbers look "
              "right, drop --diagnostic and start a real run.")
        return

    # ---------------- training ----------------
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    history = []

    print(f"\n{'=' * 70}")
    print(f"ROUTE 2: clone-aware detector training")
    print(f"  trainable      : detector{' + msg_processor' if args.train_msg_processor else ''} (LoRA)")
    print(f"  in-loop cloner : YourTTS (differentiable={differentiable_clone})")
    print(f"  held-out eval  : XTTS, baseline to beat = 0.5587-0.5687 (noise floor ~0.02)")
    print(f"  loss           : {args.lambda_clean} * Ldec(clean) + {args.lambda_clone} * Ldec(clone)"
          + (f" + mel {args.lambda_mel} + cos {args.lambda_cos} + adv {args.lambda_adv}"
             f" + aug {args.lambda_aug} + vad {args.lambda_vad}" if extra.active else ""))
    print(f"  clone term on  : from epoch {args.clone_warmup_epochs}")
    print(f"{'=' * 70}")

    acc_clean0, acc_clone0 = evaluate(backbone, surrogate, eval_loader, device,
                                      args.surrogate_text, args.eval_batches)
    print(f"\n[epoch -1  PRE-TRAINING] clean_acc={acc_clean0:.4f}  clone_acc={acc_clone0:.4f}")

    for epoch in range(args.epochs):
        backbone.model.train()
        clone_on = epoch >= args.clone_warmup_epochs
        sums = {"loss": 0.0, "clean": 0.0, "clone": 0.0, "n": 0}
        extra_sums = {}

        for step, batch in enumerate(train_loader):
            clean_audio = batch["waveform"].to(device)
            message = random_message(16, clean_audio.shape[0], device,
                                     seed=msg_seed_offset + epoch * 10000 + step)

            out = backbone.forward_full(clean_audio, message)
            recon_wm = out["recon_wm"]

            loss_clean = compute_ldec(decode_logits(backbone, recon_wm), message)
            loss = args.lambda_clean * loss_clean
            loss_clone_val = float("nan")

            if clone_on:
                cloned = make_clone(surrogate, recon_wm, step_text(), differentiable_clone)
                loss_clone = compute_ldec(decode_logits(backbone, cloned), message)
                loss = loss + args.lambda_clone * loss_clone
                loss_clone_val = loss_clone.item()

            extra_terms = {}
            if extra.active:
                extra_total, extra_terms = extra(backbone, out, clean_audio, message)
                loss = loss + extra_total
                for k, v in extra_terms.items():
                    extra_sums[k] = extra_sums.get(k, 0.0) + v.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sums["loss"] += loss.item()
            sums["clean"] += loss_clean.item()
            if clone_on:
                sums["clone"] += loss_clone_val
            sums["n"] += 1

            if step % 20 == 0:
                msg = (f"  [e{epoch} s{step}] loss={loss.item():.4f} "
                       f"Ldec_clean={loss_clean.item():.4f}")
                if clone_on:
                    msg += f" Ldec_clone={loss_clone_val:.4f}"
                else:
                    msg += "  (clone term off -- warmup)"
                if extra_terms:
                    msg += " " + " ".join(f"{k}={v.item():.4f}" for k, v in extra_terms.items())
                print(msg, flush=True)

        acc_clean, acc_clone = evaluate(backbone, surrogate, eval_loader, device,
                                        args.surrogate_text, args.eval_batches)
        n = max(sums["n"], 1)
        rec = {
            "epoch": epoch,
            "clone_term_active": clone_on,
            "mean_loss": sums["loss"] / n,
            "mean_ldec_clean": sums["clean"] / n,
            "mean_ldec_clone": (sums["clone"] / n) if clone_on else None,
            "eval_clean_acc": acc_clean,
            "eval_clone_acc": acc_clone,
            "mean_extra_terms": {k: v / n for k, v in extra_sums.items()},
        }
        history.append(rec)

        flag = ""
        if acc_clean < 0.95:
            flag = "  *** WARNING: clean ACC has regressed below 0.95 -- the detector is trading " \
                   "away its clean-audio capability. Lower --lambda_clone or raise --lambda_clean. ***"
        print(f"\n[epoch {epoch}] clean_acc={acc_clean:.4f}  clone_acc={acc_clone:.4f}  "
              f"(pre-training was {acc_clean0:.4f} / {acc_clone0:.4f}){flag}\n")

        # Save every epoch -- a Kaggle timeout or crash must never cost a run.
        lora_state = {k: v for k, v in backbone.model.state_dict().items() if "_lora." in k}
        ckpt_path = os.path.join(args.checkpoint_dir, f"route2_epoch{epoch}.pt")
        torch.save({
            "epoch": epoch, "lora_state_dict": lora_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
            "avg_acc": acc_clean, "eval_clone_acc": acc_clone,
            "trained_msg_processor": args.train_msg_processor,
            "train_args": vars(args),
        }, ckpt_path)
        torch.save({
            "epoch": epoch, "lora_state_dict": lora_state,
            "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
            "avg_acc": acc_clean, "eval_clone_acc": acc_clone,
            "trained_msg_processor": args.train_msg_processor,
            "train_args": vars(args),
        }, os.path.join(args.checkpoint_dir, "route2_final.pt"))

        if args.output:
            with open(args.output, "w") as f:
                json.dump({
                    "label": "route2_clone_aware",
                    "checkpoint_init": args.checkpoint,
                    "trained_msg_processor": args.train_msg_processor,
                    "lambda_clean": args.lambda_clean, "lambda_clone": args.lambda_clone,
                    "train_args": vars(args),
                    "clone_warmup_epochs": args.clone_warmup_epochs,
                    "pre_training": {"clean_acc": acc_clean0, "clone_acc": acc_clone0},
                    "xtts_baseline_to_beat": 0.5687,
                    "history": history,
                }, f, indent=2)

    best = max(history, key=lambda r: r["eval_clone_acc"])
    print(f"\n{'=' * 70}")
    print(f"DONE. Best epoch {best['epoch']}: clone_acc={best['eval_clone_acc']:.4f} "
          f"(pre-training {acc_clone0:.4f}), clean_acc={best['eval_clean_acc']:.4f}")
    print(f"These are YourTTS numbers -- the cloner that was IN the training loop, so they are "
          f"optimistic by construction. The real result is the held-out XTTS eval:")
    print(f"  python src/eval/xtts_transfer_eval.py --checkpoint "
          f"{os.path.join(args.checkpoint_dir, 'route2_final.pt')} \\")
    print(f"    --n_speakers 60 --n_eval_speakers 20 --eval_utterances_per_speaker 5 \\")
    print(f"    --output results/results_xtts_route2_n100.json")
    print(f"Beat 0.5687 there by more than the ~0.02 noise floor and it is a real improvement.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
