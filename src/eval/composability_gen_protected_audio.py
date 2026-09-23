"""
src/eval/composability_gen_protected_audio.py

STEP A of the CosyVoice/MaskGCT/F5-TTS composability test (2026-09-15).

WHY THIS SCRIPT EXISTS
-----------------------
The existing composability result (README Stage 4) only covers XTTS-v2:
xtts_transfer_eval.py both PGD-protects the audio AND clones it, because XTTS
ships in the same coqui-tts package the YourTTS surrogate already needs, so
everything happens in one environment. CosyVoice, MaskGCT and F5-TTS do NOT
share an environment with coqui-tts (that's the whole reason
cloner_watermark_eval.py imports nothing from disruption_pgd/surrogate_vc --
see its own module docstring) or with EACH OTHER. PGD itself, though, only
needs the differentiable YourTTS surrogate -- it never touches the cloner
that will eventually consume its output. So the composability test splits
into two steps that can each stay inside one environment:

  STEP A (this script, runs in the SAME coqui-tts/YourTTS-surrogate env as
          disruption_pgd.py / xtts_transfer_eval.py / demucs_fallback_eval.py):
          watermark n utterances, PGD-protect them (same eps=0.002, n_steps=10,
          lambda_wm=1.0 operating point Stage 2/3/the XTTS-v2 composability
          result already use), and SAVE the WAVs to disk. No cloning here.

  STEP B (cloner_watermark_eval.py, unchanged, run separately in each of the
          CosyVoice/MaskGCT/F5-TTS environments via its existing
          --input_wav_dir / --wav_suffix pre-made-audio mode): clone the saved
          WAVs for real and measure watermark ACC on each clone.

  STEP C (ecapa_sim_eval.py, unchanged, runs anywhere speechbrain is
          installed -- does not need coqui-tts OR any cloner installed):
          speaker-similarity scoring of the clones against the TRUE original
          reference this script also saves.

This script deliberately does NOT import anything from cloner_watermark_eval.py
or vice versa -- same environment-isolation discipline as everywhere else in
this project. It reuses disruption_pgd.py's build_backbone/pgd_perturb/
detect_acc EXACTLY as demucs_fallback_eval.py and xtts_transfer_eval.py already
do, so this is not a new mechanism, just a new (and narrower) driver over it.

WHAT GETS WRITTEN, per utterance i (i = 0 .. n-1), all directly under --save_dir
(NOT a subdirectory -- cloner_watermark_eval.py's --input_wav_dir globs
"sample*_{wav_suffix}.wav" directly inside the directory you point it at):

    sample{i}_reference.wav      the TRUE clean original (no watermark, no PGD)
                                  -- kept for Step C's SIM scoring against the
                                  real speaker identity, NOT against the
                                  watermarked input (that would answer a
                                  different, less interesting question).
    sample{i}_unprotected.wav    recon_wm: watermarked, NO PGD.
                                  --wav_suffix unprotected in Step B --
                                  the "does the gain survive on this cloner at
                                  all, before adding disruption" arm.
    sample{i}_protected.wav      recon_wm + PGD delta: watermarked AND
                                  protected. --wav_suffix protected in Step B --
                                  the actual composability question: does
                                  Stage 2/3's anti-cloning perturbation still
                                  transfer to this architecture, and does the
                                  watermark still survive it.

Message seed convention is seed=123+i, matching cloner_watermark_eval.py's
default --message_seed_base 123 EXACTLY -- Step B regenerates the same message
from that seed rather than reading it from the WAV, so a mismatch here would
silently produce chance-level ACC in Step B (its own guard aborts on utterance
0 if acc_source < 0.85, which would catch it, but better not to rely on that).

Dataset args default to the SAME LibriSpeechSubset config
cloner_watermark_eval.py already uses for the cross-cloner generalization
results (n_speakers=60, utterances_per_speaker=15, n_eval_speakers=20,
eval_utterances_per_speaker=5 -> n=100) so this draws the SAME held-out
utterances already used for the F5-TTS/CosyVoice/MaskGCT cross-cloner numbers
in the README -- directly comparable, not a different sample.

Usage:
    # 1. ALWAYS first -- 1 utterance, verbose, no files written:
    python src/eval/composability_gen_protected_audio.py \\
        --checkpoint ./checkpoints/route2_msgproc_r2/route2_final.pt \\
        --msgproc_lora_r 2 --diagnostic

    # 2. Full run (n=100, same operating point as the XTTS-v2 composability
    #    result already in the README):
    python src/eval/composability_gen_protected_audio.py \\
        --checkpoint ./checkpoints/route2_msgproc_r2/route2_final.pt \\
        --msgproc_lora_r 2 --epsilon 0.002 --n_steps 10 --lambda_wm 1.0 \\
        --n_utterances 100 --save_dir ./composability_audio

Then zip/upload ./composability_audio (or the relevant Kaggle output) into
each of the CosyVoice/MaskGCT/F5-TTS environments for Step B.
"""

import os
import sys
import json
import argparse
import torch
import soundfile as sf
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "losses"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from disruption_pgd import build_backbone, random_message, detect_acc, pgd_perturb
from surrogate_vc import load_yourtts_surrogate
from librispeech import LibriSpeechSubset, collate_librispeech


def _write(save_dir, tag, wav):
    os.makedirs(save_dir, exist_ok=True)
    sf.write(os.path.join(save_dir, f"{tag}.wav"),
             wav.detach().cpu().reshape(-1).numpy(), 16000)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Route 2 checkpoint. Omit only to test the pretrained-"
                        "VoiceMark composability baseline (zero-init LoRA).")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--msgproc_lora_r", type=int, default=None,
                   help="MUST match the value used at training time (2 for the "
                        "rank-2 checkpoint this composability test targets) or "
                        "load_state_dict hits a shape mismatch on msg_processor's "
                        "LoRA tensors. See build_backbone's own warning.")

    p.add_argument("--epsilon", type=float, default=0.002,
                   help="Stage 2's original operating point, same as the "
                        "existing XTTS-v2 composability result -- keep this "
                        "fixed across cloners so the results are comparable.")
    p.add_argument("--step_size", type=float, default=None)
    p.add_argument("--n_steps", type=int, default=10)
    p.add_argument("--lambda_wm", type=float, default=1.0)
    p.add_argument("--random_start", action="store_true", default=True)
    p.add_argument("--surrogate_text", type=str,
                   default="This is a test sentence for voice cloning.")

    p.add_argument("--diagnostic", action="store_true",
                   help="1 utterance, verbose, writes nothing. Run this FIRST.")
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--n_utterances", type=int, default=100)

    p.add_argument("--data_root", type=str, default="./data/librispeech")
    p.add_argument("--n_speakers", type=int, default=60)
    p.add_argument("--utterances_per_speaker", type=int, default=15)
    p.add_argument("--n_eval_speakers", type=int, default=20)
    p.add_argument("--eval_utterances_per_speaker", type=int, default=5)
    p.add_argument("--crop_seconds", type=float, default=3.0)
    args = p.parse_args()

    if not args.diagnostic and args.save_dir is None:
        p.error("--save_dir is required unless --diagnostic is set")

    step_size = args.step_size if args.step_size is not None else args.epsilon / 4.0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    eval_ds = LibriSpeechSubset(
        root=args.data_root, n_speakers=args.n_speakers,
        utterances_per_speaker=args.utterances_per_speaker,
        n_eval_speakers=args.n_eval_speakers,
        eval_utterances_per_speaker=args.eval_utterances_per_speaker,
        sample_rate=16000, crop_seconds=args.crop_seconds, split="eval",
    )
    eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False,
                              collate_fn=collate_librispeech, drop_last=False)

    backbone = build_backbone(args.checkpoint, args.lora_r, args.lora_alpha,
                               include_ffn=False, capacity_lora_r=32,
                               msgproc_lora_r=args.msgproc_lora_r)
    backbone.model.to(device)
    print("[main] Loading YourTTS surrogate (PGD objective only -- no cloning "
          "happens in this script)...")
    surrogate = load_yourtts_surrogate(device=device)

    if args.diagnostic:
        print("\n" + "=" * 60 + "\nDIAGNOSTIC MODE: 1 utterance, nothing written\n" + "=" * 60)
        batch = next(iter(eval_loader))
        clean_audio = batch["waveform"].to(device)
        message = random_message(16, clean_audio.shape[0], device, seed=123)
        recon_wm, perturbed, _delta = pgd_perturb(
            backbone, surrogate, clean_audio, message, args.surrogate_text,
            args.epsilon, step_size, args.n_steps, args.random_start,
            lambda_wm=args.lambda_wm, verbose=True,
        )
        acc_u = detect_acc(backbone, recon_wm, message)
        acc_p = detect_acc(backbone, perturbed, message)
        print(f"\nshapes: clean={tuple(clean_audio.shape)} "
              f"unprotected={tuple(recon_wm.shape)} protected={tuple(perturbed.shape)}")
        print(f"ACC unprotected (recon_wm): {acc_u:.4f}  (expect ~0.99)")
        print(f"ACC protected  (+ PGD)   : {acc_p:.4f}  (expect close to unprotected -- "
              f"lambda_wm={args.lambda_wm} exists specifically to keep this from collapsing)")
        print("\n[main] Diagnostic complete. If both ACC values look sane, drop "
              "--diagnostic and add --save_dir for a full run.")
        return

    accs_u, accs_p = [], []
    print(f"\n{'=' * 78}")
    print(f"COMPOSABILITY AUDIO GENERATION | eps={args.epsilon} n_steps={args.n_steps} "
          f"lambda_wm={args.lambda_wm} | n={args.n_utterances} -> {args.save_dir}")
    print(f"{'=' * 78}")

    for i, batch in enumerate(eval_loader):
        if i >= args.n_utterances:
            break
        clean_audio = batch["waveform"].to(device)
        message = random_message(16, clean_audio.shape[0], device, seed=123 + i)

        recon_wm, perturbed, _delta = pgd_perturb(
            backbone, surrogate, clean_audio, message, args.surrogate_text,
            args.epsilon, step_size, args.n_steps, args.random_start,
            lambda_wm=args.lambda_wm,
        )
        acc_u = detect_acc(backbone, recon_wm, message)
        acc_p = detect_acc(backbone, perturbed, message)
        accs_u.append(acc_u)
        accs_p.append(acc_p)

        _write(args.save_dir, f"sample{i}_reference", clean_audio[0])
        _write(args.save_dir, f"sample{i}_unprotected", recon_wm[0])
        _write(args.save_dir, f"sample{i}_protected", perturbed[0])

        print(f"  [{i}] acc_unprotected={acc_u:.4f}  acc_protected={acc_p:.4f}", flush=True)

        # SEED GUARD, mirroring cloner_watermark_eval.py's own convention: both
        # arms are watermarked, so acc should be ~0.99 on utterance 0. If it
        # isn't, something about the checkpoint/msgproc_lora_r is wrong -- catch
        # it here, not after Step B silently produces chance-level numbers.
        if i == 0 and acc_u < 0.85:
            raise SystemExit(
                f"\nABORT: acc_unprotected={acc_u:.4f} on utterance 0 (expected ~0.99). "
                f"Check --checkpoint / --msgproc_lora_r match the Route 2 rank-2 training run.")

        if (i + 1) % 10 == 0:
            manifest = {
                "label": "composability_audio_gen",
                "checkpoint": args.checkpoint,
                "msgproc_lora_r": args.msgproc_lora_r,
                "pgd": {"epsilon": args.epsilon, "step_size": step_size,
                        "n_steps": args.n_steps, "lambda_wm": args.lambda_wm},
                "n_completed": i + 1,
                "sanity_acc_unprotected_mean": sum(accs_u) / len(accs_u),
                "sanity_acc_protected_mean": sum(accs_p) / len(accs_p),
            }
            with open(os.path.join(args.save_dir, "_manifest.json"), "w") as f:
                json.dump(manifest, f, indent=2)

    manifest = {
        "label": "composability_audio_gen",
        "checkpoint": args.checkpoint,
        "msgproc_lora_r": args.msgproc_lora_r,
        "pgd": {"epsilon": args.epsilon, "step_size": step_size,
                "n_steps": args.n_steps, "lambda_wm": args.lambda_wm},
        "n_completed": len(accs_u),
        "sanity_acc_unprotected_mean": sum(accs_u) / len(accs_u),
        "sanity_acc_protected_mean": sum(accs_p) / len(accs_p),
    }
    with open(os.path.join(args.save_dir, "_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'=' * 78}")
    print(f"DONE. n={len(accs_u)} triplets written to {args.save_dir}")
    print(f"  mean ACC unprotected (sanity, on the audio itself): "
          f"{manifest['sanity_acc_unprotected_mean']:.4f}")
    print(f"  mean ACC protected   (sanity, on the audio itself): "
          f"{manifest['sanity_acc_protected_mean']:.4f}")
    print(f"{'=' * 78}")
    print(f"\nNext: copy/upload {args.save_dir} into each of the CosyVoice/MaskGCT/F5-TTS")
    print(f"environments and run cloner_watermark_eval.py --input_wav_dir {args.save_dir} "
          f"--wav_suffix unprotected|protected there (see the composability run commands).")


if __name__ == "__main__":
    main()
