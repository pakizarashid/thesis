"""
src/eval/xtts_transfer_eval.py

Answers the fork-in-the-road question from watermark_survival_under_cloning.py's
result: does the watermark's near-chance collapse under real zero-shot cloning
generalize, or is it specific to YourTTS's particular conditioning mechanism
(pure fixed-speaker-embedding conditioning, full waveform regeneration from
text, no reuse of the input audio's actual samples)?

XTTS-v2 is a genuinely different architecture: GPT-style autoregressive token
generation with audio-prompt conditioning, not YourTTS's VITS/flow design. It
ships in the SAME `coqui-tts` package YourTTS already comes from -- no new
dependency, no new model download infrastructure to build. Critically, THIS
SCRIPT DOES NOT NEED XTTS TO BE DIFFERENTIABLE: it only clones already-computed
audio (unprotected, and -- if --epsilon > 0 -- already-PGD-protected via
YourTTS, reusing pgd_perturb exactly as-is) and checks what comes out the other
end. That's a pure forward-pass evaluation, identical in spirit to what
VoiceMark's own paper does against CosyVoice/F5-TTS/MaskGCT in their Table 1.
Building a differentiable XTTS surrogate (to optimize PGD directly against it)
is a separate, much bigger task -- deliberately not attempted here until this
cheaper test says whether it's worth it.

TWO comparisons per run, both against the SAME YourTTS-based SIM embedding
model already used everywhere else in this project (for direct numeric
comparability with existing results, not XTTS's own internal metric):
  - watermark ACC on XTTS's clone of the unprotected watermarked audio
    (directly comparable to watermark_survival_under_cloning.py's YourTTS
    number -- same question, different cloning model)
  - watermark ACC on XTTS's clone of the PGD-protected audio, and SIM of that
    clone (only if --epsilon > 0) -- does disruption optimized against YourTTS
    transfer to a different architecture at all, and does the watermark
    survive there any better than under YourTTS

DISCIPLINE (same as disruption_pgd.py): this integration hasn't been run
against a live XTTS install before -- API details (return format, native
sample rate, exact tts() signature) are implemented from the documented
coqui-tts interface, not verified against a running instance. Run
--diagnostic first: 1 utterance, verbose, prints the raw XTTS output shape/
sample rate it actually got back, so a signature mismatch is caught in
seconds, not after a full sweep silently produces garbage.

Usage:
    # Sanity/API check first -- ALWAYS run this before a full sweep:
    python src/eval/xtts_transfer_eval.py \\
        --checkpoint ./checkpoints/stage1_final_scaleup_recalibrated/recalibrated_final.pt \\
        --diagnostic

    # Full unprotected-only run:
    python src/eval/xtts_transfer_eval.py \\
        --checkpoint ./checkpoints/stage1_final_scaleup_recalibrated/recalibrated_final.pt \\
        --output results/results_xtts_unprotected_run1.json

    # Full run including PGD-protected transfer:
    python src/eval/xtts_transfer_eval.py \\
        --checkpoint ./checkpoints/stage1_final_scaleup_recalibrated/recalibrated_final.pt \\
        --epsilon 0.002 --n_steps 10 --lambda_wm 1.0 \\
        --output results/results_xtts_protected_eps002_run1.json
"""

import os
import sys
import json
import argparse
import tempfile
import torch
import torchaudio
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "losses"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from disruption_pgd import build_backbone, random_message, detect_acc, pgd_perturb, compute_sim
from surrogate_vc import load_yourtts_surrogate
from librispeech import LibriSpeechSubset, collate_librispeech
from vctk import VCTKSubset, collate_vctk
from libritts import LibriTTSSubset, collate_libritts
from torch.utils.data import DataLoader


def load_xtts(device: str):
    """
    High-level coqui-tts API on purpose -- no gradient needed here, so the
    manual @torch.inference_mode()-bypassing reimplementation surrogate_vc.py
    had to do for YourTTS (to keep autograd alive for PGD) is unnecessary.
    Model name/id per Coqui's published model zoo; first call downloads
    weights (multi-GB) and caches them, same one-time-cost pattern as
    LibriSpeechSubset's dataset download.
    """
    import os as _os
    # Kaggle cells are non-interactive -- XTTS's first download blocks on an
    # input() prompt asking you to accept Coqui's non-commercial CPML license
    # (https://coqui.ai/cpml), which raises EOFError instead of hanging. This
    # env var is coqui-tts's own documented way to auto-accept that prompt.
    # Setting it IS agreeing to that license for this download -- fine for
    # academic/thesis (non-commercial) use, but a real legal step, not just a
    # technical flag, so it's called out explicitly rather than silently set.
    _os.environ.setdefault("COQUI_TOS_AGREED", "1")
    from TTS.api import TTS
    print("[load_xtts] Loading tts_models/multilingual/multi-dataset/xtts_v2 "
          "(first run downloads weights -- can take a few minutes)...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    return tts


def xtts_clone(tts, speaker_audio_16k: torch.Tensor, text: str, tmp_dir: str, tag: str) -> torch.Tensor:
    """
    speaker_audio_16k: [1, 1, T] or [1, T] tensor at 16kHz (this project's
    working sample rate throughout). XTTS's speaker_wav argument wants a file
    path, not an in-memory tensor -- write it out, clone, read the result back
    in, resample to 16kHz so it drops into detect_acc/compute_sim unmodified
    (both were written assuming this project's 16kHz convention throughout).
    Returns a [1, 1, T'] float32 tensor at 16kHz.
    """
    os.makedirs(tmp_dir, exist_ok=True)
    ref_path = os.path.join(tmp_dir, f"ref_{tag}.wav")
    out_path = os.path.join(tmp_dir, f"out_{tag}.wav")

    ref_np = speaker_audio_16k.detach().cpu().reshape(-1).numpy()
    sf.write(ref_path, ref_np, 16000)

    tts.tts_to_file(text=text, speaker_wav=ref_path, language="en", file_path=out_path)

    wav, sr = torchaudio.load(out_path)  # [channels, T'] at XTTS's native rate
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav.unsqueeze(0).to(speaker_audio_16k.device)  # [1, 1, T']


def run_eval(backbone, yourtts_surrogate, xtts, eval_loader, device, text: str,
             epsilon: float, step_size: float, n_steps: int, lambda_wm: float,
             tmp_dir: str, seed: int = 123, verbose: bool = False) -> dict:
    keys = ["detection_acc_on_xtts_clone_of_unprotected", "sim_of_xtts_clone_unprotected"]
    if epsilon > 0:
        keys += ["detection_acc_on_xtts_clone_of_protected", "sim_of_xtts_clone_protected"]
    metrics = {k: [] for k in keys}

    for idx, batch in enumerate(eval_loader):
        clean_audio = batch["waveform"].to(device)
        message = random_message(16, clean_audio.shape[0], device, seed=seed + idx)

        with torch.no_grad():
            out = backbone.forward_full(clean_audio, message)
            recon_wm = out["recon_wm"].detach()

        cloned_u = xtts_clone(xtts, recon_wm, text, tmp_dir, f"u{idx}")
        acc_u = detect_acc(backbone, cloned_u, message)
        sim_u = compute_sim(yourtts_surrogate, clean_audio, cloned_u)
        metrics["detection_acc_on_xtts_clone_of_unprotected"].append(acc_u)
        metrics["sim_of_xtts_clone_unprotected"].append(sim_u)

        log_line = f"[{idx}] xtts_clone(unprotected): acc={acc_u:.4f} sim={sim_u:.4f}"

        if epsilon > 0:
            _, perturbed_final, _ = pgd_perturb(
                backbone, yourtts_surrogate, clean_audio, message, text,
                epsilon, step_size, n_steps, random_start=True, lambda_wm=lambda_wm,
            )
            cloned_p = xtts_clone(xtts, perturbed_final, text, tmp_dir, f"p{idx}")
            acc_p = detect_acc(backbone, cloned_p, message)
            sim_p = compute_sim(yourtts_surrogate, clean_audio, cloned_p)
            metrics["detection_acc_on_xtts_clone_of_protected"].append(acc_p)
            metrics["sim_of_xtts_clone_protected"].append(sim_p)
            log_line += f" | xtts_clone(protected): acc={acc_p:.4f} sim={sim_p:.4f}"

        print(log_line)
        if verbose:
            print(f"    recon_wm shape={tuple(recon_wm.shape)} cloned_u shape={tuple(cloned_u.shape)}")

    # FIX (2026-09-09): also return the RAW per-utterance lists, not just the
    # means. Without them the saved JSON supports no post-hoc analysis at all --
    # no significance test, no correlation between watermark survival and clone
    # fidelity, no distribution -- and those values then exist only in stdout,
    # which is gone as soon as the log is overwritten. (This bit for real: the
    # first n=100 run's ACC/SIM correlation had to be recovered by parsing the
    # log, because the `*_values` keys any natural analysis reaches for were
    # never written to the JSON.)
    out = {k: sum(v) / len(v) for k, v in metrics.items()}
    out.update({f"{k}_values": v for k, v in metrics.items()})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--epsilon", type=float, default=0.0)
    p.add_argument("--step_size", type=float, default=None)
    p.add_argument("--n_steps", type=int, default=10)
    p.add_argument("--lambda_wm", type=float, default=1.0)
    p.add_argument("--diagnostic", action="store_true",
                    help="1 utterance, verbose, exits without writing --output. "
                         "Run this FIRST -- see module docstring.")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--tmp_dir", type=str, default="./tmp_xtts_refs")
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
    p.add_argument("--surrogate_text", type=str, default="This is a test sentence for voice cloning.")
    args = p.parse_args()

    if not args.diagnostic and args.output is None:
        p.error("--output is required unless --diagnostic is set")

    step_size = args.step_size if args.step_size is not None else args.epsilon / 4.0
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

    eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False, collate_fn=collate_fn, drop_last=False)

    backbone = build_backbone(args.checkpoint, args.lora_r, args.lora_alpha,
                               include_ffn=False, capacity_lora_r=32)
    print("[main] Loading YourTTS surrogate (frozen, for PGD + SIM embedding only)...")
    yourtts_surrogate = load_yourtts_surrogate(device=device)
    xtts = load_xtts(device=device)

    if args.diagnostic:
        print("\n" + "=" * 60 + "\nDIAGNOSTIC MODE: 1 utterance\n" + "=" * 60)
        batch = next(iter(eval_loader))
        clean_audio = batch["waveform"].to(device)
        message = random_message(16, clean_audio.shape[0], device, seed=123)
        with torch.no_grad():
            out = backbone.forward_full(clean_audio, message)
            recon_wm = out["recon_wm"].detach()
        print(f"recon_wm shape={tuple(recon_wm.shape)} dtype={recon_wm.dtype}")
        cloned = xtts_clone(xtts, recon_wm, args.surrogate_text, args.tmp_dir, "diag")
        print(f"XTTS clone shape={tuple(cloned.shape)} (after resample to 16kHz)")
        acc = detect_acc(backbone, cloned, message)
        sim = compute_sim(yourtts_surrogate, clean_audio, cloned)
        print(f"watermark ACC on XTTS clone: {acc:.4f}  |  SIM: {sim:.4f}")
        print("[main] Diagnostic complete. If shapes look sane and this didn't crash, "
              "drop --diagnostic and add --output for a full run.")
        return

    means = run_eval(backbone, yourtts_surrogate, xtts, eval_loader, device, args.surrogate_text,
                      args.epsilon, step_size, args.n_steps, args.lambda_wm, args.tmp_dir)

    print("\n" + "=" * 60)
    print(f"Watermark ACC, XTTS clone of UNPROTECTED audio: "
          f"{means['detection_acc_on_xtts_clone_of_unprotected']:.4f}  "
          f"<-- compare directly to the YourTTS number from watermark_survival_under_cloning.py")
    print(f"SIM, XTTS clone of unprotected audio: {means['sim_of_xtts_clone_unprotected']:.4f}")
    if args.epsilon > 0:
        print(f"Watermark ACC, XTTS clone of PROTECTED audio: "
              f"{means['detection_acc_on_xtts_clone_of_protected']:.4f}")
        print(f"SIM, XTTS clone of protected audio: {means['sim_of_xtts_clone_protected']:.4f}  "
              f"<-- does YourTTS-optimized PGD disruption transfer to a different architecture at all")
    print("=" * 60)

    with open(args.output, "w") as f:
        json.dump({
            "label": f"xtts_eps{args.epsilon}_lwm{args.lambda_wm}"
                      + (f"_on_{args.checkpoint}" if args.checkpoint else "_on_baseline"),
            "checkpoint": args.checkpoint,
            "dataset": args.dataset,
            "pgd": {"epsilon": args.epsilon, "step_size": step_size,
                     "n_steps": args.n_steps, "lambda_wm": args.lambda_wm},
            "results": {**means, "n_trials": len(eval_ds)},
        }, f, indent=2)
    print(f"[main] Saved results to {args.output}")


if __name__ == "__main__":
    main()
