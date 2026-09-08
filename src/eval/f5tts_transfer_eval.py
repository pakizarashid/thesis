"""
src/eval/f5tts_transfer_eval.py

Same pattern as xtts_transfer_eval.py, but against F5-TTS specifically --
one of the THREE actual models VoiceMark's own paper reports numbers for
(CosyVoice 96.4%, F5-TTS 97.9%, MaskGCT 95.7%), not an architecturally-similar
proxy. The XTTS result (ACC 0.625, n=25, ~5 SD above chance -- a real effect,
but far below VoiceMark's claimed 95%+) already ruled out an implementation
bug: this reproduction's embedder correctly targets the speaker-specific
latent layers (VQ2-8, excluding the content layer VQ1) exactly as VoiceMark's
paper describes -- confirmed directly from external/voicemark/speechtokenizer/
model.py's forward() (quantized_list[1:] for the acoustic/watermark-carrying
subset, quantized_list[0] for content, kept untouched). So the remaining gap
is specifically about how much of that latent trajectory a given cloning
model actually reconstructs vs. regenerates -- and the only way to know if
THIS reproduction reaches VoiceMark's own reported range is to test against
one of their own three models directly. F5-TTS was picked as the most
tractable of the three to integrate here (pip-installable package, simple
Python inference API) -- not picked for being the strongest or weakest of
the three, just the lowest-friction to add.

Structurally identical to xtts_transfer_eval.py: forward-pass only (no new
differentiable surrogate needed), reuses the SAME PGD-protected audio (still
optimized against YourTTS, via pgd_perturb) for the --epsilon > 0 transfer
check, and reuses YourTTS's own embedding model for SIM so all three
transfer-eval scripts (YourTTS, XTTS, F5-TTS) report SIM on the same scale
and are directly comparable to each other.

IMPORTANT UNVERIFIED DETAIL, more so than the XTTS integration: F5-TTS's
infer() takes a `ref_text` argument -- the TRANSCRIPT of the reference audio,
not just the reference audio itself (F5-TTS is a flow-matching, text-aligned
model; XTTS and YourTTS don't need this). This project's own eval utterances
don't come with transcripts, so this script passes ref_text="" and relies on
F5-TTS's own documented behavior of auto-transcribing the reference via
faster-whisper when ref_text is empty -- which means faster-whisper needs to
be installed too, and adds real per-utterance latency (ASR pass before every
clone) on top of F5-TTS's own generation time. Flagging this now rather than
letting it surface as a confusing crash: if faster-whisper isn't present,
pip install faster-whisper (or whisper) alongside f5-tts, or supply real
transcripts via --ref_text_source if that becomes worth building.

DISCIPLINE (same as xtts_transfer_eval.py): this hasn't been run against a
live f5-tts install. --diagnostic first, always -- see module usage below.

STATUS (2026-09-08): BLOCKED in this project's shared environment, not
abandoned. `pip install f5-tts` pulls transformers>=5, but this project's
coqui-tts==0.27.5 install (needed for the YourTTS surrogate + XTTS transfer
eval, both load-bearing for other results in this repo) hard-depends at
import time on transformers.pytorch_utils.isin_mps_friendly, a symbol
removed in transformers 5.x -- true regardless of what coqui-tts's own
declared metadata floor (transformers>=4.57, no upper bound) claims. So
f5-tts and coqui-tts want mutually exclusive transformers major versions
in the same environment; installing f5-tts breaks every YourTTS/XTTS
script in this repo (confirmed directly -- see commit history), and there
is no version of transformers that satisfies both at once. Until this
repo's other results move off coqui-tts/YourTTS entirely (not planned),
this script stays unrun. The code is complete and believed correct (see
the ref_text caveat above); the XTTS transfer result already stands as
this project's cross-model validation against a second, architecturally
distinct zero-shot model, which was the actual goal this script was
written to extend.

Usage:
    python src/eval/f5tts_transfer_eval.py \\
        --checkpoint ./checkpoints/stage1_final_scaleup_recalibrated/recalibrated_final.pt \\
        --diagnostic

    python src/eval/f5tts_transfer_eval.py \\
        --checkpoint ./checkpoints/stage1_final_scaleup_recalibrated/recalibrated_final.pt \\
        --output results/results_f5tts_unprotected_n25_run1.json
"""

import os
import sys
import json
import argparse
import torch
import torchaudio
import numpy as np
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


def load_f5tts(device: str):
    from f5_tts.api import F5TTS
    print("[load_f5tts] Loading F5TTS_v1_Base (first run downloads weights from "
          "HuggingFace -- can take a few minutes)...")
    f5tts = F5TTS(model="F5TTS_v1_Base", device=device)
    return f5tts


def f5tts_clone(f5tts, speaker_audio_16k: torch.Tensor, text: str, tmp_dir: str,
                 tag: str, ref_text: str = "") -> torch.Tensor:
    """
    Mirrors xtts_clone() in xtts_transfer_eval.py. speaker_audio_16k: [1,1,T]
    or [1,T] at 16kHz. Writes a reference wav (F5-TTS's ref_file wants a path,
    same as XTTS's speaker_wav), calls infer(), and uses the wav array F5-TTS
    hands back directly (no re-read from disk needed, unlike the XTTS path --
    infer() returns (wav, sr, spect) in memory per its documented API).
    Returns a [1, 1, T'] float32 tensor at 16kHz.
    """
    os.makedirs(tmp_dir, exist_ok=True)
    ref_path = os.path.join(tmp_dir, f"ref_{tag}.wav")
    ref_np = speaker_audio_16k.detach().cpu().reshape(-1).numpy()
    sf.write(ref_path, ref_np, 16000)

    wav, sr, _spect = f5tts.infer(
        ref_file=ref_path,
        ref_text=ref_text,   # "" -> F5-TTS auto-transcribes via faster-whisper
        gen_text=text,
        file_wave=None,
        file_spect=None,
        seed=None,
    )

    if isinstance(wav, np.ndarray):
        wav_t = torch.from_numpy(wav).float()
    else:
        wav_t = torch.as_tensor(wav).float()
    if wav_t.dim() == 1:
        wav_t = wav_t.unsqueeze(0)  # [1, T']

    if sr != 16000:
        wav_t = torchaudio.functional.resample(wav_t, sr, 16000)
    return wav_t.unsqueeze(0).to(speaker_audio_16k.device)  # [1, 1, T']


def run_eval(backbone, yourtts_surrogate, f5tts, eval_loader, device, text: str, ref_text: str,
             epsilon: float, step_size: float, n_steps: int, lambda_wm: float,
             tmp_dir: str, seed: int = 123, verbose: bool = False) -> dict:
    keys = ["detection_acc_on_f5tts_clone_of_unprotected", "sim_of_f5tts_clone_unprotected"]
    if epsilon > 0:
        keys += ["detection_acc_on_f5tts_clone_of_protected", "sim_of_f5tts_clone_protected"]
    metrics = {k: [] for k in keys}

    for idx, batch in enumerate(eval_loader):
        clean_audio = batch["waveform"].to(device)
        message = random_message(16, clean_audio.shape[0], device, seed=seed + idx)

        with torch.no_grad():
            out = backbone.forward_full(clean_audio, message)
            recon_wm = out["recon_wm"].detach()

        cloned_u = f5tts_clone(f5tts, recon_wm, text, tmp_dir, f"u{idx}", ref_text=ref_text)
        acc_u = detect_acc(backbone, cloned_u, message)
        sim_u = compute_sim(yourtts_surrogate, clean_audio, cloned_u)
        metrics["detection_acc_on_f5tts_clone_of_unprotected"].append(acc_u)
        metrics["sim_of_f5tts_clone_unprotected"].append(sim_u)

        log_line = f"[{idx}] f5tts_clone(unprotected): acc={acc_u:.4f} sim={sim_u:.4f}"

        if epsilon > 0:
            _, perturbed_final, _ = pgd_perturb(
                backbone, yourtts_surrogate, clean_audio, message, text,
                epsilon, step_size, n_steps, random_start=True, lambda_wm=lambda_wm,
            )
            cloned_p = f5tts_clone(f5tts, perturbed_final, text, tmp_dir, f"p{idx}", ref_text=ref_text)
            acc_p = detect_acc(backbone, cloned_p, message)
            sim_p = compute_sim(yourtts_surrogate, clean_audio, cloned_p)
            metrics["detection_acc_on_f5tts_clone_of_protected"].append(acc_p)
            metrics["sim_of_f5tts_clone_protected"].append(sim_p)
            log_line += f" | f5tts_clone(protected): acc={acc_p:.4f} sim={sim_p:.4f}"

        print(log_line)
        if verbose:
            print(f"    recon_wm shape={tuple(recon_wm.shape)} cloned_u shape={tuple(cloned_u.shape)}")

    return {k: sum(v) / len(v) for k, v in metrics.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--epsilon", type=float, default=0.0)
    p.add_argument("--step_size", type=float, default=None)
    p.add_argument("--n_steps", type=int, default=10)
    p.add_argument("--lambda_wm", type=float, default=1.0)
    p.add_argument("--ref_text", type=str, default="",
                    help="Transcript of the reference/speaker audio. Empty string (default) "
                         "relies on F5-TTS's own auto-transcription via faster-whisper -- "
                         "slower per utterance, but needs no ground-truth transcripts wired up.")
    p.add_argument("--diagnostic", action="store_true",
                    help="1 utterance, verbose, exits without writing --output. Run this FIRST.")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--tmp_dir", type=str, default="./tmp_f5tts_refs")
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
    f5tts = load_f5tts(device=device)

    if args.diagnostic:
        print("\n" + "=" * 60 + "\nDIAGNOSTIC MODE: 1 utterance\n" + "=" * 60)
        batch = next(iter(eval_loader))
        clean_audio = batch["waveform"].to(device)
        message = random_message(16, clean_audio.shape[0], device, seed=123)
        with torch.no_grad():
            out = backbone.forward_full(clean_audio, message)
            recon_wm = out["recon_wm"].detach()
        print(f"recon_wm shape={tuple(recon_wm.shape)} dtype={recon_wm.dtype}")
        cloned = f5tts_clone(f5tts, recon_wm, args.surrogate_text, args.tmp_dir, "diag", ref_text=args.ref_text)
        print(f"F5-TTS clone shape={tuple(cloned.shape)} (after resample to 16kHz)")
        acc = detect_acc(backbone, cloned, message)
        sim = compute_sim(yourtts_surrogate, clean_audio, cloned)
        print(f"watermark ACC on F5-TTS clone: {acc:.4f}  |  SIM: {sim:.4f}")
        print("[main] Diagnostic complete. If shapes look sane and this didn't crash, "
              "drop --diagnostic and add --output for a full run.")
        return

    means = run_eval(backbone, yourtts_surrogate, f5tts, eval_loader, device,
                      args.surrogate_text, args.ref_text,
                      args.epsilon, step_size, args.n_steps, args.lambda_wm, args.tmp_dir)

    print("\n" + "=" * 60)
    print(f"Watermark ACC, F5-TTS clone of UNPROTECTED audio: "
          f"{means['detection_acc_on_f5tts_clone_of_unprotected']:.4f}  "
          f"<-- directly comparable to VoiceMark's own reported 0.979 for F5-TTS")
    print(f"SIM, F5-TTS clone of unprotected audio: {means['sim_of_f5tts_clone_unprotected']:.4f}")
    if args.epsilon > 0:
        print(f"Watermark ACC, F5-TTS clone of PROTECTED audio: "
              f"{means['detection_acc_on_f5tts_clone_of_protected']:.4f}")
        print(f"SIM, F5-TTS clone of protected audio: {means['sim_of_f5tts_clone_protected']:.4f}")
    print("=" * 60)

    with open(args.output, "w") as f:
        json.dump({
            "label": f"f5tts_eps{args.epsilon}_lwm{args.lambda_wm}"
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
