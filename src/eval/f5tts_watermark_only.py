"""
src/eval/f5tts_watermark_only.py

THE DECISIVE MEASUREMENT: does the watermark survive cloning by F5-TTS — one of the
three models VoiceMark actually evaluates?

WHY THIS IS THE EXPERIMENT THAT MATTERS
----------------------------------------
Measured so far, using VoiceMark's OWN released weights (LoRA at zero-init):

    YourTTS   0.5337     XTTS-v2   0.6119        <- this project's cloners
    CosyVoice 0.964      F5-TTS 0.979   MaskGCT 0.957   <- VoiceMark's published numbers

Same model, wildly different outcomes. The cloner ARCHITECTURE decides everything —
consistent with the conditioning-bottleneck hypothesis: F5-TTS conditions by *infilling*
(the reference mel is concatenated into the input sequence, so raw acoustic detail flows
through the network), while YourTTS compresses the speaker to a single d-vector and XTTS
sits in between. A watermark living in the speaker latent can only reach the clone if the
cloner carries that latent forward.

If this script reproduces ~0.96 on F5-TTS, then attribution is not broken — it is
architecture-dependent, and every conclusion drawn from YourTTS/XTTS alone was measured on
the architectures where this class of method structurally cannot work.

WHY IT CAN RUN AT ALL (the dependency conflict is avoidable, not fundamental)
-----------------------------------------------------------------------------
`f5tts_transfer_eval.py` cannot run in an f5-tts environment because it imports the YourTTS
surrogate for SIM, which pulls in `coqui-tts`, whose `transformers` pin is mutually
exclusive with f5-tts's. But the decisive question needs NO surrogate:

    watermark -> F5-TTS clone -> detect

requires only the backbone (verified: `backbone.py` and `adapters.py` import nothing beyond
torch) and f5-tts. This script therefore imports NO coqui code and runs in a clean
f5-tts environment. No three-stage split, no environment juggling.

Speaker similarity is not computed here — it needs the surrogate. Clone WAVs are saved so
SIM can be scored afterwards with `ecapa_sim_eval.py`, which uses speechbrain (also
coqui-free) and is the SafeSpeech-comparable metric anyway.

SCOPE: watermark only, NO adversarial perturbation. PGD requires the differentiable
surrogate, hence coqui. That is deliberate — this measures whether attribution survives
F5-TTS cloning at all, which is the prerequisite. If it does, the protected case is the
follow-up (generate protected WAVs in the coqui environment, clone them here).

PRE-REGISTERED READING
----------------------
  acc_clone >= 0.90  -> VoiceMark's claim reproduces on its own architecture class.
                        Attribution works; prior negatives were cloner-specific.
                        The thesis narrative changes substantially.
  acc_clone ~ 0.6    -> intermediate, like XTTS. The bottleneck hypothesis holds on a
                        gradient rather than a cliff.
  acc_clone ~ 0.5    -> VoiceMark's published numbers do not reproduce in ANY harness
                        here, which is a much stronger claim about the paper and needs
                        the eval protocol itself audited before being asserted.

Usage:
    # In an environment with f5-tts installed (coqui-tts NOT required, and if present
    # its transformers pin may conflict -- prefer a clean env):
    pip install f5-tts faster-whisper

    python src/eval/f5tts_watermark_only.py \\
        --diagnostic                                   # 1 utterance, verbose, ALWAYS first

    python src/eval/f5tts_watermark_only.py \\
        --n_speakers 60 --n_eval_speakers 20 --eval_utterances_per_speaker 5 \\
        --n_utterances 100 \\
        --save_clones_dir ./audio_samples/f5tts_watermark \\
        --output results/results_f5tts_watermark_n100.json

    # then, for SafeSpeech-comparable SIM (speechbrain, also coqui-free):
    python src/eval/ecapa_sim_eval.py --sample_dir ./audio_samples/f5tts_watermark
"""

import os
import sys
import json
import argparse
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

# Deliberately NOT importing from disruption_pgd -- that module imports the YourTTS
# surrogate, which would drag coqui-tts back in and recreate the conflict this script
# exists to avoid. backbone/adapters/librispeech are all torch-only.
from backbone import VoiceMarkBackbone
from adapters import apply_lora_adapters
from librispeech import LibriSpeechSubset, collate_librispeech


def build_backbone(checkpoint_path, r=8, alpha=16):
    backbone = VoiceMarkBackbone()
    apply_lora_adapters(backbone, r=r, alpha=alpha)
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        backbone.model.load_state_dict(ckpt["lora_state_dict"], strict=False)
        print(f"[backbone] loaded {checkpoint_path} (epoch {ckpt.get('epoch')})")
    else:
        print("[backbone] LoRA at zero-init == PRETRAINED VOICEMARK (their released weights)")
    for p in backbone.model.parameters():
        p.requires_grad = False
    backbone.model.eval()
    return backbone


def random_message(nbits, batch_size, device, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.randint(0, 2, (batch_size, nbits), generator=gen, device=device)


def compute_detection_accuracy(chunk_logits, message, nchunk_size=4):
    pred_chunks = torch.argmax(chunk_logits, dim=-1)
    batch, nchunks = pred_chunks.shape
    correct, total = 0, 0
    for i in range(nchunks):
        true_chunk = message[:, i * nchunk_size:(i + 1) * nchunk_size]
        pred_val = pred_chunks[:, i]
        for b in range(nchunk_size):
            correct += (((pred_val >> b) & 1) == true_chunk[:, b]).sum().item()
            total += batch
    return correct / total


def detect_acc(backbone, wav, message):
    with torch.no_grad():
        feat = backbone.model.st_model.forward_feature(wav)
        _logits, chunk_logits = backbone.model.detector(feat)
    return compute_detection_accuracy(chunk_logits, message)


def load_f5tts(device):
    from f5_tts.api import F5TTS
    print("[f5tts] loading F5-TTS (first run downloads weights)...")
    return F5TTS(device=device)


def f5tts_clone(f5tts, speaker_audio_16k, text, tmp_dir, tag, ref_text=""):
    """
    ref_text="" makes F5-TTS auto-transcribe the reference via faster-whisper --
    LibriSpeech crops carry no transcript here. If faster-whisper is missing this
    fails with a confusing error, hence the explicit check in main().
    """
    import soundfile as sf
    import torchaudio
    os.makedirs(tmp_dir, exist_ok=True)
    ref_path = os.path.join(tmp_dir, f"ref_{tag}.wav")
    sf.write(ref_path, speaker_audio_16k.detach().cpu().reshape(-1).numpy(), 16000)

    wav, sr, _spect = f5tts.infer(ref_file=ref_path, ref_text=ref_text, gen_text=text)

    t = torch.as_tensor(wav, dtype=torch.float32).reshape(1, 1, -1)
    if sr != 16000:
        t = torchaudio.functional.resample(t, sr, 16000)
    return t.to(speaker_audio_16k.device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Omit for PRETRAINED VoiceMark (their released weights) -- that is "
                        "the arm that tests their published claim directly.")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--diagnostic", action="store_true")
    p.add_argument("--n_utterances", type=int, default=100)
    p.add_argument("--save_clones_dir", type=str, default=None)
    p.add_argument("--gen_text", type=str, default="This is a test sentence for voice cloning.")
    p.add_argument("--ref_text", type=str, default="")
    p.add_argument("--tmp_dir", type=str, default="./tmp_f5tts")
    p.add_argument("--data_root", type=str, default="./data/librispeech")
    p.add_argument("--n_speakers", type=int, default=60)
    p.add_argument("--utterances_per_speaker", type=int, default=15)
    p.add_argument("--n_eval_speakers", type=int, default=20)
    p.add_argument("--eval_utterances_per_speaker", type=int, default=5)
    p.add_argument("--crop_seconds", type=float, default=3.0)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not args.ref_text:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            print("[warn] ref_text is empty and faster-whisper is NOT installed. F5-TTS "
                  "auto-transcription will fail. Run: pip install faster-whisper "
                  "(or pass --ref_text).")

    eval_ds = LibriSpeechSubset(
        root=args.data_root, n_speakers=args.n_speakers,
        utterances_per_speaker=args.utterances_per_speaker,
        n_eval_speakers=args.n_eval_speakers,
        eval_utterances_per_speaker=args.eval_utterances_per_speaker,
        sample_rate=16000, crop_seconds=args.crop_seconds, split="eval",
    )
    loader = DataLoader(eval_ds, batch_size=1, shuffle=False, collate_fn=collate_librispeech)

    backbone = build_backbone(args.checkpoint, args.lora_r, args.lora_alpha)
    backbone.model.to(device)
    f5tts = load_f5tts(device)

    if args.diagnostic:
        print("\n" + "=" * 64 + "\nDIAGNOSTIC (1 utterance)\n" + "=" * 64)
        batch = next(iter(loader))
        clean = batch["waveform"].to(device)
        msg = random_message(16, 1, device, seed=123)
        out = backbone.forward_full(clean, msg)
        recon_wm = out["recon_wm"]
        cloned = f5tts_clone(f5tts, recon_wm, args.gen_text, args.tmp_dir, "diag", args.ref_text)
        print(f"  recon_wm {tuple(recon_wm.shape)}  clone {tuple(cloned.shape)}")
        print(f"  ACC on watermarked source : {detect_acc(backbone, recon_wm, msg):.4f}  (expect ~0.99)")
        print(f"  ACC on F5-TTS clone       : {detect_acc(backbone, cloned, msg):.4f}  (0.5 = chance)")
        print("\n  n=1 is indicative only. If shapes are sane and source ACC is ~0.99, run for real.")
        return

    accs_src, accs_clone = [], []
    print(f"\n{'=' * 76}")
    print(f"F5-TTS WATERMARK SURVIVAL | {'PRETRAINED VoiceMark' if args.checkpoint is None else args.checkpoint}")
    print(f"VoiceMark's published F5-TTS number: 0.979 | this project's XTTS: 0.6119, YourTTS: 0.5337")
    print(f"{'=' * 76}")

    for i, batch in enumerate(loader):
        if i >= args.n_utterances:
            break
        clean = batch["waveform"].to(device)
        msg = random_message(16, 1, device, seed=123 + i)
        with torch.no_grad():
            recon_wm = backbone.forward_full(clean, msg)["recon_wm"]
        cloned = f5tts_clone(f5tts, recon_wm, args.gen_text, args.tmp_dir, f"u{i}", args.ref_text)

        a_src = detect_acc(backbone, recon_wm, msg)
        a_cln = detect_acc(backbone, cloned, msg)
        accs_src.append(a_src)
        accs_clone.append(a_cln)
        print(f"  [{i}] acc_source={a_src:.4f}  acc_f5tts_clone={a_cln:.4f}", flush=True)

        if args.save_clones_dir:
            import soundfile as sf
            os.makedirs(args.save_clones_dir, exist_ok=True)
            sf.write(os.path.join(args.save_clones_dir, f"sample{i}_reference.wav"),
                     clean[0].detach().cpu().reshape(-1).numpy(), 16000)
            sf.write(os.path.join(args.save_clones_dir, f"sample{i}_clone_f5tts.wav"),
                     cloned[0].detach().cpu().reshape(-1).numpy(), 16000)

        if args.output and (i + 1) % 10 == 0:
            with open(args.output, "w") as f:
                json.dump({"label": "f5tts_watermark_only", "checkpoint": args.checkpoint,
                           "n_completed": i + 1,
                           "results": {"acc_source_mean": sum(accs_src) / len(accs_src),
                                       "acc_clone_mean": sum(accs_clone) / len(accs_clone),
                                       "acc_source_values": accs_src,
                                       "acc_clone_values": accs_clone}}, f, indent=2)

    m_src = sum(accs_src) / len(accs_src)
    m_cln = sum(accs_clone) / len(accs_clone)
    print(f"\n{'=' * 76}")
    print(f"RESULT (n={len(accs_clone)})")
    print(f"{'=' * 76}")
    print(f"  ACC on watermarked source audio : {m_src:.4f}   (sanity -- the mark is present)")
    print(f"  ACC on F5-TTS clone      [KEY]  : {m_cln:.4f}   (0.5 = chance)")
    print(f"\n  Reference points, same detector, same 16-bit payload:")
    print(f"    VoiceMark published, F5-TTS : 0.979")
    print(f"    this project, XTTS-v2       : 0.6119")
    print(f"    this project, YourTTS       : 0.5337")
    print(f"\n  >=0.90 -> VoiceMark's claim reproduces; attribution is architecture-dependent,")
    print(f"            and the YourTTS/XTTS negatives were cloner-specific, not fundamental.")
    print(f"  ~0.6   -> a gradient rather than a cliff; bottleneck hypothesis still holds.")
    print(f"  ~0.5   -> their numbers do not reproduce in ANY harness here -- audit the")
    print(f"            evaluation protocol before asserting anything that strong.")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"label": "f5tts_watermark_only", "checkpoint": args.checkpoint,
                       "n_completed": len(accs_clone),
                       "reference_points": {"voicemark_published_f5tts": 0.979,
                                             "this_project_xtts": 0.6119,
                                             "this_project_yourtts": 0.5337},
                       "results": {"acc_source_mean": m_src, "acc_clone_mean": m_cln,
                                   "acc_source_values": accs_src,
                                   "acc_clone_values": accs_clone}}, f, indent=2)
        print(f"\n[main] Saved to {args.output}")


if __name__ == "__main__":
    main()
