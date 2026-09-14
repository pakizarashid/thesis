"""
src/eval/gen_samples_yourtts.py

Small standalone generator: writes sample{i}_reference.wav (= recon_wm, the
watermarked source) and sample{i}_clone_yourtts.wav (= YourTTS's zero-shot clone
of it) pairs to disk, in the SAME naming convention cloner_watermark_eval.py
--save_clones_dir already uses for CosyVoice/MaskGCT/F5-TTS -- so
carrier_probe.py can read either without modification. Needed because
watermark_survival_under_cloning.py (the script that originally measured
YourTTS's 0.5337 ACC point on the ladder) has no save mechanism at all.

Reuses build_backbone/random_message/detect_acc from disruption_pgd.py and
load_yourtts_surrogate from surrogate_vc.py -- both already verified elsewhere
in this project. No PGD, no gradient, no new dependency -- pure generation.

TEXT: every other cloning script in this project (watermark_survival_under_cloning.py,
xtts_transfer_eval.py, cloner_watermark_eval.py's --gen_text default) synthesizes a
FIXED sentence ("This is a test sentence for voice cloning.") regardless of what the
reference utterance actually says. That is fine for ACC-style measurement (the
watermark detector doesn't care what words are spoken), but CARRIER-PROBE compares
raw per-layer latents frame-by-frame, which only means something if source and clone
contain the SAME words -- otherwise cosine similarity mixes "different content" with
"different voice/architecture" and the number isn't interpretable. So this script
synthesizes the reference's OWN transcript (LibriSpeechSubset already provides it,
no new transcription dependency) instead of the generic fixed sentence.

CROP_SECONDS (2026-09-12 fix): LibriSpeechSubset's transcript is for the FULL
utterance, but its waveform is cropped/padded to a FIXED length (--crop_seconds,
and for the eval split a CENTER crop when the utterance is longer than that) --
the transcript does NOT get cropped to match. At the old default (3.0s) this meant
asking the TTS to say a whole sentence while the reference audio was only a 3-second
(often mid-utterance) slice of it -- exactly what produced the 60%+ frame-count
mismatches seen on first use of --use_own_transcript. Default bumped to 20.0s here,
comfortably above nearly every LibriSpeech utterance, so cropping essentially never
triggers and the transcript matches what's actually in the (at most zero-padded at
the tail) reference waveform.

TRAILING-SILENCE TRIM (2026-09-12, second fix): the 20.0s bump above created a NEW
problem instead of the one it fixed -- most LibriSpeech utterances are well under
20s, so _crop_or_pad zero-pads the reference waveform out to a full 20s, often
5-15s of which is pure trailing silence. carrier_probe.py's frame-count-mismatch
check compares RAW lengths (reference including that padding, vs the clone's
natural, unpadded length), so it was flagging padding overhang as if it were a
content mismatch -- 15/15 YourTTS samples got flagged "LARGE" on the very run this
fix was meant to clean up, which is backwards. The crop-to-shorter-length in
layerwise_cosine was never actually wrong (padding sits at the tail, the crop takes
the prefix, so real content was still being compared correctly) -- only the
mismatch *percentage* was misleading. Fixed at the source instead of downstream:
trim_trailing_silence() below cuts the zero-padding off clean_audio right after
loading it, before recon_wm/cloning/saving, so the saved reference contains only
real recorded audio and the frame-count-mismatch check means what it says again.

Usage:
    python src/eval/gen_samples_yourtts.py --n_utterances 15 \\
        --save_clones_dir ./audio_samples/carrierprobe_yourtts
"""
import os
import sys
import argparse
import soundfile as sf
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "losses"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from disruption_pgd import build_backbone, random_message, detect_acc
from surrogate_vc import load_yourtts_surrogate
from librispeech import LibriSpeechSubset, collate_librispeech


def trim_trailing_silence(waveform: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    waveform: [1, T] or [T]. Cuts trailing near-zero samples (the _crop_or_pad
    padding LibriSpeechSubset appends when an utterance is shorter than
    --crop_seconds) so the returned tensor is just the real recorded audio, plus
    at most a few silent samples at the true end of speech. A threshold, not exact
    zero, since real recordings have low-level room noise that pads with exact
    zeros wouldn't. If the whole clip is silence (shouldn't happen with real
    LibriSpeech audio), returns it unchanged rather than collapsing to empty.
    """
    w = waveform.squeeze(0) if waveform.dim() > 1 else waveform
    nonzero = (w.abs() > eps).nonzero()
    if nonzero.numel() == 0:
        return waveform
    last = nonzero[-1].item()
    trimmed = w[: last + 1]
    return trimmed.unsqueeze(0) if waveform.dim() > 1 else trimmed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Omit for pretrained VoiceMark -- matches the five-architecture "
                        "ladder's own protocol (prereg-conditioning-bandwidth-2026-09-10.md).")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--n_utterances", type=int, default=15)
    p.add_argument("--message_seed_base", type=int, default=123)
    p.add_argument("--surrogate_text", type=str, default="This is a test sentence for voice cloning.",
                    help="Fallback ONLY if a reference utterance's own transcript is empty. By "
                         "default this script uses each utterance's own transcript (see "
                         "--use_own_transcript) so source and clone contain the same words.")
    p.add_argument("--use_own_transcript", action="store_true", default=True,
                    help="Synthesize the reference's own LibriSpeech transcript instead of "
                         "--surrogate_text (default True -- required for CARRIER-PROBE's "
                         "per-layer comparison to be valid; pass --no_use_own_transcript to "
                         "restore the old fixed-sentence behavior for other experiments).")
    p.add_argument("--no_use_own_transcript", dest="use_own_transcript", action="store_false")
    p.add_argument("--save_clones_dir", type=str, default=None,
                    help="Omit to skip writing audio to disk entirely -- "
                         "CARRIER-REWEIGHT only needs the printed "
                         "acc_yourtts_clone numbers, not the audio itself. Pass "
                         "a path only when you actually want to keep the clones.")
    p.add_argument("--data_root", type=str, default="./data/librispeech")
    p.add_argument("--n_speakers", type=int, default=60)
    p.add_argument("--utterances_per_speaker", type=int, default=15)
    p.add_argument("--n_eval_speakers", type=int, default=20)
    p.add_argument("--eval_utterances_per_speaker", type=int, default=5)
    p.add_argument("--crop_seconds", type=float, default=20.0,
                    help="Bumped up from the project's usual 3.0s default -- see the "
                         "CROP_SECONDS note in this file's docstring. Only safe to lower back "
                         "to 3.0 if --no_use_own_transcript is also passed.")
    p.add_argument("--layer_scales", type=str, default="1,1,1,1,1,1,1",
                    help="CARRIER-REWEIGHT (2026-09-14): 7 comma-separated floats, one per "
                         "RVQ layer 2-8 in order, scaling that layer's msg_processor output "
                         "before summing into acoustic_wm. No retraining. Default all-1.0 "
                         "leaves behavior exactly unchanged.")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = build_backbone(args.checkpoint, args.lora_r, args.lora_alpha, False, 32)
    layer_scales = [float(s) for s in args.layer_scales.split(",")]
    assert len(layer_scales) == 7, f"--layer_scales needs 7 floats, got {len(layer_scales)}"
    if any(s != 1.0 for s in layer_scales):
        from layer_reweight import LayerReweightMsgProcessor
        backbone.model.msg_processor = LayerReweightMsgProcessor(backbone.model.msg_processor, layer_scales)
        print(f"[layer_reweight] ACTIVE: scales={layer_scales} (layer2..layer8 in order)")
    print("[gen_samples_yourtts] loading YourTTS surrogate...")
    surrogate = load_yourtts_surrogate(device=device)

    eval_ds = LibriSpeechSubset(
        root=args.data_root, n_speakers=args.n_speakers,
        utterances_per_speaker=args.utterances_per_speaker,
        n_eval_speakers=args.n_eval_speakers,
        eval_utterances_per_speaker=args.eval_utterances_per_speaker,
        sample_rate=16000, crop_seconds=args.crop_seconds, split="eval",
    )
    loader = DataLoader(eval_ds, batch_size=1, shuffle=False, collate_fn=collate_librispeech)

    if args.save_clones_dir:
        os.makedirs(args.save_clones_dir, exist_ok=True)
    n_written = 0
    accs_clone = []
    for i, batch in enumerate(loader):
        if i >= args.n_utterances:
            break
        clean_audio = trim_trailing_silence(batch["waveform"][0]).unsqueeze(0).to(device)
        message = random_message(16, clean_audio.shape[0], device, seed=args.message_seed_base + i)

        text = args.surrogate_text
        if args.use_own_transcript:
            own_transcript = (batch.get("transcript") or [""])[0].strip()
            if own_transcript:
                text = own_transcript
            elif i == 0:
                print(f"[gen_samples_yourtts] WARNING: empty transcript for utterance 0 -- "
                      f"falling back to --surrogate_text ({args.surrogate_text!r}). If this "
                      f"keeps happening, content will not match the reference and "
                      f"carrier_probe.py's similarity numbers will be unreliable.")

        with torch.no_grad():
            out = backbone.forward_full(clean_audio, message)
            recon_wm = out["recon_wm"].detach()
            cloned = surrogate.clone_voice(recon_wm, text=text)
            a_cln = detect_acc(backbone, cloned, message)
            accs_clone.append(a_cln)
            print(f"[gen_samples_yourtts] [{i}] acc_yourtts_clone={a_cln:.4f}")

        # Seed-mismatch guard, same discipline as cloner_watermark_eval.py: acc_source
        # should be ~0.99 (watermark decodes from the source itself) before trusting
        # anything saved from this run.
        if i == 0:
            a_src = detect_acc(backbone, recon_wm, message)
            print(f"[gen_samples_yourtts] sample 0 acc_source={a_src:.4f} (expect ~0.99 -- if "
                  f"much lower, the checkpoint/message-seed convention doesn't match what "
                  f"carrier_probe.py will later assume).")
            if a_src < 0.85:
                raise SystemExit("ABORT: acc_source too low -- fix checkpoint/seed before saving.")

        if args.save_clones_dir:
            sf.write(os.path.join(args.save_clones_dir, f"sample{i}_reference.wav"),
                     recon_wm[0].detach().cpu().reshape(-1).numpy(), 16000)
            sf.write(os.path.join(args.save_clones_dir, f"sample{i}_clone_yourtts.wav"),
                     cloned[0].detach().cpu().reshape(-1).numpy(), 16000)
        n_written += 1

    if args.save_clones_dir:
        print(f"[gen_samples_yourtts] wrote {n_written} reference/clone pairs to {args.save_clones_dir}")
    else:
        print(f"[gen_samples_yourtts] processed {n_written} utterances (audio not saved -- "
              f"pass --save_clones_dir to keep the .wav files).")
    if accs_clone:
        mean_acc = sum(accs_clone) / len(accs_clone)
        print(f"[gen_samples_yourtts] RESULT mean acc_yourtts_clone={mean_acc:.4f} (n={len(accs_clone)}, "
              f"layer_scales={layer_scales})")


if __name__ == "__main__":
    main()
