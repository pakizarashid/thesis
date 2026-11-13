#!/usr/bin/env python
"""Protect a recording before publishing it, or verify one that was.

    python protect.py --input voice.wav --output protected.wav
    python protect.py --verify protected.wav --message 1010001111110111

Two stages are applied, in this order:

  1. a clone-aware watermark -- a 16-bit message written into the
     speaker-specific codec latents, trained so that it survives a zero-shot
     clone of the audio;
  2. an F-ratio spectral mask -- the K STFT bins that best separate speakers
     are zeroed and the result is RMS-matched, so loudness is unchanged.

Neither stage uses a speaker verifier, a surrogate model or a gradient.

The 16-bit message is what makes a clone traceable, so STORE IT. --verify
reports how many of the 16 bits survive; 13 or more counts as traced.
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("src", "src/eval", "src/data"):
    q = os.path.join(HERE, p)
    if q not in sys.path:
        sys.path.insert(0, q)
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import numpy as np
import soundfile as sf
import torch

import cloner_watermark_eval as cwe
from clone_eval_v2 import decode_bits
from nonadditive_transforms import SR, notch, select_bins

DEFAULT_CKPT = "checkpoints/route2_vmaug_R0/route2_epoch19.pt"
DEFAULT_BINS = "results/n1_fratio_bins.json"
DEFAULT_K    = 12          # LibriSpeech; VCTK uses 24. Chosen by perceptual cost.
NBITS        = 16
TRACE_BITS   = 13          # >= 13 of 16 recovered == traced


def read_audio(path):
    x, sr = sf.read(path, always_2d=False)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    return x, sr


def load_backbone(ckpt, device):
    b = cwe.build_backbone(ckpt, 8, 16)
    b.model.to(device).eval()
    return b


def parse_message(s, device):
    if s is None:
        bits = torch.randint(0, 2, (1, NBITS))
    else:
        s = s.strip().replace(" ", "")
        if len(s) != NBITS or set(s) - {"0", "1"}:
            raise SystemExit(f"--message must be exactly {NBITS} bits of 0/1, got {s!r}")
        bits = torch.tensor([[int(c) for c in s]], dtype=torch.long)
    return bits.to(device)


def bits_to_str(b):
    return "".join(str(int(v)) for v in b)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", help="audio to protect (any rate; resampled to 16 kHz)")
    ap.add_argument("--output", help="where to write the protected audio")
    ap.add_argument("--verify", metavar="FILE", help="read the message back out of a file")
    ap.add_argument("--message", help=f"{NBITS} bits, e.g. 1010001111110111. Random if omitted")
    ap.add_argument("--k", type=int, default=DEFAULT_K,
                    help=f"mask strength: how many bins to remove (default {DEFAULT_K})")
    ap.add_argument("--watermark-only", action="store_true", help="skip the spectral mask")
    ap.add_argument("--mask-only", action="store_true", help="skip the watermark")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--bins_file", default=DEFAULT_BINS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    os.chdir(HERE)
    dev = a.device

    # ---------------------------------------------------------------- verify
    if a.verify:
        x, sr_in = read_audio(a.verify)
        b = load_backbone(a.checkpoint, dev)
        wav = torch.tensor(x, dtype=torch.float32, device=dev).reshape(1, 1, -1)
        got = decode_bits(b, wav)
        print(f"\nfile      {a.verify}  ({len(x)/SR:.2f}s)")
        print(f"recovered {bits_to_str(got)}")
        if a.message:
            want = [int(c) for c in a.message.strip().replace(" ", "")]
            if len(want) != NBITS:
                raise SystemExit(f"--message must be {NBITS} bits")
            n = sum(int(g == w) for g, w in zip(got, want))
            print(f"expected  {bits_to_str(want)}")
            print(f"\n{n}/{NBITS} bits recovered -> "
                  f"{'TRACED' if n >= TRACE_BITS else 'NOT TRACED'} "
                  f"(threshold {TRACE_BITS}/{NBITS})")
        else:
            print("\nPass --message to check it against the one you embedded.")
        return

    if not (a.input and a.output):
        raise SystemExit("need --input and --output (or --verify FILE)")
    if a.watermark_only and a.mask_only:
        raise SystemExit("--watermark-only and --mask-only are mutually exclusive")

    # ------------------------------------------------------------- protect
    x, sr_in = read_audio(a.input)
    n0 = len(x)
    print(f"input     {a.input}  {sr_in} Hz -> {SR} Hz, {n0/SR:.2f}s")

    y, msg_str = x, None
    if not a.mask_only:
        b = load_backbone(a.checkpoint, dev)
        msg = parse_message(a.message, dev)
        msg_str = bits_to_str(msg[0].tolist())
        with torch.no_grad():
            out = b(torch.tensor(x, dtype=torch.float32, device=dev).reshape(1, 1, -1),
                    message=msg)
        y = out["recon_wm"].detach().cpu().numpy().reshape(-1).astype(np.float64)
        if len(y) != n0:                       # the codec can return a shorter tail
            y = y[:n0] if len(y) > n0 else np.pad(y, (0, n0 - len(y)))
        print(f"watermark embedded, message {msg_str}")

    if not a.watermark_only:
        d = json.load(open(a.bins_file))
        bins = select_bins(d["fratio"], a.k)
        hz = [round(bn * SR / d["n_fft"]) for bn in bins]
        y = np.asarray(notch(y, bins), dtype=np.float64).reshape(-1)
        print(f"mask applied, K={a.k}, bins {bins}")
        print(f"                       Hz   {hz}")

    assert len(y) == n0, f"length changed: {len(y)} vs {n0}"
    sf.write(a.output, np.clip(y.astype(np.float32), -1.0, 1.0), SR)
    print(f"\nwrote     {a.output}")
    if msg_str:
        print(f"\n  STORE THIS MESSAGE -- it is what makes a clone traceable:\n"
              f"      {msg_str}\n"
              f"  Check it later with:\n"
              f"      python protect.py --verify {a.output} --message {msg_str}")


if __name__ == "__main__":
    main()
