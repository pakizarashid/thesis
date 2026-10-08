"""
robustness_attacks.py

Applies post-hoc signal-processing attacks to a directory of protected (watermarked +
PGD-perturbed) WAV files, producing a new directory of attacked WAVs in the SAME layout
and filenames as the input — so your existing eval scripts (cloner_watermark_eval.py in
pre-made-audio mode, ecapa_sim_eval.py) can be pointed at the output exactly the way they
already are for demucs_fallback_eval.py's output. This script does NOT do any cloning or
watermark detection itself — same division of labor as the DEMUCS step: this only produces
the attacked audio, your existing scripts do the measurement.

Attacks implemented, each at two severities (mild / aggressive), matching the two-point
"trend + confirm" convention already used for the epsilon sweep rather than a single
arbitrary operating point:

  mp3        -- MP3 lossy round-trip via ffmpeg.      mild=128kbps   aggressive=32kbps
  opus       -- Opus lossy round-trip via ffmpeg.      mild=64kbps    aggressive=16kbps
  resample   -- downsample/upsample round-trip.        mild=16k->22.05k->16k
                                                        aggressive=16k->8k->16k (telephone-band)
  amplitude  -- linear gain scaling.                   mild=+/-3dB    aggressive=+/-6dB
                                                        (randomly up or down per utterance,
                                                         seeded per-file for reproducibility)
  noise      -- additive white Gaussian noise.         mild=20dB SNR  aggressive=10dB SNR

Usage:
    python robustness_attacks.py --input_dir composability_quality_f5tts_protected \
        --output_dir attacked_mp3_mild_f5tts_protected --attack mp3 --severity mild

    # or run the whole battery against one input dir in one call:
    python robustness_attacks.py --input_dir <protected_dir> --output_root attacked \
        --all --tag f5tts_protected

Requires: numpy, scipy (both already used elsewhere in this project -- deliberately NOT
using soundfile/librosa here to avoid another libsndfile-style environment landmine; wav
read/write goes through scipy.io.wavfile only).
ffmpeg must be on PATH for mp3/opus (check with `which ffmpeg`; `apt-get install -y ffmpeg`
if missing -- do this once per Kaggle session, it does not persist).
"""

import argparse
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


def _read_wav_float32(path):
    sr, audio = wavfile.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    else:
        audio = audio.astype(np.float32)
    return sr, audio


def _write_wav_float32(path, sr, audio):
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    int16 = (audio * 32767.0).astype(np.int16)
    wavfile.write(path, sr, int16)

SEVERITIES = {
    "mp3":       {"mild": 128, "aggressive": 32},          # kbps
    "opus":      {"mild": 64,  "aggressive": 16},           # kbps
    "resample":  {"mild": 22050, "aggressive": 8000},        # intermediate sample rate (Hz)
    "amplitude": {"mild": 3.0, "aggressive": 6.0},           # +/- dB
    "noise":     {"mild": 20.0, "aggressive": 10.0},         # target SNR in dB
}


def _run_ffmpeg_roundtrip(wav_path, out_path, codec, bitrate_kbps, sr):
    """Encode wav_path to `codec` at bitrate_kbps, decode back to wav at sr, write to out_path."""
    with tempfile.NamedTemporaryFile(suffix={"mp3": ".mp3", "opus": ".opus"}[codec], delete=False) as tmp:
        tmp_compressed = tmp.name
    try:
        enc_cmd = [
            "ffmpeg", "-y", "-loglevel", "error", "-i", wav_path,
            "-b:a", f"{bitrate_kbps}k",
        ]
        if codec == "opus":
            enc_cmd += ["-c:a", "libopus"]
        enc_cmd += [tmp_compressed]
        subprocess.run(enc_cmd, check=True)

        dec_cmd = [
            "ffmpeg", "-y", "-loglevel", "error", "-i", tmp_compressed,
            "-ar", str(sr), "-ac", "1", out_path,
        ]
        subprocess.run(dec_cmd, check=True)
    finally:
        if os.path.exists(tmp_compressed):
            os.remove(tmp_compressed)


def attack_mp3(wav_path, out_path, severity, sr):
    _run_ffmpeg_roundtrip(wav_path, out_path, "mp3", SEVERITIES["mp3"][severity], sr)


def attack_opus(wav_path, out_path, severity, sr):
    _run_ffmpeg_roundtrip(wav_path, out_path, "opus", SEVERITIES["opus"][severity], sr)


def attack_resample(wav_path, out_path, severity, sr, seed=None):
    native_sr, audio = _read_wav_float32(wav_path)
    intermediate_sr = SEVERITIES["resample"][severity]
    # native -> intermediate -> native, using polyphase resampling (avoids libresample
    # dependency mismatches across environments -- scipy only)
    down = resample_poly(audio, intermediate_sr, native_sr)
    back = resample_poly(down, native_sr, intermediate_sr)
    # length can drift by a few samples due to rounding; pad/crop to match original
    if len(back) < len(audio):
        back = np.pad(back, (0, len(audio) - len(back)))
    else:
        back = back[: len(audio)]
    _write_wav_float32(out_path, sr, back)


def attack_amplitude(wav_path, out_path, severity, sr, seed):
    native_sr, audio = _read_wav_float32(wav_path)
    db = SEVERITIES["amplitude"][severity]
    rng = np.random.RandomState(seed)
    sign = 1.0 if rng.rand() < 0.5 else -1.0
    gain = 10 ** ((sign * db) / 20.0)
    scaled = audio * gain
    # clip to [-1, 1] happens inside _write_wav_float32 -- this is itself part of the
    # attack's realism at +6dB (a real attacker's re-amplification can clip too)
    _write_wav_float32(out_path, sr, scaled)


def attack_noise(wav_path, out_path, severity, sr, seed):
    native_sr, audio = _read_wav_float32(wav_path)
    target_snr_db = SEVERITIES["noise"][severity]
    rng = np.random.RandomState(seed)
    signal_power = np.mean(audio ** 2)
    noise_power = signal_power / (10 ** (target_snr_db / 10.0))
    noise = rng.normal(0, np.sqrt(noise_power), size=audio.shape)
    noisy = audio + noise
    _write_wav_float32(out_path, sr, noisy)


ATTACK_FNS = {
    "mp3": attack_mp3,
    "opus": attack_opus,
    "resample": attack_resample,
    "amplitude": attack_amplitude,
    "noise": attack_noise,
}


def run_one_attack(input_dir, output_dir, attack, severity, sr=16000, pattern="*.wav"):
    os.makedirs(output_dir, exist_ok=True)
    wavs = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not wavs:
        print(f"WARNING: no files matching {pattern!r} found in {input_dir}", file=sys.stderr)
        return 0
    fn = ATTACK_FNS[attack]
    n_ok = 0
    for i, wav_path in enumerate(wavs):
        fname = os.path.basename(wav_path)
        out_path = os.path.join(output_dir, fname)
        try:
            if attack in ("mp3", "opus"):
                fn(wav_path, out_path, severity, sr)
            else:
                fn(wav_path, out_path, severity, sr, seed=i)
            n_ok += 1
        except Exception as e:
            print(f"FAILED on {fname}: {e}", file=sys.stderr)
    print(f"[{attack}/{severity}] {n_ok}/{len(wavs)} files written to {output_dir}")
    return n_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Directory of protected WAVs to attack")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--tag", default="", help="Suffix used in --all mode output dir names")
    ap.add_argument("--pattern", default="*.wav",
                     help="Glob pattern (relative to --input_dir) selecting which files to attack. "
                          "Use this when input_dir has multiple arms mixed together, e.g. "
                          "'*_protected.wav' to attack only the protected arm and skip "
                          "*_reference.wav / *_unprotected.wav sitting in the same directory.")

    single = ap.add_argument_group("single-attack mode")
    single.add_argument("--output_dir")
    single.add_argument("--attack", choices=list(ATTACK_FNS.keys()))
    single.add_argument("--severity", choices=["mild", "aggressive"])

    batch = ap.add_argument_group("batch mode")
    batch.add_argument("--all", action="store_true",
                        help="Run all 5 attacks x 2 severities (10 output dirs) against input_dir")
    batch.add_argument("--output_root", default="attacked")

    args = ap.parse_args()

    if args.all:
        for attack in ATTACK_FNS:
            for severity in ("mild", "aggressive"):
                out_dir = f"{args.output_root}_{attack}_{severity}_{args.tag}".rstrip("_")
                run_one_attack(args.input_dir, out_dir, attack, severity, args.sr, args.pattern)
    else:
        if not (args.output_dir and args.attack and args.severity):
            ap.error("single-attack mode requires --output_dir --attack --severity (or pass --all)")
        run_one_attack(args.input_dir, args.output_dir, args.attack, args.severity, args.sr, args.pattern)


if __name__ == "__main__":
    main()
