"""
audio_level_and_noise_diagnostic.py  (2026-10-08)

Two questions, answered from the WAVs the pipeline already wrote (no model, no GPU, numpy + scipy + matplotlib only):

  (1) LOUDNESS of the watermarked audio vs the original, per clip (not just the mean): RMS change, peak change, and the change in energy
      in three frequency bands.  A clip that "sounds quieter" even though the mean RMS is +0.8 dB will show up here (negative RMS change,
      lower peak, or a dull high band).
  (2) WHERE THE ANTI-CLONING NOISE SITS: how loud the PGD noise (protected - watermark-only) is relative to the speech, frame by frame,
      separately in speech frames and in pauses, and per frequency band.  This decides whether a perturbation shaped to the local loudness /
      spectrum can hide the noise at the same strength (see the printed reading guide at the end).

It can also WRITE level-matched copies (--write_matched): every watermarked / protected clip scaled to the original's RMS, i.e. the
"last step" loudness fix, without regenerating anything.

Usage (Kaggle):
    python audio_level_and_noise_diagnostic.py \
        --dirs 0.002=/kaggle/input/datasets/pakeezarasheed8/composability-audio 0.005=eps_sweep_route2/eps0.005_audio 0.008=eps_sweep_route2/eps0.008_audio \
        --show 3 7 --out /kaggle/working/noise_diag
    # level-matched copies of one directory (same file names, usable as --input_wav_dir for cloner_watermark_eval.py):
    python audio_level_and_noise_diagnostic.py --dirs 0.005=eps_sweep_route2/eps0.005_audio --write_matched /kaggle/working/matched_eps0.005

Each directory must hold sample{i}_reference.wav, sample{i}_unprotected.wav (watermark only) and sample{i}_protected.wav (watermark + PGD).
"""
import argparse, glob, os, re, sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import welch

SR = 16000
FRAME, HOP = 400, 160                    # 25 ms / 10 ms
BANDS = [(0, 1000), (1000, 4000), (4000, 8000)]
PAUSE_BELOW_DB = 30.0                    # a frame is a "pause" if it is more than 30 dB below the loudest frame of the original
EPS = 1e-12


def read(path):
    sr, x = wavfile.read(path)
    assert sr == SR, f"{path}: {sr} Hz"
    if x.dtype.kind == "i":
        x = x.astype(np.float64) / np.iinfo(x.dtype).max
    else:
        x = x.astype(np.float64)
    return x.mean(axis=1) if x.ndim > 1 else x


def write(path, x):
    wavfile.write(path, SR, (np.clip(x, -1, 1) * 32767).astype(np.int16))


def rms(x):
    return np.sqrt(np.mean(x ** 2) + EPS)


def db(x):
    return 20 * np.log10(x + 1e-9)


def frames_energy(x):
    n = 1 + (len(x) - FRAME) // HOP
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    return np.mean(x[idx] ** 2, axis=1) + EPS


def band_power(x):
    f, p = welch(x, fs=SR, nperseg=512)
    return [p[(f >= lo) & (f < hi)].sum() + EPS for lo, hi in BANDS]


def clips(d):
    out = []
    for p in sorted(glob.glob(os.path.join(d, "sample*_reference.wav"))):
        out.append(int(re.search(r"sample(\d+)_", os.path.basename(p)).group(1)))
    return sorted(out)


def match_rms(x, ref, peak=0.99):
    y = x * (rms(ref) / rms(x))
    pk = np.max(np.abs(y))
    return y * (peak / pk) if pk > peak else y


def loudness(label, d, ids, show):
    rows = []
    for i in ids:
        r, w, p = (read(os.path.join(d, f"sample{i}_{t}.wav")) for t in ("reference", "unprotected", "protected"))
        n = min(len(r), len(w), len(p)); r, w, p = r[:n], w[:n], p[:n]
        br, bw = band_power(r), band_power(w)
        rows.append(dict(i=i, wm_rms=db(rms(w)) - db(rms(r)), pr_rms=db(rms(p)) - db(rms(r)),
                         wm_peak=db(np.max(np.abs(w))) - db(np.max(np.abs(r))),
                         wm_band=[10 * np.log10(a / b) for a, b in zip(bw, br)]))
    a = np.array([r["wm_rms"] for r in rows])
    print(f"\n=== LOUDNESS, watermark-only vs original   [{label}]   n={len(rows)} clips ===")
    print(f"RMS change (dB):  mean {a.mean():+.2f}   min {a.min():+.2f}   max {a.max():+.2f}   clips quieter than -0.5 dB: {(a < -0.5).sum()}   louder than +1.5 dB: {(a > 1.5).sum()}")
    pk = np.array([r["wm_peak"] for r in rows]); pr = np.array([r["pr_rms"] for r in rows])
    print(f"peak change (dB): mean {pk.mean():+.2f}   min {pk.min():+.2f}   max {pk.max():+.2f}   |   protected RMS change mean {pr.mean():+.2f}")
    bands = np.array([r["wm_band"] for r in rows]).mean(axis=0)
    print("energy change per band, mean (dB):  " + "   ".join(f"{lo}-{hi} Hz {v:+.2f}" for (lo, hi), v in zip(BANDS, bands)))
    for r in rows:
        if r["i"] in show:
            print(f"  sample {r['i']}: RMS {r['wm_rms']:+.2f} dB, peak {r['wm_peak']:+.2f} dB, bands "
                  + " / ".join(f"{v:+.1f}" for v in r["wm_band"]) + " dB")
    return rows


def noise_stats(label, d, ids):
    """PGD noise = protected - unprotected;  WM noise = unprotected - reference. Local SNR = original frame energy / noise frame energy."""
    res = {}
    for name, a_t, b_t in (("PGD noise", "protected", "unprotected"), ("watermark", "unprotected", "reference")):
        sp, pa, bandsnr, lvl, pause_frac = [], [], [], [], []
        for i in ids:
            r = read(os.path.join(d, f"sample{i}_reference.wav"))
            a = read(os.path.join(d, f"sample{i}_{a_t}.wav")); b = read(os.path.join(d, f"sample{i}_{b_t}.wav"))
            n = min(len(r), len(a), len(b)); r, a, b = r[:n], a[:n], b[:n]
            z = a - b
            er, ez = frames_energy(r), frames_energy(z)
            er_db = 10 * np.log10(er)
            pause = er_db < er_db.max() - PAUSE_BELOW_DB
            snr = 10 * np.log10(er / ez)
            sp.append(snr[~pause]); pa.append(snr[pause]); pause_frac.append(pause.mean()); lvl.append(db(rms(z)))
            bandsnr.append([10 * np.log10(x / y) for x, y in zip(band_power(r), band_power(z))])
        sp, pa = np.concatenate(sp), (np.concatenate(pa) if any(len(x) for x in pa) else np.array([np.nan]))
        res[name] = dict(level=np.median(lvl), pause=100 * np.mean(pause_frac), sp=sp, pa=pa, band=np.mean(bandsnr, axis=0))
    return res


def print_noise(label, res):
    print(f"\n=== WHERE THE NOISE SITS   [{label}] ===")
    print(f"{'source':<11}{'level dBFS':>11}{'pause%':>8} | {'SNR speech':>11}{'SNR pause':>10}{'gap':>6} | {'speech<10dB':>12}{'pause<10dB':>11} | {'band SNR 0-1k / 1-4k / 4-8k (dB)':>34}")
    for name, v in res.items():
        s, p = np.median(v["sp"]), np.nanmedian(v["pa"])
        print(f"{name:<11}{v['level']:>11.1f}{v['pause']:>8.0f} | {s:>11.1f}{p:>10.1f}{s - p:>6.1f} | {100 * np.mean(v['sp'] < 10):>11.0f}%{100 * np.nanmean(v['pa'] < 10):>10.0f}% | "
              f"{v['band'][0]:>9.1f} / {v['band'][1]:>5.1f} / {v['band'][2]:>5.1f}")


def reading_guide():
    print("""
READING GUIDE (rules of thumb, not calibrated thresholds)
 SNR = original energy / added-noise energy, so HIGHER = noise quieter relative to the signal. Noise within ~10 dB of the signal is easy to hear.
 - 'gap' (speech SNR minus pause SNR) large (> ~15 dB) and 'pause<10dB' high: the noise is a flat floor that is far louder than the recording's own pauses.
   Shaping the noise to follow the local loudness (relative bound per frame) has real room to help.
 - band SNR much lower at 4-8k than at 0-1k: the noise is spectrally flat while speech is not; hiss dominates the high band.
   Shaping the noise to the speech spectrum (or a masking threshold) has room to help.
 - Both flat (small gap, similar band SNR): shaping buys little; the noise is simply too strong for this metric budget.
 The watermark row is the control: your ears say the watermark alone is fine, so its numbers show what 'inaudible enough' looks like here.""")


def figure(allres, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labs = list(allres)
    fig, axes = plt.subplots(2, len(labs), figsize=(4.2 * len(labs), 6.5), squeeze=False)
    for c, lab in enumerate(labs):
        v = allres[lab]["PGD noise"]
        ax = axes[0, c]
        ax.hist(v["sp"], bins=60, alpha=0.6, label="speech frames", density=True)
        ax.hist(v["pa"][~np.isnan(v["pa"])], bins=60, alpha=0.6, label="pause frames", density=True)
        ax.axvline(10, color="k", ls="--", lw=0.8); ax.set_title(f"eps {lab}: local SNR of the PGD noise (dB)", fontsize=9); ax.legend(fontsize=7)
        ax2 = axes[1, c]
        ax2.bar(["0-1k", "1-4k", "4-8k"], v["band"], color="#4c72b0"); ax2.axhline(10, color="k", ls="--", lw=0.8)
        ax2.set_title("band SNR (dB), higher = quieter noise", fontsize=9)
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
    print("wrote", out_png)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+", required=True, help="label=path pairs")
    ap.add_argument("--show", nargs="*", type=int, default=[3, 7], help="clips whose loudness numbers are printed individually")
    ap.add_argument("--max_clips", type=int, default=100)
    ap.add_argument("--out", default=None, help="folder for noise_location.png")
    ap.add_argument("--write_matched", default=None, help="write RMS-matched copies of the (single) --dirs directory here")
    a = ap.parse_args()
    pairs = [x.split("=", 1) for x in a.dirs]

    if a.write_matched:
        assert len(pairs) == 1, "--write_matched takes exactly one directory"
        d = pairs[0][1]; os.makedirs(a.write_matched, exist_ok=True)
        for i in clips(d)[:a.max_clips]:
            r = read(os.path.join(d, f"sample{i}_reference.wav")); write(os.path.join(a.write_matched, f"sample{i}_reference.wav"), r)
            for t in ("unprotected", "protected"):
                x = read(os.path.join(d, f"sample{i}_{t}.wav"))
                write(os.path.join(a.write_matched, f"sample{i}_{t}.wav"), match_rms(x, r))
        print(f"wrote level-matched copies to {a.write_matched}. Re-check the watermark on them before trusting them "
              f"(cloner_watermark_eval.py prints 'ACC on watermarked source audio' first; it should stay ~0.99).")
        return

    allres = {}
    for label, d in pairs:
        ids = clips(d)[:a.max_clips]
        if not ids:
            print(f"[skip] {label}: no sample*_reference.wav in {d}"); continue
        loudness(label, d, ids, set(a.show))
        allres[label] = noise_stats(label, d, ids)
        print_noise(label, allres[label])
    reading_guide()
    if a.out and allres:
        os.makedirs(a.out, exist_ok=True); figure(allres, os.path.join(a.out, "noise_location.png"))


if __name__ == "__main__":
    main()
