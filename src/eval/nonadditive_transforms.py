"""
Non-additive protection transforms -- pre-registration N1
(claude/pre-registration-nonadditive-2026-11-03.md, revised 2026-11-03).

The six mechanism families tried so far all ADD signal to the published audio
(perturbation energy, or watermark imprint) and all trace one
protection-versus-perceptual-cost curve. N1 tests whether a transform that adds
nothing sits below that curve.

Design rules, enforced here rather than trusted:

  * NO speaker verifier and NO cloner is imported or called anywhere in this
    file. Bin selection uses a speaker F-ratio -- between-speaker variance of
    per-speaker mean log-magnitude over mean within-speaker variance, per
    frequency bin -- computed on the 60 TRAINING speakers only. That is a
    property of speech, not of a verifier, so in-loop inflation is impossible
    by construction. It is NOT evidence that the selected bins are causally
    necessary for speaker identity: it ranks bins by between- versus
    within-speaker variation, and the experiment tests whether suppressing them
    actually reduces clone identity preservation.
  * Every transform preserves sample count EXACTLY. clone_eval_v2 asserts
    prompt length equality, and a length change would also silently alter the
    cloner's prompt.
  * Every transform is RMS-matched to its input, so no arm can obtain its
    effect from added loudness. Six families already trace the loudness curve;
    an unmatched arm would just be a seventh point on it.

UNTESTED AS WRITTEN -- no GPU, data or deps were available when this was
authored. Run `--diagnostic` FIRST: it checks the invariants above on 2 clips
and prints every quantity the pre-registration needs, without writing an arm.

Subcommands
  fratio     compute + freeze the per-bin speaker F-ratio on the train split
  calibrate  sweep K / L on clips 0-19, interpolate to a target PESQ, freeze
  generate   write an arm's prompts in protect_prompts_v2's directory layout
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("models", "data", "losses", "."):
    sys.path.insert(0, os.path.join(HERE, "..", sub))
sys.path.insert(0, HERE)

import clone_eval_v2 as ce                                    # noqa: E402
import cloner_watermark_eval as cwe                           # noqa: E402
from quality_metrics import compute_pesq_stoi, compute_si_snr  # noqa: E402

SR = 16000
N_FFT, HOP = 2048, 512
N_BINS = N_FFT // 2 + 1          # 1025, as in ClearMask


# --------------------------------------------------------------------------- io
def write_flac(path, wav):
    import soundfile as sf
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x = np.clip(np.asarray(wav, dtype=np.float32).reshape(-1), -1.0, 1.0)
    sf.write(path, x, SR, format="FLAC", subtype="PCM_16")


def wav_sha(x):
    a = np.ascontiguousarray(np.asarray(x, dtype=np.float32).reshape(-1))
    return hashlib.sha256(a.tobytes()).hexdigest()


def as1d(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).reshape(-1)


# ------------------------------------------------------------------- transforms
def _stft(x):
    return torch.stft(torch.from_numpy(x).float(), n_fft=N_FFT, hop_length=HOP,
                      window=torch.hann_window(N_FFT), center=True,
                      return_complex=True)


def _istft(X, length):
    return torch.istft(X, n_fft=N_FFT, hop_length=HOP,
                       window=torch.hann_window(N_FFT), center=True,
                       length=length).numpy().astype(np.float64)


def rms_match(y, x):
    """Scale y to x's RMS. Keeps every arm off the loudness axis."""
    ry, rx = float(np.sqrt(np.mean(y ** 2))), float(np.sqrt(np.mean(x ** 2)))
    if ry < 1e-12:
        return y
    return y * (rx / ry)


def notch(x, bins):
    # K=0 means no notch: the watermark-only arm. RMS-matching x to
    # itself is the identity, so the invariants still hold.
    if bins is None or len(bins) == 0:
        return rms_match(x, x)
    """Zero the given STFT bins. Non-additive: removes, never adds."""
    x = as1d(x)
    X = _stft(x)
    X[torch.as_tensor(np.asarray(bins, dtype=np.int64)), :] = 0
    return rms_match(_istft(X, len(x)), x)


def make_rir(length_ms, seed=1234, direct_gain=1.0, decay_db=60.0):
    """Exponentially decaying noise tail after a unit direct path.

    Short by design (<= 30 ms, as in ClearMask): long enough to smear fine
    spectral structure, too short to be heard as a room.
    """
    n = max(2, int(round(SR * length_ms / 1000.0)))
    g = np.random.default_rng(seed)
    tail = g.standard_normal(n)
    t = np.arange(n) / float(n)
    tail *= 10.0 ** (-decay_db * t / 20.0)
    h = tail
    h[0] = direct_gain
    return h / np.sqrt(np.sum(h ** 2))


def apply_rir(x, h):
    """Convolve, then trim to the original length. Non-additive: redistributes."""
    x = as1d(x)
    y = np.convolve(x, np.asarray(h, dtype=np.float64), mode="full")[:len(x)]
    return rms_match(y, x)


def transform(x, arm, bins=None, rir=None):
    if arm == "n1notch":
        return notch(x, bins)
    if arm == "n1rir":
        return apply_rir(x, rir)
    if arm == "n1both":
        return rms_match(apply_rir(notch(x, bins), rir), as1d(x))
    raise ValueError("unknown arm %r" % arm)


# ----------------------------------------------------------------- the F-ratio
def band_mask(lo_hz, hi_hz, mean_mag, floor_db):
    """Which bins are eligible at all (Amendment A1).

    Two exclusions, both decided before any clip exists:
      * outside [lo_hz, hi_hz] -- speech carries almost no energy below ~80 Hz,
        and log(mag + 1e-8) there is dominated by the recording noise floor and
        DC offset, so between-speaker variance is large for reasons that have
        nothing to do with identity. The unamended criterion selected bins
        0-16 (0-125 Hz) for exactly that reason.
      * mean magnitude more than `floor_db` below the loudest bin -- the same
        near-silence problem, measured rather than assumed.
    """
    k = np.arange(N_BINS)
    hz = k * SR / float(N_FFT)
    m = (hz >= lo_hz) & (hz <= hi_hz)
    ref = float(np.max(mean_mag)) if np.max(mean_mag) > 0 else 1.0
    db = 20.0 * np.log10(np.maximum(mean_mag, 1e-20) / ref)
    return m & (db >= floor_db)


def speaker_fratio(ds, min_per_speaker=3, lo_hz=80.0, hi_hz=7800.0,
                   floor_db=-60.0, level_norm=True):
    """Per-bin F-ratio over the speakers in ds.

    F[k] = Var_between(mu_s[k]) / mean_s(Var_within_s[k])

    mu_s[k] is speaker s's mean log-magnitude in bin k; Var_within_s[k] is the
    variance of that speaker's frame-level log-magnitudes. A high F[k] means
    bin k separates speakers well relative to its variation inside a speaker.
    No verifier is consulted.
    """
    per_spk = {}
    mag_sum = np.zeros(N_BINS, dtype=np.float64)
    mag_n = 0
    for i in range(len(ds)):
        it = ds[i]
        spk = str(it["speaker_id"])
        x = as1d(it["waveform"])
        mag = _stft(x).abs().numpy().astype(np.float64)          # [bins, frames]
        mag_sum += mag.sum(axis=1)
        mag_n += mag.shape[1]
        lm = np.log(mag + 1e-8)
        if level_norm:
            # Amendment A1: remove each frame's overall level so the criterion
            # ranks SPECTRAL SHAPE, not loudness. Without this, gain and
            # recording-level differences between speakers inflate every bin.
            lm = lm - lm.mean(axis=0, keepdims=True)
        per_spk.setdefault(spk, []).append(lm)
    mean_mag = mag_sum / max(mag_n, 1)

    mus, withins = [], []
    used = 0
    for spk, chunks in per_spk.items():
        lm = np.concatenate(chunks, axis=1)
        if lm.shape[1] < min_per_speaker:
            continue
        mus.append(lm.mean(axis=1))
        withins.append(lm.var(axis=1))
        used += 1
    if used < 2:
        raise RuntimeError("need >= 2 speakers with frames; got %d" % used)

    mus = np.stack(mus)                                           # [spk, bins]
    withins = np.stack(withins)
    between = mus.var(axis=0)
    within = withins.mean(axis=0)
    f = between / np.maximum(within, 1e-12)
    elig = band_mask(lo_hz, hi_hz, mean_mag, floor_db)
    f = np.where(elig, f, -np.inf)          # ineligible bins can never be selected
    return f, used, int(sum(c.shape[1] for cs in per_spk.values() for c in cs)), elig, mean_mag


def select_bins(f, k):
    f = np.asarray(f, dtype=np.float64)
    order = [int(b) for b in np.argsort(f)[::-1] if np.isfinite(f[int(b)])]
    assert len(order) >= int(k), "only %d eligible bins, need %d" % (len(order), k)
    return sorted(order[:int(k)])


# ------------------------------------------------------------------- quality
def quality(clean, wav):
    c, w = as1d(clean), as1d(wav)
    pq, st = compute_pesq_stoi(c, w, SR)
    noise = w - c
    return {"pesq": float(pq), "stoi": float(st),
            "si_snr": float(compute_si_snr(c, w)),
            "snr_db": float(10 * np.log10(np.mean(c ** 2) / max(np.mean(noise ** 2), 1e-20))),
            "rms_dbfs": float(20 * np.log10(max(np.sqrt(np.mean(w ** 2)), 1e-12))),
            "rms_dbfs_clean": float(20 * np.log10(max(np.sqrt(np.mean(c ** 2)), 1e-12))),
            "len_ok": bool(len(c) == len(w))}


# ------------------------------------------------------------- watermarked audio
def load_backbone(args, device):
    """Exactly the loader clone_eval_v2 and protect_prompts_v2 use, so the
    watermarked audio here is bit-identical to theirs."""
    b = cwe.build_backbone(args.checkpoint, args.lora_r, args.lora_alpha,
                           msgproc_lora_r=args.msgproc_lora_r)
    b.model.to(device)
    return b


def watermark(backbone, clean_t, msg):
    with torch.no_grad():
        return backbone.forward_full(clean_t, msg)["recon_wm"]


# ------------------------------------------------------------------------ main
def add_common(p):
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--msgproc_lora_r", type=int, default=None)
    p.add_argument("--dataset", choices=("librispeech", "vctk"), default="librispeech")
    p.add_argument("--data_root", default="./data/librispeech")
    p.add_argument("--libri_subset", default="train-clean-100")
    p.add_argument("--vctk_root", default=None)
    p.add_argument("--n_speakers", type=int, default=60)
    p.add_argument("--utterances_per_speaker", type=int, default=15)
    p.add_argument("--n_eval_speakers", type=int, default=20)
    p.add_argument("--eval_utterances_per_speaker", type=int, default=5)
    p.add_argument("--max_full_seconds", type=float, default=20.0)
    p.add_argument("--n_utterances", type=int, default=100)
    p.add_argument("--prompt_seconds", default="3")
    p.add_argument("--message_seed_base", type=int, default=123)
    p.add_argument("--rir_seed", type=int, default=1234)
    p.add_argument("--bins_file", default="results/n1_fratio_bins.json")
    p.add_argument("--calib_file", default="results/n1_calib.json")


def train_split(args):
    """The 60 TRAINING speakers. The F-ratio must never see the eval speakers."""
    if getattr(args, 'dataset', 'librispeech') == 'vctk':
        from vctk import VCTKSubset
        return _TrimmedDataset(VCTKSubset(root=args.vctk_root, n_speakers=args.n_speakers,
                          utterances_per_speaker=args.utterances_per_speaker,
                          n_eval_speakers=args.n_eval_speakers,
                          eval_utterances_per_speaker=args.eval_utterances_per_speaker,
                          sample_rate=SR, crop_seconds=args.max_full_seconds,
                          split="train"))
    from librispeech import LibriSpeechSubset
    return _TrimmedDataset(LibriSpeechSubset(root=args.data_root, n_speakers=args.n_speakers,
                             utterances_per_speaker=args.utterances_per_speaker,
                             n_eval_speakers=args.n_eval_speakers,
                             eval_utterances_per_speaker=args.eval_utterances_per_speaker,
                             sample_rate=SR, crop_seconds=args.max_full_seconds,
                             split="train", subset=args.libri_subset))


def eval_prompts(args, device, backbone, n):
    """Yield (i, item, clean_prompt_1d, watermarked_1d, true_bits) per decisive clip."""
    ds = ce.load_dataset(args, crop=args.max_full_seconds)
    globals()["_DS"] = ds
    spec = args.prompt_seconds
    for i in range(min(n, len(ds))):
        it = ds[i]
        full = ce.trim_padding(it["waveform"].reshape(-1))      # as clone_eval_v2 does
        prompt = ce.make_prompt(full, spec, args.max_full_seconds) \
            .to(device).reshape(1, 1, -1)
        msg = ce.message_bits(args.message_seed_base + i, device)
        true_bits = np.array(msg[0].tolist())
        pin = getattr(args, "prompt_in", None)
        if pin:
            # N3: the base signal is an existing arm's published audio (e.g. q2).
            import soundfile as _sf
            x, sr = _sf.read(os.path.join(pin, "%d.flac" % i), dtype="float32")
            assert sr == SR, (pin, i, sr)
            base = np.asarray(x, dtype=np.float64).reshape(-1)
            assert len(base) == prompt.shape[-1], \
                "clip %d: %s is %d samples, the clean prompt is %d -- not the same clips" \
                % (i, pin, len(base), prompt.shape[-1])
        else:
            base = as1d(watermark(backbone, prompt, msg))
        yield i, it, as1d(prompt), base, true_bits


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fratio", help="compute and freeze the per-bin speaker F-ratio")
    add_common(p)
    p.add_argument("--k_list", default="8,16,32,64,128")
    p.add_argument("--band_hz_lo", type=float, default=80.0)
    p.add_argument("--band_hz_hi", type=float, default=7800.0)
    p.add_argument("--energy_floor_db", type=float, default=-60.0)
    p.add_argument("--no_level_norm", action="store_true")

    p = sub.add_parser("calibrate", help="sweep to a target PESQ on clips 0-19, then freeze")
    add_common(p)
    p.add_argument("--arm", required=True, choices=("n1notch", "n1rir"))
    p.add_argument("--target_pesq", type=float, required=True)
    p.add_argument("--k_list", default="8,16,32,64,128")
    p.add_argument("--l_list", default="5,10,20,30")
    p.add_argument("--n_calib", type=int, default=20)

    p = sub.add_parser("generate", help="write an arm's prompts")
    add_common(p)
    p.add_argument("--arm", required=True, choices=("n1notch", "n1rir", "n1both"))
    p.add_argument("--out_dir", required=True)
    p.add_argument("--diagnostic", action="store_true",
                   help="2 clips, print everything, write nothing. RUN THIS FIRST.")
    p.add_argument("--k", type=int, default=None,
                   help="Amendment A5 (curve design): notch K explicitly, bypassing "
                        "calibration. PESQ is then REPORTED, not matched.")
    p.add_argument("--l_ms", type=int, default=None,
                   help="Amendment A5: RIR length explicitly, bypassing calibration.")
    p.add_argument("--prompt_in", default=None,
                   help="N3: apply the transform to EXISTING prompt audio in this directory "
                        "(e.g. .../RQA/q2) instead of to freshly watermarked audio. The clean "
                        "prompt is still used as the PESQ reference and for message bits, so "
                        "the numbers stay comparable with every other arm.")
    p.add_argument("--out_arm_name", default=None,
                   help="N3: name of the output subfolder and manifest arm key "
                        "(default: --arm). Use e.g. c1 so composed arms are not confused "
                        "with the plain notch arms.")
    a = ap.parse_args()

    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("results", exist_ok=True)

    # ------------------------------------------------------------------ fratio
    if a.cmd == "fratio":
        ds = train_split(a)
        print("[fratio] train split: %d items" % len(ds))
        f, n_spk, n_frames, elig, mean_mag = speaker_fratio(
            ds, lo_hz=a.band_hz_lo, hi_hz=a.band_hz_hi,
            floor_db=a.energy_floor_db, level_norm=not a.no_level_norm)
        n_elig = int(elig.sum())
        assert n_elig >= 128, "only %d eligible bins -- band/floor too strict" % n_elig
        out = {"criterion_version": "A1", "n_fft": N_FFT, "hop": HOP,
               "n_bins": N_BINS, "sr": SR,
               "n_speakers_used": n_spk, "n_frames": n_frames,
               "split": "train", "libri_subset": a.libri_subset,
               "band_hz": [a.band_hz_lo, a.band_hz_hi],
               "energy_floor_db": a.energy_floor_db,
               "level_norm": not a.no_level_norm,
               "n_eligible_bins": n_elig,
               "fratio": [None if not bool(e) else float(v) for v, e in zip(f, elig)],
               "mean_mag_db": [float(20 * np.log10(max(v, 1e-20) /
                               max(float(np.max(mean_mag)), 1e-20))) for v in mean_mag],
               "bins_by_k": {k: select_bins(f, int(k)) for k in a.k_list.split(",")},
               "note": "Amendment A1. Verifier-independent: ranks bins by between- vs "
                       "within-speaker variation of per-frame level-normalised log-magnitude "
                       "on TRAIN speakers only, restricted to the speech band and to bins "
                       "above an energy floor. Not a causal claim about speaker identity."}
        json.dump(out, open(a.bins_file, "w"), indent=1)
        top = select_bins(f, 16)
        print("[fratio] %d speakers, %d frames, %d eligible bins" % (n_spk, n_frames, n_elig))
        print("[fratio] top-16 bins:", top)
        print("[fratio] as Hz:", [round(b * SR / N_FFT, 1) for b in top])
        print("[fratio] saved", a.bins_file)
        return

    backbone = load_backbone(a, device)
    bins_all = json.load(open(a.bins_file)) if os.path.exists(a.bins_file) else None

    # --------------------------------------------------------------- calibrate
    if a.cmd == "calibrate":
        if bins_all is None:
            raise SystemExit("run `fratio` first")
        f = np.array([(-np.inf if v is None else float(v))
                      for v in bins_all["fratio"]], dtype=np.float64)
        grid = [int(v) for v in (a.k_list if a.arm == "n1notch" else a.l_list).split(",")]
        # the floor: what the watermark alone costs against clean. The transform can
        # only move PESQ DOWN from here, so if the target is above this the arm is
        # not calibratable for a reason that has nothing to do with the transform.
        wm_pq = []
        for i, it, clean, wm, _tb in eval_prompts(a, device, backbone, a.n_calib):
            wm_pq.append(quality(clean, wm)["pesq"])
        wm_floor = float(np.mean(wm_pq))
        print("[calib] watermark-only PESQ vs clean (no transform) = %.4f ; target = %.4f"
              % (wm_floor, a.target_pesq))
        if wm_floor < a.target_pesq:
            print("[calib] *** the watermark ALONE is already below the target. No setting of "
                  "this transform can reach it. Report 'not calibratable' and stop.")
        rows = []
        for g in grid:
            bins = select_bins(f, g) if a.arm == "n1notch" else None
            rir = None if a.arm == "n1notch" else make_rir(g, seed=a.rir_seed)
            pqs = []
            for i, it, clean, wm, _tb in eval_prompts(a, device, backbone, a.n_calib):
                y = transform(wm, a.arm, bins=bins, rir=rir)
                assert len(y) == len(wm), (i, len(y), len(wm))
                # Amendment A2: reference is the CLEAN prompt, matching
                # protect_prompts_v2's quality(clean, back). Measuring against
                # the watermarked audio excludes the watermark's own cost and
                # makes Q2's target unreachable by construction.
                pqs.append(quality(clean, y)["pesq"])
            rows.append((g, float(np.mean(pqs))))
            print("[calib] %s=%s  PESQ=%.4f" % ("K" if a.arm == "n1notch" else "L_ms",
                                                g, rows[-1][1]))
        # Amendment A4: interpolate only across ADJACENT grid settings, then MEASURE
        # the result and verify it. The A3 grids are non-monotone in PESQ, so picking
        # the nearest bracket by PESQ can straddle a dip and return a setting whose
        # real PESQ is nowhere near the interpolated line. That produced K=12 from a
        # 16<->4 bracket spanning the dip at 8, measuring ~1.70 against a 1.8768
        # target.
        by_setting = sorted(rows, key=lambda r: r[0])
        tol = 0.05                      # the registered gate width
        cand = []
        for (g0, p0), (g1, p1) in zip(by_setting, by_setting[1:]):
            if (p0 - a.target_pesq) * (p1 - a.target_pesq) <= 0:      # adjacent straddle
                if p1 == p0:
                    cand.append(float(g0)); continue
                w = (a.target_pesq - p0) / (p1 - p0)
                cand.append(float(g0) + w * (g1 - g0))
        setting = None
        probe = []
        if not cand:
            print("[calib] no ADJACENT pair of grid settings straddles the target -- the arm "
                  "is not calibratable on this grid (extend once per the pre-registration, "
                  "else report it dropped)")
        else:
            # verify by measurement: try the interpolated integer and its neighbours,
            # and keep the one whose MEASURED PESQ is closest to the target.
            seen = set()
            order = []
            for c in cand:
                for cc in (int(round(c)), int(np.floor(c)), int(np.ceil(c)),
                           int(np.floor(c)) - 1, int(np.ceil(c)) + 1):
                    if cc >= 1 and cc not in seen:
                        seen.add(cc); order.append(cc)
            for cc in order:
                hit = [r for r in rows if r[0] == cc]
                if hit:
                    pq = hit[0][1]
                else:
                    bins_c = select_bins(f, cc) if a.arm == "n1notch" else None
                    rir_c = None if a.arm == "n1notch" else make_rir(cc, seed=a.rir_seed)
                    pqs = []
                    for i, it, clean, wm, _tb in eval_prompts(a, device, backbone, a.n_calib):
                        y = transform(wm, a.arm, bins=bins_c, rir=rir_c)
                        pqs.append(quality(clean, y)["pesq"])
                    pq = float(np.mean(pqs))
                probe.append((cc, pq))
                print("[calib] verify %s=%d -> measured PESQ %.4f (target %.4f, |d|=%.4f)"
                      % ("K" if a.arm == "n1notch" else "L_ms", cc, pq,
                         a.target_pesq, abs(pq - a.target_pesq)))
                if abs(pq - a.target_pesq) <= tol:
                    setting = float(cc)
                    break
            if setting is None:
                best = min(probe, key=lambda r: abs(r[1] - a.target_pesq))
                print("[calib] NO setting measured within %.2f of the target. Closest is "
                      "%s=%d at PESQ %.4f (|d|=%.4f). Reporting NOT CALIBRATABLE rather than "
                      "freezing a setting that fails the registered quality condition."
                      % (tol, "K" if a.arm == "n1notch" else "L_ms", best[0], best[1],
                         abs(best[1] - a.target_pesq)))
        calib = json.load(open(a.calib_file)) if os.path.exists(a.calib_file) else {}
        if a.arm in calib:
            print("[calib] %s ALREADY FROZEN as %r -- not overwriting (pre-registration: "
                  "no retuning after the fact)" % (a.arm, calib[a.arm]))
            return
        calib[a.arm] = {"grid": rows, "verify_probe": probe, "tol": 0.05,
                        "target_pesq": a.target_pesq,
                        "setting": setting,
                        "setting_rounded": None if setting is None else int(round(setting)),
                        "n_calib": a.n_calib, "clips": "0-%d" % (a.n_calib - 1),
                        "note": "PESQ only; no clone, accept rate or verifier score entered this."}
        json.dump(calib, open(a.calib_file, "w"), indent=1)
        print("[calib] frozen:", json.dumps(calib[a.arm]["setting_rounded"]))
        print("[calib] saved", a.calib_file)
        return

    # ---------------------------------------------------------------- generate
    forced = (a.k is not None) or (a.l_ms is not None)
    if bins_all is None:
        raise SystemExit("run `fratio` first -- the bin ranking must be frozen before any clip")
    if not forced and not os.path.exists(a.calib_file):
        raise SystemExit("run `calibrate` first, or pass --k / --l_ms under Amendment A5 "
                         "(curve design: PESQ reported, not matched)")
    calib = json.load(open(a.calib_file)) if os.path.exists(a.calib_file) else {}
    f = np.array([(-np.inf if v is None else float(v))
                  for v in bins_all["fratio"]], dtype=np.float64)
    need = {"n1notch": ("n1notch",), "n1rir": ("n1rir",), "n1both": ("n1notch", "n1rir")}[a.arm]
    if forced:
        K = a.k if "n1notch" in need else None
        L = a.l_ms if "n1rir" in need else None
        if "n1notch" in need and K is None:
            raise SystemExit("--k is required for %s under A5" % a.arm)
        if "n1rir" in need and L is None:
            raise SystemExit("--l_ms is required for %s under A5" % a.arm)
        print("[gen] A5 curve design: setting passed explicitly, PESQ will be REPORTED "
              "rather than matched to a target")
    else:
        for nm in need:
            if nm not in calib or calib[nm].get("setting_rounded") is None:
                raise SystemExit("%s is not calibrated/frozen" % nm)
        K = calib["n1notch"]["setting_rounded"] if "n1notch" in need else None
        L = calib["n1rir"]["setting_rounded"] if "n1rir" in need else None
    bins = select_bins(f, K) if K else None
    rir = make_rir(L, seed=a.rir_seed) if L else None
    out_name = getattr(a, "out_arm_name", None) or a.arm
    if getattr(a, "prompt_in", None):
        assert os.path.isdir(a.prompt_in), "--prompt_in %s is not a directory" % a.prompt_in
        print("[gen] N3 composition: transform applied to EXISTING audio in %s" % a.prompt_in)
    print("[gen] arm=%s out_name=%s K=%s L=%s" % (a.arm, out_name, K, L))

    ck_sha = ce.sha256(a.checkpoint)
    n = 2 if a.diagnostic else a.n_utterances
    rec_dir = os.path.join(a.out_dir, "records")
    recs = []
    for i, it, clean, wm, true_bits in eval_prompts(a, device, backbone, n):
        y = transform(wm, a.arm, bins=bins, rir=rir)
        q = quality(clean, y)                     # A2: vs CLEAN, comparable to Q2
        q_vs_wm = quality(wm, y)                  # reported too: the transform's own cost
        q["pesq_vs_wm"] = q_vs_wm["pesq"]
        q["stoi_vs_wm"] = q_vs_wm["stoi"]
        q["rms_dbfs_wm"] = q_vs_wm["rms_dbfs_clean"]
        # watermark accuracy ON THE TRANSFORMED FILE, measured from the file an
        # attacker gets. HN2 lives or dies here: notching may remove the bins
        # the message rides in.
        y_t = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(device).reshape(1, 1, -1)
        dec = np.array(ce.decode_bits(backbone, y_t))
        q["acc_source"] = float((dec == true_bits).mean())
        q["traced_source"] = bool(q["acc_source"] >= 13.0 / 16.0)
        assert q["len_ok"], "length changed on clip %d" % i
        # RMS is matched to the transform's INPUT (the watermarked audio), which is
        # what rms_match() does. Keeps every arm off the loudness axis.
        assert abs(q["rms_dbfs"] - q["rms_dbfs_wm"]) < 0.2, \
            "RMS not matched on clip %d (%.3f vs %.3f dBFS) -- the arm would be a loudness point" \
            % (i, q["rms_dbfs"], q["rms_dbfs_wm"])
        rec = {"index": i, "utterance_id": ce.utterance_id(globals().get("_DS"), i, it),
               "speaker": str(it["speaker_id"]), "clean_sha": wav_sha(clean),
               "n_samples": int(len(clean)), "checkpoint_sha256": ck_sha,
               "arm": out_name, "base_arm": a.arm, "K": K, "L_ms": L,
               "prompt_in": getattr(a, "prompt_in", None),
               "base_sha": wav_sha(wm),
               "arms": {out_name: q}}
        if a.diagnostic:
            print(json.dumps({k: v for k, v in rec.items() if k != "arms"}, indent=1))
            print(json.dumps(q, indent=1))
            continue
        write_flac(os.path.join(a.out_dir, "clean", "%d.flac" % i), clean)  # pairing reference
        write_flac(os.path.join(a.out_dir, out_name, "%d.flac" % i), y)
        os.makedirs(rec_dir, exist_ok=True)
        json.dump(rec, open(os.path.join(rec_dir, "%d.json" % i), "w"), indent=1)
        recs.append(rec)
        print("[%d] %s PESQ=%.4f (vs_in %.4f) STOI=%.4f SI-SNR=%.3f acc_src=%.3f" %
              (i, out_name, q["pesq"], q["pesq_vs_wm"], q["stoi"], q["si_snr"],
               q["acc_source"]), flush=True)

    if a.diagnostic:
        print("\nDIAGNOSTIC OK -- length and RMS invariants hold. Nothing was written.")
        print("Check the PESQ above against the calibration target before generating.")
        return

    summ = {k: float(np.mean([r["arms"][out_name][k] for r in recs]))
            for k in ("pesq", "stoi", "si_snr", "snr_db", "acc_source",
                      "pesq_vs_wm", "stoi_vs_wm")}
    summ["traced_source"] = float(np.mean([r["arms"][out_name]["traced_source"] for r in recs]))
    man = {"script": "nonadditive_transforms", "arm": out_name, "base_arm": a.arm,
           "prompt_in": getattr(a, "prompt_in", None), "K": K, "L_ms": L,
           "design": "A5_curve" if forced else "matched_pesq",
           "settings": dict({k: v for k, v in vars(a).items() if k != "cmd"},
                            checkpoint_sha256=ck_sha, git=ce.git_info(),
                            packages=ce.package_versions()),
           "checkpoint_sha256": ck_sha, "git": ce.git_info(),
           "packages": ce.package_versions(), "n_clips": len(recs),
           "missing": sorted(set(range(a.n_utterances)) - {r["index"] for r in recs}),
           "complete": len(recs) == a.n_utterances, "summary": summ,
           "records": sorted(recs, key=lambda r: r["index"])}
    json.dump(man, open(os.path.join(a.out_dir, "manifest.json"), "w"), indent=1)
    print("\nDONE ->", a.out_dir)
    print(json.dumps(summ, indent=1))



# ===== N16 fix: the F-ratio must see REAL audio, not padding =====
# LibriSpeechSubset and VCTKSubset return fixed 20 s items, padding anything
# shorter. speaker_fratio never trimmed, though eval_prompts does. With the A1
# per-frame level normalisation, padding becomes amplified noise.
class _TrimmedDataset(torch.utils.data.Dataset):
    def __init__(self, inner): self.inner = inner
    def __len__(self): return len(self.inner)
    def __getitem__(self, i):
        it = dict(self.inner[i])
        w = ce.trim_padding(it["waveform"].reshape(-1))
        it["waveform"] = w.reshape(1, -1)
        return it


if __name__ == "__main__":
    main()
