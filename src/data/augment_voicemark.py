"""
src/data/augment_voicemark.py

VC-simulated augmentation following the VoiceMark paper (Li et al., Interspeech
2025, arXiv 2505.21568, Section 2.3). This replaces src/data/augment.py for
training. augment.py is kept unchanged so older checkpoints stay reproducible.

What the paper states (verbatim, Section 2.3):
  1. "We mask certain frames to zero with a probability of 20%."
  2. "We randomly shuffle the audio in 50 ms windows with a 50% probability."
  3. "We replace 50 ms segments with the original audio at a probability of 50%."
  4. "We encode and decode the audio using EnCodec."
  5. "We apply speed perturbation, amplitude scaling, filtering, or resampling
     with 10% probability."
  VAD labels (Section 2.5): "Frames containing both speech and a watermark are
  labeled as 1, while silent, masked or replaced frames are labeled as 0."

What the paper does NOT state, and how this file decides it (document these in
the methodology chapter as our interpretation):
  - The window for masking: 50 ms, the same window the paper gives for
    shuffling and replacing.
  - The probabilities are read per 50 ms window for masking (20%), shuffling
    (50% of windows take part in one random permutation among themselves) and
    replacing (50%).
  - Whether one augmentation is used per example or several are stacked:
    configurable. Default "one": each example gets exactly one augmentation,
    chosen uniformly, or none with probability p_identity. "stack" applies
    them in sequence, each switched on per example with probability p_stack
    (the perturbation step keeps its stated 10%).
  - EnCodec settings: the 24 kHz model, bandwidth drawn from
    {1.5, 3, 6, 12} kbps per call.

Differences from the old augment.py, which are the reason this file exists:
  - The augmented audio is RETURNED AND USED as detector input (the old
    src/train.py discarded it and only changed VAD labels).
  - Replacing uses the ORIGINAL unwatermarked audio (partial watermark
    survival), not noise or another utterance.
  - The neural step is a real EnCodec round-trip. There is no silent fallback:
    if EnCodec cannot be loaded the augmenter raises, because the old sox
    fallback silently turned the "neural" step into 8-bit rounding.

Gradients: masking, replacing, shuffling, amplitude, filtering, resampling and
speed are all differentiable with respect to the watermarked input. EnCodec's
quantiser is not, so it uses a straight-through estimator:
    out = x + (encodec(x) - x).detach()
so gradients still reach the embedder through that branch.

Each call returns a dict:
  audio      [B, 1, T]  augmented watermarked audio (feed this to the detector)
  vad_ref    [B, 1, T]  the CLEAN audio after the same structural edits
                        (masking, shuffling, speed), so a VAD computed on it is
                        aligned with `audio`; replaced segments still contain
                        speech here and are excluded through `nowm_mask`
  nowm_mask  [B, T]     1.0 where the sample carries no watermark
                        (masked or replaced), else 0.0
  names      list[str]  which augmentation(s) each example received

Use samples_to_frames(nowm_mask, n_frames) to get the per-frame mask that
compute_lvad() expects as augmentation_mask.
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF


ALL_AUGMENTATIONS = ("mask", "shuffle", "replace", "encodec", "perturb")


def samples_to_frames(sample_mask: torch.Tensor, n_frames: int) -> torch.Tensor:
    """[B, T] sample-level mask -> [B, n_frames] frame-level mask. A frame is
    marked if ANY of its samples is marked (conservative: a partially replaced
    frame is not a clean watermark carrier)."""
    return F.adaptive_max_pool1d(sample_mask.float().unsqueeze(1), n_frames).squeeze(1)


class VoiceMarkAugmenter(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        mode: str = "one",
        augmentations=ALL_AUGMENTATIONS,
        p_identity: float = 0.2,
        p_stack: float = 0.5,
        window_ms: float = 50.0,
        p_mask: float = 0.2,
        p_shuffle: float = 0.5,
        p_replace: float = 0.5,
        p_perturb: float = 0.1,
        encodec_bandwidths=(1.5, 3.0, 6.0, 12.0),
        speed_range=(0.9, 1.1),
        gain_range=(0.5, 1.5),
        resample_rates=(8000, 11025, 12000),
    ):
        super().__init__()
        if mode not in ("one", "stack"):
            raise ValueError(f"mode must be 'one' or 'stack', got {mode!r}")
        unknown = set(augmentations) - set(ALL_AUGMENTATIONS)
        if unknown:
            raise ValueError(f"unknown augmentations {unknown}; choose from {ALL_AUGMENTATIONS}")
        self.sr = sample_rate
        self.mode = mode
        self.augmentations = tuple(augmentations)
        self.p_identity = p_identity
        self.p_stack = p_stack
        self.win = int(round(sample_rate * window_ms / 1000.0))
        self.p_mask, self.p_shuffle, self.p_replace = p_mask, p_shuffle, p_replace
        self.p_perturb = p_perturb
        self.encodec_bandwidths = tuple(encodec_bandwidths)
        self.speed_range, self.gain_range = speed_range, gain_range
        self.resample_rates = tuple(resample_rates)
        if "encodec" in self.augmentations:
            self._load_encodec()

    # ------------------------------------------------------------------ setup
    def _load_encodec(self):
        try:
            from encodec import EncodecModel
        except ImportError as e:
            raise ImportError(
                "EnCodec is required for the 'encodec' augmentation (pip install encodec). "
                "Refusing to fall back silently; drop 'encodec' from --aug_list explicitly "
                "if you really want to train without it."
            ) from e
        model = EncodecModel.encodec_model_24khz()
        model.eval()
        for prm in model.parameters():
            prm.requires_grad_(False)
        # kept out of nn.Module registration so it is never saved in a checkpoint
        # or picked up by .parameters(); moved to the input's device on use
        object.__setattr__(self, "_encodec_model", model)
        self._encodec_sr = model.sample_rate

    # --------------------------------------------------------- window helpers
    def _window_select(self, T, p, device):
        """[T] float mask: each 50 ms window independently selected with prob p."""
        n_win = math.ceil(T / self.win)
        sel = (torch.rand(n_win, device=device) < p).float()
        return sel.repeat_interleave(self.win)[:T]

    # ------------------------------------------- augmentations (one example)
    # Every op takes and returns (x [1, T], ref [1, T], nowm [T]) and never
    # writes in place, so autograd can save any intermediate it needs.
    def _mask(self, x, ref, nowm, clean):
        m = self._window_select(x.shape[-1], self.p_mask, x.device)
        return x * (1.0 - m), ref * (1.0 - m), torch.maximum(nowm, m)

    def _replace(self, x, ref, nowm, clean):
        m = self._window_select(x.shape[-1], self.p_replace, x.device)
        return x * (1.0 - m) + clean * m, ref, torch.maximum(nowm, m)

    def _shuffle(self, x, ref, nowm, clean):
        T = x.shape[-1]
        n_win = math.ceil(T / self.win)
        take = [w for w in range(n_win) if random.random() < self.p_shuffle]
        if len(take) < 2:
            return x, ref, nowm
        shuffled = take[:]
        random.shuffle(shuffled)
        perm = list(range(n_win))
        for src, dst in zip(take, shuffled):
            perm[dst] = src
        order = torch.tensor(perm, device=x.device)
        pad = n_win * self.win - T

        def apply(t2):  # [C, T]
            t2 = F.pad(t2, (0, pad))
            return t2.reshape(t2.shape[0], n_win, self.win)[:, order].reshape(t2.shape[0], -1)[:, :T]

        return apply(x), apply(ref), apply(nowm.unsqueeze(0))[0]

    def _encodec(self, x, ref, nowm, clean):
        model = self._encodec_model.to(x.device)
        T = x.shape[-1]
        with torch.no_grad():
            model.set_target_bandwidth(random.choice(self.encodec_bandwidths))
            up = AF.resample(x.detach(), self.sr, self._encodec_sr)
            dec = model(up.unsqueeze(0))[0]
            down = AF.resample(dec, self._encodec_sr, self.sr)
            down = F.pad(down, (0, max(0, T - down.shape[-1])))[..., :T]
        return x + (down - x).detach(), ref, nowm  # straight-through

    def _perturb(self, x, ref, nowm, clean):
        T = x.shape[-1]
        kind = random.choice(("speed", "gain", "filter", "resample"))
        if kind == "gain":
            return x * random.uniform(*self.gain_range), ref, nowm
        if kind == "filter":
            if random.random() < 0.5:
                y = AF.lowpass_biquad(x, self.sr, cutoff_freq=random.uniform(3000.0, 7000.0))
            else:
                y = AF.highpass_biquad(x, self.sr, cutoff_freq=random.uniform(50.0, 500.0))
            return y, ref, nowm
        if kind == "resample":
            mid = random.choice(self.resample_rates)
            y = AF.resample(AF.resample(x, self.sr, mid), mid, self.sr)
            return F.pad(y, (0, max(0, T - y.shape[-1])))[..., :T], ref, nowm
        # speed: stretch time (and pitch) by a factor, crop/pad back to T.
        # Padding is silence in ref as well, so the VAD labels it 0.
        factor = random.uniform(*self.speed_range)
        new_len = max(2, int(round(T / factor)))

        def stretch(t2, mode):  # [C, T]
            kw = {"align_corners": False} if mode == "linear" else {}
            y = F.interpolate(t2.unsqueeze(0), size=new_len, mode=mode, **kw)[0]
            return F.pad(y, (0, max(0, T - new_len)))[..., :T]

        return stretch(x, "linear"), stretch(ref, "linear"), stretch(nowm.unsqueeze(0), "nearest")[0]

    # ---------------------------------------------------------------- forward
    # fixed order: content edits first, then codec, then signal perturbation
    _ORDER = ("replace", "mask", "shuffle", "encodec", "perturb")

    def forward(self, x_wm: torch.Tensor, x_clean: torch.Tensor) -> dict:
        """x_wm, x_clean: [B, 1, T] watermarked and clean audio at self.sr."""
        if x_wm.shape != x_clean.shape:
            raise ValueError(f"shape mismatch {tuple(x_wm.shape)} vs {tuple(x_clean.shape)}")
        B, _, T = x_wm.shape
        clean_all = x_clean.detach()
        ops = {"mask": self._mask, "shuffle": self._shuffle, "replace": self._replace,
               "encodec": self._encodec, "perturb": self._perturb}
        xs, refs, masks, names = [], [], [], []
        for b in range(B):
            if self.mode == "one":
                chosen = [] if random.random() < self.p_identity else [random.choice(self.augmentations)]
            else:
                chosen = [a for a in self.augmentations
                          if random.random() < (self.p_perturb if a == "perturb" else self.p_stack)]
            x, ref, clean = x_wm[b], clean_all[b], clean_all[b]
            nowm = torch.zeros(T, device=x_wm.device)
            for a in self._ORDER:
                if a in chosen:
                    x, ref, nowm = ops[a](x, ref, nowm, clean)
            xs.append(x)
            refs.append(ref)
            masks.append(nowm)
            names.append("+".join(a for a in self._ORDER if a in chosen) or "identity")
        return {
            "audio": torch.stack(xs),
            "vad_ref": torch.stack(refs),
            "nowm_mask": torch.stack(masks),
            "names": names,
        }


def add_augmentation_args(p):
    """Shared argparse options so train.py and the Route 2 script stay in sync."""
    p.add_argument("--aug_mode", choices=("one", "stack"), default="one",
                   help="'one': one augmentation per example (or none with --aug_p_identity). "
                        "'stack': each switched on per example with --aug_p_stack. "
                        "The paper does not say which; see augment_voicemark.py.")
    p.add_argument("--aug_list", type=str, default=",".join(ALL_AUGMENTATIONS),
                   help=f"Comma-separated subset of {ALL_AUGMENTATIONS}.")
    p.add_argument("--aug_p_identity", type=float, default=0.2)
    p.add_argument("--aug_p_stack", type=float, default=0.5)
    return p


def build_augmenter(args, sample_rate=16000) -> VoiceMarkAugmenter:
    return VoiceMarkAugmenter(
        sample_rate=sample_rate,
        mode=args.aug_mode,
        augmentations=[a.strip() for a in args.aug_list.split(",") if a.strip()],
        p_identity=args.aug_p_identity,
        p_stack=args.aug_p_stack,
    )


# ---------------------------------------------------------------------------
# Self-test: python src/data/augment_voicemark.py   (CPU is fine, ~1 minute)
# ---------------------------------------------------------------------------
def _selftest():
    torch.manual_seed(0)
    random.seed(0)
    sr, T, B = 16000, 48000, 4
    t = torch.arange(T) / sr
    speechlike = (torch.sin(2 * math.pi * 220 * t) * (torch.sin(2 * math.pi * 1.5 * t) > 0)).float()
    clean = speechlike.repeat(B, 1, 1) * 0.3
    wm = clean + 0.01 * torch.randn_like(clean)
    wm.requires_grad_(True)

    fails = []
    for name in ALL_AUGMENTATIONS:
        aug = VoiceMarkAugmenter(sr, mode="one", augmentations=[name], p_identity=0.0)
        out = aug(wm, clean)
        a, ref, nowm = out["audio"], out["vad_ref"], out["nowm_mask"]
        ok_shape = a.shape == wm.shape and ref.shape == wm.shape and nowm.shape == (B, T)
        changed = (a.detach() - wm.detach()).abs().mean().item()
        a.sum().backward()
        g = wm.grad.abs().sum().item()
        wm.grad = None
        frac = nowm.mean().item()
        line = (f"  {name:8s} shape_ok={ok_shape} mean|change|={changed:.4f} "
                f"grad_to_input={g:.1f} no-watermark fraction={frac:.2f} names={out['names']}")
        print(line)
        if not ok_shape or changed == 0.0 or g == 0.0:
            fails.append(name)
        if name == "replace":
            sel = nowm > 0.5
            if sel.any() and not torch.allclose(a.detach()[:, 0][sel], clean[:, 0][sel]):
                fails.append("replace-not-original")
            if not (0.3 < frac < 0.7):
                fails.append("replace-rate")
        if name == "mask":
            sel = nowm > 0.5
            if sel.any() and a.detach()[:, 0][sel].abs().max() > 0:
                fails.append("mask-not-zero")
            if not (0.1 < frac < 0.3):
                fails.append("mask-rate")
        if name == "encodec" and frac != 0.0:
            fails.append("encodec-should-keep-watermark-labels")

    fm = samples_to_frames(torch.tensor([[0.0] * 640 + [1.0] + [0.0] * 639]), 4)
    if fm.tolist() != [[0.0, 0.0, 1.0, 0.0]]:
        fails.append(f"samples_to_frames {fm.tolist()}")

    print("SELFTEST", "PASSED" if not fails else f"FAILED: {fails}")
    return not fails


if __name__ == "__main__":
    raise SystemExit(0 if _selftest() else 1)
