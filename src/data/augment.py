"""
src/data/augment.py

VC-simulated augmentations, per the paper: "masking, shuffling, replacing, and
neural [distortion]" -- used to make the watermark detector robust to the kinds
of corruption a zero-shot VC pipeline introduces (content/duration changes,
partial watermark survival).

CONFIDENCE NOTE (consistent with voicemark_losses.py): the paper names these
four augmentation categories but the exact implementation (segment lengths,
probabilities, which codec is used for the "neural" distortion) is not in any
available source file (VoiceMark's training code isn't public). This is a
reasonable reconstruction of each named category, not a verified reproduction.
Document this explicitly in your methodology chapter.

Each augmentation function returns (augmented_waveform, frame_mask), where
frame_mask is a [n_frames] binary tensor at the CODEC's latent frame rate
(aligned via downsample_rate, matching what compute_lvad's augmentation_mask
expects) marking which frames were altered by this augmentation -- these
frames get excluded from the VAD-positive label set, since their
speaker-specific latents are no longer trustworthy watermark carriers.
"""

import random
import torch
import torch.nn.functional as F
import torchaudio


def _frame_mask_from_sample_mask(sample_mask: torch.Tensor, downsample_rate: int, n_frames: int) -> torch.Tensor:
    """
    sample_mask: [T] binary tensor at raw audio sample rate (1 = this sample
        was altered by the augmentation)
    downsample_rate: codec's total stride (product of encoder strides -- see
        SpeechTokenizer's self.downsample_rate, confirmed present in
        speechtokenizer/model.py's __init__)
    n_frames: target number of codec latent frames to align to (should match
        the detector's actual output time dimension for this input length)

    A codec frame is considered "altered" if ANY sample within its receptive
    window was altered -- conservative, since even partial corruption of a
    frame's input can materially change its latent representation.
    """
    sample_mask = sample_mask.float().unsqueeze(0).unsqueeze(0)  # [1, 1, T]
    frame_mask = F.max_pool1d(
        sample_mask, kernel_size=downsample_rate, stride=downsample_rate, ceil_mode=True
    ).squeeze()
    if frame_mask.dim() == 0:
        frame_mask = frame_mask.unsqueeze(0)
    if frame_mask.shape[0] != n_frames:
        frame_mask = F.interpolate(
            frame_mask.unsqueeze(0).unsqueeze(0), size=n_frames, mode="nearest"
        ).squeeze()
    return frame_mask


def augment_masking(waveform: torch.Tensor, downsample_rate: int, n_frames: int,
                     min_frac: float = 0.05, max_frac: float = 0.20) -> tuple:
    """Zeroes out a random contiguous segment (silences it), simulating a
    dropped/masked region."""
    T = waveform.shape[-1]
    frac = random.uniform(min_frac, max_frac)
    seg_len = int(T * frac)
    start = random.randint(0, max(1, T - seg_len))

    out = waveform.clone()
    out[..., start:start + seg_len] = 0.0

    sample_mask = torch.zeros(T)
    sample_mask[start:start + seg_len] = 1.0
    frame_mask = _frame_mask_from_sample_mask(sample_mask, downsample_rate, n_frames)
    return out, frame_mask


def augment_shuffling(waveform: torch.Tensor, downsample_rate: int, n_frames: int,
                       n_chunks: int = 8) -> tuple:
    """Splits the waveform into n_chunks contiguous pieces and randomly
    permutes them, simulating the reordering/duration distortion a VC
    pipeline's duration model can introduce."""
    T = waveform.shape[-1]
    chunk_len = T // n_chunks
    if chunk_len == 0:
        # too short to chunk meaningfully -- no-op
        return waveform.clone(), torch.zeros(n_frames)

    chunks = [waveform[..., i * chunk_len:(i + 1) * chunk_len] for i in range(n_chunks)]
    remainder = waveform[..., n_chunks * chunk_len:]

    order = list(range(n_chunks))
    random.shuffle(order)
    shuffled = torch.cat([chunks[i] for i in order], dim=-1)
    if remainder.numel() > 0:
        shuffled = torch.cat([shuffled, remainder], dim=-1)

    # Every chunk that moved position is "altered" from the detector's
    # perspective (its temporal context changed even if content didn't) --
    # mark all chunks whose order changed.
    sample_mask = torch.zeros(T)
    for new_pos, orig_pos in enumerate(order):
        if new_pos != orig_pos:
            sample_mask[new_pos * chunk_len:(new_pos + 1) * chunk_len] = 1.0
    frame_mask = _frame_mask_from_sample_mask(sample_mask, downsample_rate, n_frames)
    return shuffled, frame_mask


def augment_replacing(waveform: torch.Tensor, downsample_rate: int, n_frames: int,
                       other_waveform: torch.Tensor = None,
                       min_frac: float = 0.05, max_frac: float = 0.20) -> tuple:
    """Replaces a random contiguous segment with either noise (if no
    other_waveform given) or a segment from a different utterance (simulating
    content substitution)."""
    T = waveform.shape[-1]
    frac = random.uniform(min_frac, max_frac)
    seg_len = int(T * frac)
    start = random.randint(0, max(1, T - seg_len))

    out = waveform.clone()
    if other_waveform is not None and other_waveform.shape[-1] >= seg_len:
        other_start = random.randint(0, other_waveform.shape[-1] - seg_len)
        replacement = other_waveform[..., other_start:other_start + seg_len]
    else:
        replacement = torch.randn(*waveform.shape[:-1], seg_len) * waveform.std()

    out[..., start:start + seg_len] = replacement

    sample_mask = torch.zeros(T)
    sample_mask[start:start + seg_len] = 1.0
    frame_mask = _frame_mask_from_sample_mask(sample_mask, downsample_rate, n_frames)
    return out, frame_mask


def augment_neural(waveform: torch.Tensor, downsample_rate: int, n_frames: int,
                    sample_rate: int = 16000) -> tuple:
    """
    Simulates 'neural' distortion via lossy codec round-trip (MP3), a standard
    proxy for the kind of compression artifacts real-world audio sharing
    introduces. Requires torchaudio's ffmpeg-backed codec support.

    NOTE: this affects the ENTIRE waveform (compression is global, not
    localized to a segment), so the returned frame_mask marks all frames as
    altered -- unlike the other three augmentations, this isn't a partial
    corruption.
    """
    try:
        # torchaudio.io / sox_effects-based codec application; exact API
        # depends on torchaudio version and available backends. Falls back to
        # a lightweight quantization-noise approximation if unavailable.
        effects = [["compand", "0.02,0.05", "-60,-40,-10", "-5", "-90", "0.05"]]
        out, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, sample_rate, effects)
        if out.shape[-1] != waveform.shape[-1]:
            out = F.interpolate(out.unsqueeze(0), size=waveform.shape[-1], mode="linear",
                                 align_corners=False).squeeze(0)
    except Exception as e:
        if not getattr(augment_neural, "_warned", False):
            print(f"[augment_neural] WARNING - sox effects unavailable ({e}), "
                  f"falling back to quantization-noise approximation for the "
                  f"rest of this run (this message prints once, not per-call).")
            augment_neural._warned = True
        # Bit-depth reduction as a crude lossy-codec proxy
        out = torch.round(waveform * 255) / 255

    frame_mask = torch.ones(n_frames)
    return out, frame_mask


AUGMENTATION_FNS = {
    "masking": augment_masking,
    "shuffling": augment_shuffling,
    "replacing": augment_replacing,
    "neural": augment_neural,
}


def apply_random_augmentation(waveform: torch.Tensor, downsample_rate: int, n_frames: int,
                               other_waveform: torch.Tensor = None,
                               augmentation_names: list = None) -> tuple:
    """
    Applies ONE randomly-chosen augmentation (matching the paper's description
    of augmentations as alternatives applied during training, not all four
    stacked every step -- though stacking is a reasonable ablation to try
    later if robustness needs strengthening).

    Returns (augmented_waveform, frame_mask, augmentation_name).
    """
    if augmentation_names is None:
        augmentation_names = list(AUGMENTATION_FNS.keys())
    name = random.choice(augmentation_names)
    fn = AUGMENTATION_FNS[name]

    if name == "replacing":
        aug_wav, frame_mask = fn(waveform, downsample_rate, n_frames, other_waveform=other_waveform)
    else:
        aug_wav, frame_mask = fn(waveform, downsample_rate, n_frames)

    return aug_wav, frame_mask, name


if __name__ == "__main__":
    # Smoke test with dummy audio (downsample_rate=320 matches SpeechTokenizer's
    # product of strides for its default config -- verify against your actual
    # config if this assumption matters to you: 2*4*5*8*2 or similar SEANet
    # stride product; check st_model.downsample_rate directly for the real value)
    dummy_wav = torch.randn(1, 48000)  # 3 seconds at 16kHz
    downsample_rate = 320
    n_frames = 48000 // downsample_rate

    for name, fn in AUGMENTATION_FNS.items():
        if name == "replacing":
            aug, mask = fn(dummy_wav, downsample_rate, n_frames, other_waveform=torch.randn(1, 48000))
        else:
            aug, mask = fn(dummy_wav, downsample_rate, n_frames)
        print(f"{name}: aug shape={aug.shape}, mask shape={mask.shape}, "
              f"mask altered frac={mask.mean().item():.3f}")

    aug, mask, name = apply_random_augmentation(dummy_wav, downsample_rate, n_frames)
    print(f"Random pick: {name}")
