"""
src/losses/voicemark_losses.py

VoiceMark's five training losses, per the paper (arXiv 2505.21568):
    L = lambda_vad * Lvad + lambda_cos * Lcos + lambda_mel * Lmel
        + lambda_adv * Ladv + lambda_dec * Ldec
    weights from the paper: lambda_vad=1, lambda_cos=2, lambda_mel=2,
                             lambda_adv=1, lambda_dec=1
    optimizer: Adam, lr=5e-5

CONFIDENCE LEVELS -- read before trusting this file blindly:
  - Lcos, Ldec: HIGH confidence. Built directly against confirmed tensor shapes
    from the vendored speechtokenizer/model.py and models.py source (acoustic /
    acoustic_wm from SpeechTokenizer.forward(); chunk_logits [batch,4,16] and
    the bit->chunk-index packing from WMDetector.forward() / WMEmbedder.forward()).
  - Lmel, Ladv: MEDIUM confidence. Structure follows the paper's description and
    standard EnCodec-family convention (which SpeechTokenizer itself descends
    from), but the exact per-scale FFT sizes / feature-matching weighting were
    not found in any available source file -- these are reasonable, defensible
    choices, not a verified reproduction.
  - Lvad: LOWEST confidence. The paper cites Rabiner (1978) dual-threshold VAD
    by name, but no VAD implementation exists in any file we have access to
    (VoiceMark's own training code, which would contain it, is not public --
    see Week 1 audit). The energy+ZCR dual-threshold VAD below is a reasonable
    reconstruction of the cited classical algorithm, NOT a reproduction of
    VoiceMark's own code. Treat this as your own implementation choice to
    document explicitly in the thesis methodology section, not as "the same
    VAD VoiceMark used."
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


# ---------------------------------------------------------------------------
# Ldec: watermark decoding loss
# ---------------------------------------------------------------------------

def bits_to_chunk_indices(message: torch.Tensor, nchunk_size: int = 4) -> torch.Tensor:
    """
    Converts a [batch, nbits] binary tensor into [batch, nchunks] integer chunk
    indices, using the EXACT same bit-packing WMEmbedder.forward() uses
    internally (confirmed in models.py):
        chunk_val = sum(chunk_bits[:, bit_idx] << bit_idx for bit_idx in range(nchunk_size))
    This must match exactly, since it's what the embedder actually encoded --
    if this function used a different bit order, Ldec would be training against
    the wrong target.
    """
    batch_size, nbits = message.shape
    assert nbits % nchunk_size == 0, "nbits must be a multiple of nchunk_size"
    nchunks = nbits // nchunk_size

    chunk_indices = []
    for i in range(nchunks):
        chunk_bits = message[:, i * nchunk_size:(i + 1) * nchunk_size]  # [batch, nchunk_size]
        chunk_val = torch.zeros(batch_size, dtype=torch.long, device=message.device)
        for bit_idx in range(nchunk_size):
            chunk_val = chunk_val + (chunk_bits[:, bit_idx].long() << bit_idx)
        chunk_indices.append(chunk_val.unsqueeze(1))
    return torch.cat(chunk_indices, dim=1)  # [batch, nchunks]


def compute_ldec(chunk_logits: torch.Tensor, message: torch.Tensor, nchunk_size: int = 4) -> torch.Tensor:
    """
    chunk_logits: [batch, nchunks, 2**nchunk_size] from WMDetector.forward()
    message: [batch, nbits] ground-truth binary watermark
    Returns scalar cross-entropy loss averaged over chunks and batch.
    """
    target = bits_to_chunk_indices(message, nchunk_size=nchunk_size)  # [batch, nchunks]
    # cross_entropy expects [N, C] vs [N] -- flatten batch and chunk dims together
    batch, nchunks, n_classes = chunk_logits.shape
    return F.cross_entropy(
        chunk_logits.reshape(batch * nchunks, n_classes),
        target.reshape(batch * nchunks),
    )


# ---------------------------------------------------------------------------
# Lcos: speaker-latent cosine similarity loss
# ---------------------------------------------------------------------------

def compute_lcos(acoustic: torch.Tensor, acoustic_wm: torch.Tensor) -> torch.Tensor:
    """
    acoustic, acoustic_wm: [batch, dim, time] -- pre/post watermark
    speaker-specific latents, returned directly by
    VoiceMarkBackbone.forward_full() (which calls st_model.forward() directly
    to expose these, since SBW.forward() discards them).

    Lcos = 1 - mean(cosine_similarity along the channel dim)
    """
    cos_sim = F.cosine_similarity(acoustic, acoustic_wm, dim=1)  # [batch, time]
    return 1.0 - cos_sim.mean()


# ---------------------------------------------------------------------------
# Lmel: multi-scale mel-spectrogram loss (EnCodec-style)
# ---------------------------------------------------------------------------

class MultiScaleMelLoss(nn.Module):
    """
    Standard multi-scale mel loss, following EnCodec/SpeechTokenizer-family
    convention: L1 on log-mel + L2 on linear mel, summed across several FFT
    scales, since larger windows capture pitch/timbre and smaller windows
    capture transients.

    NOTE (see module docstring): the exact scale list VoiceMark itself used is
    not confirmed from any available source -- this uses EnCodec's commonly
    published defaults (n_fft in powers of 2 from 32 to 2048) as a defensible
    standard choice.
    """

    def __init__(self, sample_rate: int = 16000, n_mels: int = 64,
                 fft_sizes=(32, 64, 128, 256, 512, 1024, 2048)):
        super().__init__()
        self.sample_rate = sample_rate
        self.fft_sizes = fft_sizes
        # Small FFT sizes have too few frequency bins to support n_mels=64
        # filters without several filters going all-zero (wasted, and a less
        # meaningful loss at that scale) -- cap n_mels per scale so every
        # filter has at least ~2 frequency bins to work with.
        self.mel_transforms = nn.ModuleList([
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=n_fft // 4,
                n_mels=min(n_mels, max(1, (n_fft // 2 + 1) // 2)),
                power=1.0,  # magnitude, not power -- matches common EnCodec-style mel loss
            )
            for n_fft in fft_sizes
        ])

    def forward(self, x_wm: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        x_wm, x: [batch, 1, T] watermarked reconstruction and original clean
        waveform respectively.
        """
        total = 0.0
        for mel in self.mel_transforms:
            mel_wm = mel(x_wm.squeeze(1))  # [batch, n_mels, frames]
            mel_x = mel(x.squeeze(1))
            l1 = F.l1_loss(mel_wm, mel_x)
            l2 = F.mse_loss(
                torch.log(mel_wm.clamp(min=1e-5)),
                torch.log(mel_x.clamp(min=1e-5)),
            )
            total = total + l1 + l2
        return total / len(self.mel_transforms)


# ---------------------------------------------------------------------------
# Ladv: adversarial loss via the pretrained msstftd discriminator
# ---------------------------------------------------------------------------

def compute_ladv_generator(discriminator, x_wm: torch.Tensor, x: torch.Tensor,
                            feature_matching_weight: float = 1.0):
    """
    Generator-side adversarial loss: hinge loss (standard for EnCodec-family
    discriminators) + feature-matching L1 loss between real/fake intermediate
    discriminator features.

    discriminator: VoiceMarkDiscriminator instance (frozen or fine-tuned --
    your choice; VoiceMark's checkpoint includes optimizer state for it,
    suggesting it may have been fine-tuned jointly, see build_optimizer() in
    backbone.py if you want to continue training it).

    Confirmed contract (speechtokenizer.discriminators.MultiScaleSTFTDiscriminator):
        forward(y, y_hat) -> (logits, logits_fake, fmaps, fmaps_fake)
    ONE call computes both real and fake in a single pass -- do not call this
    twice separately.

    Returns (adv_loss, feature_matching_loss) as separate scalars so you can
    weight/log them independently if needed.

    NOTE (see module docstring): whether VoiceMark's own Ladv term includes
    feature matching, and at what relative weight, is not confirmed from any
    available source. This follows standard EnCodec-family practice.
    """
    logits_real, logits_fake, fmaps_real, fmaps_fake = discriminator(x, x_wm)

    # Generator hinge loss: wants D(fake) to be large (classified as real).
    # NOTE: this is NOT the same formula as the discriminator's real-data term
    # (relu(1 - logit)) -- that was a bug in an earlier version of this file.
    # Correct hinge generator loss is simply -mean(logit_fake), no relu.
    adv_loss = 0.0
    for lf in logits_fake:
        adv_loss = adv_loss + (-lf.mean())
    adv_loss = adv_loss / len(logits_fake)

    fm_loss = 0.0
    n_layers = 0
    for f_feats, r_feats in zip(fmaps_fake, fmaps_real):
        for ff, rf in zip(f_feats, r_feats):
            fm_loss = fm_loss + F.l1_loss(ff, rf.detach())
            n_layers += 1
    fm_loss = fm_loss / max(n_layers, 1)

    return adv_loss, feature_matching_weight * fm_loss


def compute_ladv_discriminator(discriminator, x_wm: torch.Tensor, x: torch.Tensor):
    """
    Discriminator-side hinge loss, for updating the discriminator itself
    (separate optimizer step from the generator/adapter update -- use
    VoiceMarkDiscriminator.build_optimizer() for this).

    x_wm should be detached by the caller before this is invoked (or pass
    x_wm.detach() directly) so no gradient flows into the generator during the
    discriminator's own update step.
    """
    logits_real, logits_fake, _, _ = discriminator(x, x_wm.detach())

    d_loss = 0.0
    for rl, fl in zip(logits_real, logits_fake):
        d_loss = d_loss + F.relu(1.0 - rl).mean() + F.relu(1.0 + fl).mean()
    return d_loss / len(logits_real)


# ---------------------------------------------------------------------------
# Lvad: VAD-weighted detection loss
# ---------------------------------------------------------------------------

def rabiner_dual_threshold_vad(
    waveform: torch.Tensor,
    sample_rate: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    energy_high_pctile: float = 0.75,
    energy_low_pctile: float = 0.35,
) -> torch.Tensor:
    """
    Classical Rabiner (1978)-style dual-threshold VAD using short-time energy
    and zero-crossing rate. See module docstring: this is a reconstruction of
    the cited classical algorithm, not a verified reproduction of VoiceMark's
    own (unavailable) VAD code.

    waveform: [batch, 1, T] or [batch, T]
    Returns: [batch, n_frames] binary tensor (1 = speech, 0 = silence/unvoiced)
    at frame_ms/hop_ms resolution -- caller must downsample/align this to the
    codec's latent frame rate (see align_vad_to_latent_frames below).
    """
    if waveform.dim() == 3:
        waveform = waveform.squeeze(1)  # [batch, T]
    batch, T = waveform.shape

    frame_len = int(sample_rate * frame_ms / 1000)
    hop_len = int(sample_rate * hop_ms / 1000)

    frames = waveform.unfold(-1, frame_len, hop_len)  # [batch, n_frames, frame_len]

    # Short-time energy
    energy = (frames ** 2).mean(dim=-1)  # [batch, n_frames]

    # Zero-crossing rate
    signs = torch.sign(frames)
    zcr = (signs[..., 1:] != signs[..., :-1]).float().mean(dim=-1)  # [batch, n_frames]

    vad_labels = torch.zeros_like(energy)
    for b in range(batch):
        e = energy[b]
        e_sorted = torch.sort(e).values
        high_thresh = e_sorted[int(energy_high_pctile * (len(e_sorted) - 1))]
        low_thresh = e_sorted[int(energy_low_pctile * (len(e_sorted) - 1))]

        # Dual threshold: frames above high_thresh are speech; frames between
        # low and high extend a speech region if adjacent to an already-speech
        # frame (classic Rabiner hangover logic), else silence.
        above_high = e >= high_thresh
        above_low = e >= low_thresh
        labels = above_high.clone()
        # Simple forward+backward pass to extend regions through the low band
        for t in range(1, len(labels)):
            if above_low[t] and labels[t - 1]:
                labels[t] = True
        for t in range(len(labels) - 2, -1, -1):
            if above_low[t] and labels[t + 1]:
                labels[t] = True
        vad_labels[b] = labels.float()

    return vad_labels  # [batch, n_frames]


def align_vad_to_latent_frames(vad_labels: torch.Tensor, target_n_frames: int) -> torch.Tensor:
    """
    Interpolates VAD labels (computed at frame_ms/hop_ms resolution) to match
    the codec's actual latent frame count (target_n_frames = detector logits'
    time dimension). Uses nearest-neighbor interpolation since these are
    binary labels.
    """
    vad_labels = vad_labels.unsqueeze(1)  # [batch, 1, n_frames]
    aligned = F.interpolate(vad_labels, size=target_n_frames, mode="nearest")
    return aligned.squeeze(1)  # [batch, target_n_frames]


def compute_lvad(
    detection_logits: torch.Tensor,
    waveform: torch.Tensor,
    sample_rate: int,
    augmentation_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    detection_logits: [batch, time_steps] raw (pre-sigmoid) logits from
        WMDetector.forward()
    waveform: [batch, 1, T] the ORIGINAL clean audio (VAD is computed on clean
        speech content, not the watermarked/distorted version)
    augmentation_mask: optional [batch, time_steps] binary tensor, 1 where a
        frame was masked/shuffled/replaced by VC-simulated augmentation (these
        frames get forced to label 0 regardless of VAD, per the paper: "silence
        and voiceless frames... lack such [speaker-specific] latents" and
        should not be counted as valid watermark-bearing frames). If you
        haven't implemented augmentation yet (Week 5+ per the project plan),
        pass None and this term is skipped.

    Returns scalar BCEWithLogits loss.
    """
    vad_raw = rabiner_dual_threshold_vad(waveform, sample_rate)
    vad_aligned = align_vad_to_latent_frames(vad_raw, detection_logits.shape[1])

    if augmentation_mask is not None:
        vad_aligned = vad_aligned * (1.0 - augmentation_mask)

    return F.binary_cross_entropy_with_logits(detection_logits, vad_aligned)


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

class VoiceMarkLosses(nn.Module):
    """
    Combines all five losses with the paper's stated weights. Call
    `.forward(...)` once per training step with everything needed; returns a
    dict of individual losses plus 'total' (the weighted sum used for
    backward()).

    Usage:
        losses_fn = VoiceMarkLosses(sample_rate=16000)
        out = backbone.forward_full(clean_audio, message)
        detect_feat = backbone.model.st_model.forward_feature(out['recon_wm'])
        logits, chunk_logits = backbone.model.detector(detect_feat)

        loss_dict = losses_fn(
            recon_wm=out['recon_wm'], clean_audio=clean_audio,
            acoustic=out['acoustic'], acoustic_wm=out['acoustic_wm'],
            detection_logits=logits, chunk_logits=chunk_logits,
            message=message, discriminator=discriminator,
        )
        loss_dict['total'].backward()
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        lambda_vad: float = 1.0,
        lambda_cos: float = 2.0,
        lambda_mel: float = 2.0,
        lambda_adv: float = 1.0,
        lambda_dec: float = 1.0,
        nchunk_size: int = 4,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.lambdas = dict(vad=lambda_vad, cos=lambda_cos, mel=lambda_mel, adv=lambda_adv, dec=lambda_dec)
        self.nchunk_size = nchunk_size
        self.mel_loss = MultiScaleMelLoss(sample_rate=sample_rate)

    def forward(
        self,
        recon_wm: torch.Tensor,
        clean_audio: torch.Tensor,
        acoustic: torch.Tensor,
        acoustic_wm: torch.Tensor,
        detection_logits: torch.Tensor,
        chunk_logits: torch.Tensor,
        message: torch.Tensor,
        discriminator=None,
        augmentation_mask: torch.Tensor = None,
    ) -> dict:
        l_cos = compute_lcos(acoustic, acoustic_wm)
        l_mel = self.mel_loss(recon_wm, clean_audio)
        l_dec = compute_ldec(chunk_logits, message, nchunk_size=self.nchunk_size)
        l_vad = compute_lvad(detection_logits, clean_audio, self.sample_rate, augmentation_mask)

        if discriminator is not None:
            l_adv, l_fm = compute_ladv_generator(discriminator, recon_wm, clean_audio)
        else:
            l_adv = torch.tensor(0.0, device=recon_wm.device)
            l_fm = torch.tensor(0.0, device=recon_wm.device)

        total = (
            self.lambdas["vad"] * l_vad
            + self.lambdas["cos"] * l_cos
            + self.lambdas["mel"] * l_mel
            + self.lambdas["adv"] * (l_adv + l_fm)
            + self.lambdas["dec"] * l_dec
        )

        return {
            "total": total,
            "lvad": l_vad.detach(),
            "lcos": l_cos.detach(),
            "lmel": l_mel.detach(),
            "ladv": l_adv.detach(),
            "lfm": l_fm.detach(),
            "ldec": l_dec.detach(),
        }