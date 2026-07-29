"""
src/losses/safespeech_losses.py

SafeSpeech's SPEC (Speech PErturbative Concealment) loss, adapted from their
actual source (external/safespeech/protect.py) for our use case.

CONFIDENCE: the underlying formulas (pivotal mel loss, KL-to-noise,
L1-to-noise) are copied directly from SafeSpeech's own compute_reconstruction_loss
and compute_kl_divergence functions -- HIGH confidence on the math itself.

STRUCTURAL ADAPTATION (read before trusting this blindly):
SafeSpeech's protect.py poisons TRAINING DATA before someone fine-tunes a TTS
model on it, via epsilon-bounded PGD (sign-gradient) perturbation of the
waveform, jointly with the surrogate's own training loop. Their loss_mel term
is MINIMIZED to pull the surrogate's reconstruction toward the (bounded)
perturbed input, as part of that joint adversarial-training dynamic.

Our setting is different: we are not poisoning training data or attacking the
surrogate itself. We train the WATERMARK EMBEDDER'S weights (via standard
Adam, same as Stage 1 -- not epsilon-bounded PGD) so that embedding the
watermark ALSO disrupts zero-shot cloning at inference time. This requires two
changes from their code, both implemented below:

  1. Pivotal mel loss target: instead of L1(mel(perturbed_input),
     mel(surrogate_reconstruction)) minimized during their joint training, we
     use L1(mel(original_clean_audio), mel(cloned_output)) and NEGATE it in
     the total loss -- we want to MAXIMIZE dissimilarity between the original
     speaker's voice and the surrogate's zero-shot clone of the watermarked
     audio, i.e. disrupt voice similarity directly. Since our total loss is
     MINIMIZED via standard gradient descent, disruption enters as a negative
     term.
  2. KL-to-noise and L1-to-noise (their loss_kl, loss_nr) are used UNCHANGED
     (not negated) -- their goal (push the surrogate's output toward
     resembling random Gaussian noise) is identical to what we want here, no
     sign flip needed.

Default weight_beta=10 is SafeSpeech's own tuned value for their setup
(BERT-VITS2 surrogate, PGD perturbation optimization). Given our surrogate
(YourTTS) and training paradigm (LoRA adapter weights via Adam, not raw
waveform PGD) are both different, this should be treated as a REASONABLE
STARTING POINT for your Stage 2 hyperparameter search, not a value to trust
without checking -- see the project plan's Section 3 grid (lambda/beta
scheduling).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class SafeSpeechMelSpectrogram(nn.Module):
    """
    Single-scale mel-spectrogram, matching SafeSpeech's own mel_spectrogram_torch
    usage (hps.data.filter_length/n_mel_channels/sampling_rate/hop_length/
    win_length/mel_fmin/mel_fmax), reimplemented via torchaudio rather than
    importing their mel_processing.py directly, to avoid pulling in that
    module's own dependency chain. Defaults below match common VITS-family
    configs (BERT-VITS2 typically runs at 44.1kHz -- ADJUST sampling_rate to
    match your surrogate's actual output rate, e.g. YourTTS's, before trusting
    this numerically).
    """

    def __init__(self, sampling_rate: int = 22050, filter_length: int = 1024,
                 n_mel_channels: int = 80, hop_length: int = 256,
                 win_length: int = 1024, mel_fmin: float = 0.0, mel_fmax: float = None):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sampling_rate, n_fft=filter_length, win_length=win_length,
            hop_length=hop_length, f_min=mel_fmin, f_max=mel_fmax,
            n_mels=n_mel_channels, power=1.0,
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: [batch, 1, T] or [batch, T]. Returns [batch, n_mels, frames]."""
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        return self.mel(wav)


def compute_pivotal_disruption_loss(mel_fn: SafeSpeechMelSpectrogram,
                                     clean_audio: torch.Tensor,
                                     cloned_output: torch.Tensor) -> torch.Tensor:
    """
    ADAPTED pivotal loss (see module docstring): L1 mel distance between the
    ORIGINAL clean audio and the surrogate's CLONED output. This is what we
    want to MAXIMIZE (returned as a positive distance here; the caller
    negates it when composing the total loss, since we minimize total loss
    via standard gradient descent).
    """
    clean_mel = mel_fn(clean_audio)
    cloned_mel = mel_fn(cloned_output)
    # Cloned output may have a different length than clean_audio (VITS
    # generates its own duration) -- crop to the shorter of the two.
    min_len = min(clean_mel.shape[-1], cloned_mel.shape[-1])
    return F.l1_loss(clean_mel[..., :min_len], cloned_mel[..., :min_len])


def compute_kl_to_noise(mel_fn: SafeSpeechMelSpectrogram,
                         cloned_output: torch.Tensor,
                         random_noise: torch.Tensor) -> torch.Tensor:
    """
    Copied directly from SafeSpeech's compute_kl_divergence: mel-spectrograms
    of the surrogate's output and of random Gaussian noise, softmax'd along
    the TIME dimension (per their exact code -- F.log_softmax/softmax with
    dim=-1, treating each mel channel's temporal profile as a distribution),
    then batch-mean KL divergence. Used UNCHANGED, no sign flip -- pushing
    the cloned output's mel statistics toward the noise's is exactly what we
    want too.
    """
    x_mel = mel_fn(cloned_output)
    z_mel = mel_fn(random_noise)
    min_len = min(x_mel.shape[-1], z_mel.shape[-1])
    x_mel = x_mel[..., :min_len]
    z_mel = z_mel[..., :min_len]

    p_log = F.log_softmax(x_mel, dim=-1)
    q = F.softmax(z_mel, dim=-1)
    return F.kl_div(p_log, q, reduction="batchmean")


def compute_l1_to_noise(mel_fn: SafeSpeechMelSpectrogram,
                         cloned_output: torch.Tensor,
                         random_noise: torch.Tensor) -> torch.Tensor:
    """
    SafeSpeech's loss_nr: L1 mel loss between the surrogate's output and
    random Gaussian noise (reuses their compute_reconstruction_loss with
    wav_hat=cloned_output, wav=random_noise). Used UNCHANGED.
    """
    cloned_mel = mel_fn(cloned_output)
    noise_mel = mel_fn(random_noise)
    min_len = min(cloned_mel.shape[-1], noise_mel.shape[-1])
    return F.l1_loss(cloned_mel[..., :min_len], noise_mel[..., :min_len])


class SafeSpeechDisruptionLoss(nn.Module):
    """
    Combines the three SPEC components into a single disruption loss for
    Stage 2 training. Call once per training step with the clean input audio
    and the surrogate's cloned output (from watermarked audio).

    total = -lambda_mel * pivotal_disruption_loss + weight_beta * (l1_to_noise + kl_to_noise)

    weight_beta=10 is SafeSpeech's own default (see module docstring --
    treat as a starting point, not a verified-correct value for your setup).
    lambda_mel defaults to 1.0; the project plan's overall Stage 2 loss
    already introduces its own outer lambda multiplying this whole term
    (see THESIS_PLAN.md Section 3: L = VoiceMark losses + lambda * SafeSpeech losses),
    so avoid double-counting if you also scale this externally.
    """

    def __init__(self, sampling_rate: int = 22050, filter_length: int = 1024,
                 n_mel_channels: int = 80, hop_length: int = 256,
                 win_length: int = 1024, mel_fmin: float = 0.0, mel_fmax: float = None,
                 lambda_mel: float = 1.0, weight_beta: float = 10.0, seed: int = None):
        super().__init__()
        self.mel_fn = SafeSpeechMelSpectrogram(
            sampling_rate=sampling_rate, filter_length=filter_length,
            n_mel_channels=n_mel_channels, hop_length=hop_length,
            win_length=win_length, mel_fmin=mel_fmin, mel_fmax=mel_fmax,
        )
        self.lambda_mel = lambda_mel
        self.weight_beta = weight_beta
        self.seed = seed

    def forward(self, clean_audio: torch.Tensor, cloned_output: torch.Tensor) -> dict:
        device = cloned_output.device
        if self.seed is not None:
            gen = torch.Generator(device=device).manual_seed(self.seed)
            random_noise = torch.randn(cloned_output.shape, generator=gen, device=device)
        else:
            random_noise = torch.randn_like(cloned_output)

        pivotal_loss = compute_pivotal_disruption_loss(self.mel_fn, clean_audio, cloned_output)
        l1_noise_loss = compute_l1_to_noise(self.mel_fn, cloned_output, random_noise)
        kl_noise_loss = compute_kl_to_noise(self.mel_fn, cloned_output, random_noise)

        total = (
            -self.lambda_mel * pivotal_loss
            + self.weight_beta * (l1_noise_loss + kl_noise_loss)
        )

        return {
            "total": total,
            "pivotal_disruption": pivotal_loss.detach(),
            "l1_to_noise": l1_noise_loss.detach(),
            "kl_to_noise": kl_noise_loss.detach(),
        }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    disruption_loss = SafeSpeechDisruptionLoss(sampling_rate=16000).to(device)

    dummy_clean = torch.randn(1, 1, 16000, device=device)
    dummy_cloned = torch.randn(1, 1, 19968, device=device, requires_grad=True)  # matches YourTTS's own output length variability

    out = disruption_loss(dummy_clean, dummy_cloned)
    print({k: v.item() for k, v in out.items()})
    out["total"].backward()
    print("Backward OK, grad reached cloned_output:", dummy_cloned.grad is not None)
