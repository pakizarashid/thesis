"""
src/models/surrogate_vc.py

Differentiable YourTTS surrogate for Stage 2's disruption loss. YourTTS's own
convenience methods (SpeakerManager.compute_embedding_from_clip,
ResNetSpeakerEncoder.compute_embedding, Vits.inference) are ALL decorated with
@torch.inference_mode(), which permanently detaches tensors from autograd --
unusable for training. Confirmed via direct source inspection (not assumed):

  - ResNetSpeakerEncoder.forward() itself has NO such decorator, is fully
    differentiable (real nn.Module ops throughout, no numpy -- confirmed
    use_torch_spec=True on this checkpoint, so mel-spectrogram computation is
    also a real nn.Module, not a numpy preprocessing step).
  - Vits.inference() IS decorated, but its internal computation is built from
    real differentiable submodules (text_encoder, duration_predictor, flow,
    waveform_decoder). The one genuine non-differentiable step is
    torch.ceil() on the duration predictor's output (used to build the hard
    text-to-frame alignment path) -- this is a structural property of
    VITS-style models, not a workaround artifact. It means gradients CANNOT
    flow back to the speaker embedding through the rhythm/duration pathway,
    but CAN flow through the two direct conditioning points that matter for a
    disruption loss: the normalizing flow (self.flow(..., g=g)) and the
    waveform decoder (self.waveform_decoder(..., g=g)). Document this as a
    known, structural limitation in the thesis methodology -- not a bug.

This module reimplements the embedding + inference computation verbatim
(matching the real source exactly), minus the decorators, calling the same
pretrained submodules and weights.

UNVERIFIED ASSUMPTIONS (confirm via the __main__ smoke test before trusting
this for real training):
  - Speaker encoder's expected input sample rate matches your watermarked
    audio's 16kHz. If not, add a differentiable resample step (torchaudio's
    Resample is conv-based and differentiable, safe to insert if needed).
  - Text tokenizer call signature (tts_model.tokenizer.text_to_ids) --
    standard Coqui TTS convention across versions, but not verified against
    this specific installed version's exact API.
"""

import os
import sys
import torch
import torch.nn as nn

from TTS.tts.utils.helpers import generate_path, sequence_mask


class YourTTSSurrogate(nn.Module):
    """
    Wraps a loaded YourTTS (Vits) model + its ResNet speaker encoder for
    differentiable use: watermarked audio -> speaker embedding -> cloned
    speech, with gradients flowing back through the embedding and the flow/
    decoder conditioning points (not through the duration/rhythm path, which
    is structurally non-differentiable in VITS -- see module docstring).

    The surrogate's own weights are always frozen (matches SafeSpeech's
    "surrogate-model-based universal perturbation" design -- the surrogate is
    a fixed proxy for a real cloner, only the upstream watermark generator's
    parameters should receive gradient updates).
    """

    def __init__(self, tts_model, speaker_encoder, tokenizer, device="cuda"):
        super().__init__()
        self.tts_model = tts_model
        self.speaker_encoder = speaker_encoder
        self.tokenizer = tokenizer
        self.device = device

        for p in self.tts_model.parameters():
            p.requires_grad = False
        for p in self.speaker_encoder.parameters():
            p.requires_grad = False
        self.tts_model.eval()
        self.speaker_encoder.eval()

    def compute_speaker_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform: [batch, 1, T] -- matches VoiceMarkBackbone's convention.
        Returns: [batch, embedding_dim] L2-normalized speaker embedding,
        differentiable w.r.t. waveform.

        Reimplements ResNetSpeakerEncoder.forward() usage directly (NOT
        compute_embedding/compute_embedding_from_clip, both @inference_mode
        decorated). forward() internally does x.squeeze_(1) IN-PLACE -- clone
        first so we don't mutate the caller's tensor.
        """
        x = waveform.clone()
        return self.speaker_encoder.forward(x, l2_norm=True)

    def differentiable_inference(self, text: str, d_vector: torch.Tensor, language: str = "en") -> dict:
        """
        Reimplementation of Vits.inference(), verbatim, minus @torch.inference_mode().
        d_vector: [batch, embedding_dim] speaker embedding (from
        compute_speaker_embedding -- keep this in the autograd graph, don't
        detach it here).

        Returns the same dict Vits.inference() returns:
        {'model_outputs', 'alignments', 'durations', 'z', 'z_p', 'm_p', 'logs_p', 'y_mask'}
        'model_outputs' is the generated waveform: [batch, 1, T_wav].
        """
        tts_model = self.tts_model
        token_ids = self.tokenizer.text_to_ids(text, language=language)
        x = torch.tensor(token_ids, dtype=torch.long, device=d_vector.device).unsqueeze(0)
        if d_vector.shape[0] > 1 and x.shape[0] == 1:
            x = x.expand(d_vector.shape[0], -1)

        # Look up the numeric language ID -- confirmed via
        # tts_model.language_manager.name_to_id, e.g. {'en': 0, 'fr-fr': 1, 'pt-br': 2}.
        # Without this, lang_emb stays None and the text encoder's internal
        # concatenation of [text_emb ; lang_emb] silently mismatches channel
        # counts (192 vs 196), which is what caused the earlier crash.
        language_id = None
        if hasattr(tts_model, "language_manager") and tts_model.language_manager is not None:
            language_id = tts_model.language_manager.name_to_id.get(language)
            if language_id is None:
                raise ValueError(
                    f"Language '{language}' not found in language_manager.name_to_id "
                    f"({tts_model.language_manager.name_to_id}). Use one of those keys."
                )
        language_ids_tensor = (
            torch.tensor([language_id] * d_vector.shape[0], dtype=torch.long, device=d_vector.device)
            if language_id is not None else None
        )

        aux_input = {"x_lengths": None, "d_vectors": d_vector, "speaker_ids": None,
                     "language_ids": language_ids_tensor, "durations": None}

        sid, g, lid, durations = tts_model._set_cond_input(aux_input)
        x_lengths = tts_model._set_x_lengths(x, aux_input)

        if tts_model.args.use_speaker_embedding and sid is not None:
            g = tts_model.emb_g(sid).unsqueeze(-1)
        # NOTE: when using d_vectors (our case), g is set via _set_cond_input
        # directly from aux_input['d_vectors'] -- the use_speaker_embedding
        # branch above is for discrete speaker_ids, not our zero-shot d_vector
        # path, but reproduced here for exact fidelity to the original.

        lang_emb = None
        if tts_model.args.use_language_embedding and lid is not None:
            lang_emb = tts_model.emb_l(lid).unsqueeze(-1)

        x_enc, m_p, logs_p, x_mask = tts_model.text_encoder(x, x_lengths, lang_emb=lang_emb)

        if durations is None:
            if tts_model.args.use_sdp:
                logw = tts_model.duration_predictor(
                    x_enc, x_mask,
                    g=g if tts_model.args.condition_dp_on_speaker else None,
                    reverse=True, noise_scale=tts_model.inference_noise_scale_dp,
                    lang_emb=lang_emb,
                )
            else:
                logw = tts_model.duration_predictor(
                    x_enc, x_mask, g=g if tts_model.args.condition_dp_on_speaker else None,
                    lang_emb=lang_emb,
                )
            w = torch.exp(logw) * x_mask * tts_model.length_scale
        else:
            w = durations.unsqueeze(0)

        # NOTE (see module docstring): torch.ceil() here is a hard, non-
        # differentiable step -- gradient does not flow back to g/d_vector
        # through this duration/alignment pathway. This is structural to
        # VITS, not something this reimplementation introduces.
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = sequence_mask(y_lengths, None).to(x_mask.dtype).unsqueeze(1)
        attn_mask = x_mask * y_mask.transpose(1, 2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1).transpose(1, 2))
        m_p = torch.matmul(attn.transpose(1, 2), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.transpose(1, 2), logs_p.transpose(1, 2)).transpose(1, 2)

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * tts_model.inference_noise_scale
        # g IS in the autograd graph here (differentiable conditioning point #1)
        z = tts_model.flow(z_p, y_mask, g=g, reverse=True)

        z, _, _, y_mask = tts_model.upsampling_z(z, y_lengths=y_lengths, y_mask=y_mask)
        # g IS in the autograd graph here too (differentiable conditioning point #2)
        o = tts_model.waveform_decoder((z * y_mask)[:, :, :tts_model.max_inference_len], g=g)

        return {
            "model_outputs": o, "alignments": attn.squeeze(1), "durations": w_ceil,
            "z": z, "z_p": z_p, "m_p": m_p, "logs_p": logs_p, "y_mask": y_mask,
        }

    def clone_voice(self, watermarked_audio: torch.Tensor, text: str = "This is a test sentence for voice cloning.",
                     language: str = "en") -> torch.Tensor:
        """
        Full differentiable pipeline: watermarked audio -> speaker embedding
        -> cloned speech in that voice. Returns [batch, 1, T_wav].
        Gradients flow from the returned waveform back to watermarked_audio
        through the speaker embedding and the flow/decoder conditioning
        (NOT through duration/rhythm -- see module docstring).
        """
        d_vector = self.compute_speaker_embedding(watermarked_audio)
        out = self.differentiable_inference(text, d_vector, language=language)
        return out["model_outputs"]


def load_yourtts_surrogate(device: str = "cuda" if torch.cuda.is_available() else "cpu") -> YourTTSSurrogate:
    from TTS.api import TTS
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False)
    tts_model = tts.synthesizer.tts_model.to(device)
    speaker_encoder = tts.synthesizer.tts_model.speaker_manager.encoder.to(device)
    tokenizer = tts.synthesizer.tts_model.tokenizer
    return YourTTSSurrogate(tts_model, speaker_encoder, tokenizer, device=device)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[surrogate_vc] Loading YourTTS surrogate on {device}...")
    surrogate = load_yourtts_surrogate(device=device)

    print("[surrogate_vc] Checking speaker encoder's expected sample rate...")
    sr = surrogate.speaker_encoder.audio_config.get("sample_rate", "UNKNOWN -- check manually")
    print(f"  Speaker encoder audio_config sample_rate: {sr}")
    print(f"  (Your watermarked audio is 16kHz -- if this doesn't match, "
          f"a differentiable resample step needs to be added before compute_speaker_embedding)")

    print("\n[surrogate_vc] Testing differentiable forward + backward pass...")
    dummy_watermarked_audio = torch.randn(1, 1, 16000, device=device, requires_grad=True)

    cloned = surrogate.clone_voice(dummy_watermarked_audio, text="Testing the surrogate.")
    print(f"  Cloned output shape: {cloned.shape}")

    loss = cloned.mean()
    loss.backward()

    grad = dummy_watermarked_audio.grad
    if grad is None:
        print("  FAILURE: gradient did NOT flow back to the input audio. "
              "Autograd graph is broken somewhere -- likely the embedding or "
              "conditioning path needs further debugging.")
    else:
        print(f"  SUCCESS: gradient flowed back to input audio. "
              f"Grad norm: {grad.norm().item():.6f}, "
              f"nonzero elements: {(grad != 0).sum().item()} / {grad.numel()}")
