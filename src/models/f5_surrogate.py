"""
src/models/f5_surrogate.py  (2026-10-26)

F5-TTS as the in-loop cloner for clone-aware watermark training, with the same interface as the YourTTS
surrogate in surrogate_vc.py:  clone_voice(audio [B, 1, T] at 16 kHz, text) -> clone [B, 1, T'] at 16 kHz.

The differentiable sampler is the one the anti-cloning perturbation Q2 already uses
(src/eval/f5_whitebox_pgd.py: F5Attacker = CFM.sample re-implemented without no_grad, K Euler steps with
classifier-free guidance, gradient checkpointing; Vocos decode called outside inference mode), so the
gradient reaches the watermarked waveform through the prompt mel. Nothing here is new maths; it is a
batch wrapper.

Per item:
  prompt transcript : faster-whisper base.en of the watermarked prompt (no grad), as F5-TTS itself would
                      transcribe a prompt given without text, and as Q2 does
  rollout noise     : seeded from a running counter (seed_base + call index), so a run is reproducible
                      and every call sees fresh noise
  output            : the generated region only (the prompt part removed), as f5_tts.api returns it,
                      Vocos-decoded at 24 kHz and resampled to 16 kHz
The batch is truncated to its shortest clone (lengths differ by a few frames because the duration rule
depends on the transcript length), so the watermark decoder sees one tensor, as with YourTTS.
"""
import os
import sys

import numpy as np
import torch
import torchaudio

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))


class F5Surrogate:
    def __init__(self, device, steps=6, cfg=2.0, seed_base=11000, whisper_size="base.en"):
        import f5_whitebox_pgd as wb
        from f5_tts.api import F5TTS
        from faster_whisper import WhisperModel
        self.device = device
        self.att = wb.F5Attacker(F5TTS(device=device), steps=steps, cfg=cfg)
        self.asr = WhisperModel(whisper_size, device="cuda" if str(device).startswith("cuda") else "cpu",
                                compute_type="float16" if str(device).startswith("cuda") else "int8")
        self.seed_base, self.calls = seed_base, 0

    def transcribe(self, wav16k):
        x = np.ascontiguousarray(wav16k.detach().float().cpu().reshape(-1).numpy())
        if not np.isfinite(x).all():       # never hand Whisper a non-finite signal (it stalled on one, 2026-10-26)
            return "the quick brown fox jumps over the lazy dog"
        segs, _ = self.asr.transcribe(x, language="en", beam_size=5)
        t = " ".join(s.text for s in segs).strip()
        return t if t else "the quick brown fox jumps over the lazy dog"

    def clone_one(self, audio16k, text):
        a = self.att
        ref_text = self.transcribe(audio16k)
        a24 = a.prompt_audio_24k(audio16k)
        text_list, duration = a.prepare(a24.shape[-1], ref_text, text)
        cond = a.prompt_mel(audio16k)                       # carries the gradient
        n = cond.shape[1]
        seed = self.seed_base + self.calls
        self.calls += 1
        mel = a.generate_mel(cond, text_list, duration, seed=seed)[:, n:]
        wav24 = a.vocode(mel)                               # [1, T24]
        return torchaudio.functional.resample(wav24.float(), a.sr, 16000).reshape(-1)

    def clone_voice(self, watermarked_audio, text="This is a test sentence for voice cloning."):
        outs = [self.clone_one(watermarked_audio[b].reshape(1, -1), text) for b in range(watermarked_audio.shape[0])]
        L = min(o.shape[-1] for o in outs)
        return torch.stack([o[:L] for o in outs]).unsqueeze(1)   # [B, 1, L]
