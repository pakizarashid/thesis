#!/bin/bash
# scripts/setup_env.sh
#
# Installs every dependency this project needs, in ONE call, in the correct
# order. Order matters: speechtokenizer's install can pull in a newer
# transformers than coqui-tts tolerates, so the pin is applied AFTER
# speechtokenizer and BEFORE coqui-tts (see this project's own debugging
# history -- installing in the wrong order previously broke the transformers
# version coqui-tts needs).
#
# Usage (from repo root, after cloning):
#   bash scripts/setup_env.sh

set -e  # stop on first error, so a failed install doesn't silently continue

echo "[setup_env] Installing core dependencies..."
pip install -q beartype speechtokenizer soundfile einops omegaconf librosa

echo "[setup_env] Pinning transformers (required by coqui-tts, must come after speechtokenizer)..."
pip install -q "transformers>=4.57,<5"

echo "[setup_env] Installing coqui-tts (YourTTS surrogate)..."
pip install -q coqui-tts

echo "[setup_env] Installing quality-metric tools (PESQ/STOI/WER)..."
pip install -q pesq pystoi openai-whisper jiwer

echo "[setup_env] Clearing pip cache to save disk space..."
pip cache purge

echo "[setup_env] Done. Verifying key imports..."
python -c "
import torch, torchaudio, speechtokenizer, transformers
print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
print('transformers:', transformers.__version__)
try:
    from TTS.api import TTS
    print('coqui-tts: OK')
except Exception as e:
    print('coqui-tts: FAILED --', e)
"
echo "[setup_env] Setup complete."
