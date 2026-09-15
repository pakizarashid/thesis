"""
cloner_watermark_eval.py -- watermark survival through zero-shot voice cloning,
across multiple cloner architectures, with ONE evaluation protocol.

WHY THIS SCRIPT EXISTS
----------------------
The claim under test is:

    "Watermark survival depends strongly on the zero-shot cloning architecture and
     the information it preserves from the reference audio."

That claim is only as strong as the number of architectures it is measured on, and
it is only *credible* if every architecture goes through the SAME detector, the SAME
16-bit payload, the SAME reference audio and the SAME message seeds. A per-cloner
bespoke script invites per-cloner bugs, and a per-cloner bug is indistinguishable
from the architecture effect being claimed.

So: one script, one protocol, cloner selected by --cloner.

    YourTTS   fixed d-vector              0.5337   (measured, xtts_transfer_eval.py)
    XTTS-v2   GPT audio-prompt tokens     0.6119   (measured)
    F5-TTS    mel infilling               0.9300   (measured, f5tts_watermark_only.py)
    CosyVoice flow-matching + prompt mel   ?       <- this script
    MaskGCT   masked RVQ acoustic infill   ?       <- this script

CosyVoice and MaskGCT are the two remaining models from VoiceMark's own evaluation
set (published 0.964 and 0.957). They are the arms that decide whether the
architecture account is a real mechanism or a two-point coincidence.

ENVIRONMENT ISOLATION
---------------------
Every cloner backend is imported LAZILY, inside its own loader function. Nothing at
module scope imports f5_tts, cosyvoice, or Amphion. This script therefore runs in
any of the three environments without the other two installed, and -- like
f5tts_watermark_only.py -- it imports NO coqui-tts code, so the transformers pin
conflict never arises.

Run each cloner in its OWN environment (separate Kaggle notebook / conda env).
Do not try to co-install them.

TWO INPUT MODES
---------------
1. Default: watermark LibriSpeech eval audio here, then clone it.
   -> the "watermark only" arm, directly comparable to VoiceMark's published numbers.

2. --input_wav_dir: read audio that was ALREADY protected elsewhere, as
   sample{i}_{suffix}.wav. Written by demucs_fallback_eval.py --save_clones_dir
   (audio/ subdirectory). Gives the protected and post-DEMUCS arms with no PGD
   re-run and no coqui-tts in this environment.
   --wav_suffix picks the arm: clean | watermarked | denoised.

   In this mode the message must be REGENERATED with the same seed convention used
   when the WAVs were made (123 + utterance index). A mismatch does not error --
   it silently produces chance accuracy, which reads exactly like "protection
   destroys attribution". The seed guard below aborts on utterance 0 instead.
"""

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))

# Deliberately NOT importing anything from disruption_pgd or surrogate_vc: those pull
# in the coqui-tts surrogate. backbone/adapters/librispeech are torch-only.
from backbone import VoiceMarkBackbone
from adapters import apply_lora_adapters
from librispeech import LibriSpeechSubset, collate_librispeech


# --------------------------------------------------------------------------------
# detector side -- identical to f5tts_watermark_only.py, deliberately duplicated so
# the two scripts cannot drift apart through a shared-module edit
# --------------------------------------------------------------------------------

def preflight():
    """
    Cheap environment checks that run BEFORE any model load or dataset scan.

    The VoiceMark checkpoint is a pickle referencing omegaconf classes, so torch.load()
    fails with "No module named 'omegaconf.base'" when omegaconf is missing or older
    than 2.0. That traceback points at torch.serialization and reads like a corrupt
    checkpoint; it is an environment problem, and it has now cost time twice -- once
    when the 'denoiser' package downgraded omegaconf, once on a CosyVoice env where the
    requirements install aborted before omegaconf was ever installed.

    Installing a cloner's pinned requirements can also silently downgrade numpy under
    the installed torch. Both are checked here so the failure names its own fix.
    """
    problems, ver = [], None
    try:
        import omegaconf
        import omegaconf.base  # noqa: F401
        ver = getattr(omegaconf, "__version__", "unknown")
    except ImportError as e:
        problems.append(
            f"omegaconf missing or too old ({e}). The VoiceMark checkpoint unpickles\n"
            f"  omegaconf classes, so torch.load() cannot succeed.\n"
            f"  FIX:  pip install 'omegaconf>=2.3.0'")

    try:
        import numpy as _np
        import torch as _t
        _t.from_numpy(_np.zeros(2, dtype=_np.float32))
    except Exception as e:
        problems.append(
            f"numpy/torch ABI mismatch ({e}). A cloner's pinned requirements most likely\n"
            f"  downgraded numpy under the installed torch.\n"
            f"  FIX:  reinstall a numpy matching your torch build, then re-run.")

    if problems:
        raise SystemExit("\nPREFLIGHT FAILED\n\n" + "\n\n".join("  " + p for p in problems) + "\n")
    print(f"[preflight] omegaconf {ver} ok; numpy/torch ok")


def build_backbone(checkpoint_path, r=8, alpha=16, msgproc_lora_r=None):
    # msgproc_lora_r wiring (2026-09-14): matches disruption_pgd.py /
    # xtts_transfer_eval.py / save_audio_samples.py -- only needed to load a
    # checkpoint trained with train_route2_clone_aware.py --msgproc_lora_r.
    # MUST match the value used at training time or load_state_dict will hit
    # a shape mismatch on msg_processor's LoRA tensors.
    backbone = VoiceMarkBackbone()
    target_ranks = {"msg_processor": msgproc_lora_r} if msgproc_lora_r else None
    apply_lora_adapters(backbone, r=r, alpha=alpha, target_ranks=target_ranks)
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        backbone.model.load_state_dict(ckpt["lora_state_dict"], strict=False)
        print(f"[backbone] loaded {checkpoint_path} (epoch {ckpt.get('epoch')})")
    else:
        print("[backbone] LoRA at zero-init == PRETRAINED VOICEMARK (their released weights)")
    for p in backbone.model.parameters():
        p.requires_grad = False
    backbone.model.eval()
    return backbone


def random_message(nbits, batch_size, device, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.randint(0, 2, (batch_size, nbits), generator=gen, device=device)


def compute_detection_accuracy(chunk_logits, message, nchunk_size=4):
    pred_chunks = torch.argmax(chunk_logits, dim=-1)
    batch, nchunks = pred_chunks.shape
    correct, total = 0, 0
    for i in range(nchunks):
        true_chunk = message[:, i * nchunk_size:(i + 1) * nchunk_size]
        pred_val = pred_chunks[:, i]
        for b in range(nchunk_size):
            correct += (((pred_val >> b) & 1) == true_chunk[:, b]).sum().item()
            total += batch
    return correct / total


def detect_acc(backbone, wav, message):
    with torch.no_grad():
        feat = backbone.model.st_model.forward_feature(wav)
        _logits, chunk_logits = backbone.model.detector(feat)
    return compute_detection_accuracy(chunk_logits, message)


# --------------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------------

def write_ref_wav(speaker_audio_16k, tmp_dir, tag):
    """All three backends take a reference WAV path. Write once, reuse."""
    import soundfile as sf
    os.makedirs(tmp_dir, exist_ok=True)
    path = os.path.join(tmp_dir, f"ref_{tag}.wav")
    sf.write(path, speaker_audio_16k.detach().cpu().reshape(-1).numpy(), 16000)
    return path


def to_16k_tensor(wav, sr, device):
    """Backends return numpy or torch at 24 kHz. The detector wants (1,1,T) @ 16 kHz."""
    import torchaudio
    t = torch.as_tensor(wav, dtype=torch.float32).reshape(1, 1, -1)
    if sr != 16000:
        t = torchaudio.functional.resample(t, sr, 16000)
    return t.to(device)


class Transcriber:
    """
    CosyVoice and MaskGCT both REQUIRE the prompt transcript; they will not infer it.
    F5-TTS auto-transcribes internally with faster-whisper when ref_text="".

    Transcribing here with faster-whisper makes all three backends use the SAME
    transcript source, which is the point -- a different transcription pipeline per
    cloner would be a confound sitting directly on top of the effect being measured.

    Transcribes the REFERENCE AUDIO ACTUALLY PASSED TO THE CLONER (watermarked or
    protected), not the clean original, because that is what F5-TTS does internally.
    """

    def __init__(self, model_size="base.en", device="cuda"):
        self.model_size = model_size
        self.device = device
        self._m = None

    def _model(self):
        if self._m is None:
            from faster_whisper import WhisperModel
            compute = "float16" if self.device == "cuda" else "int8"
            print(f"[transcribe] loading faster-whisper {self.model_size} ({compute})")
            self._m = WhisperModel(self.model_size, device=self.device, compute_type=compute)
        return self._m

    def __call__(self, wav_path):
        segments, _info = self._model().transcribe(wav_path, language="en", beam_size=5)
        text = " ".join(s.text for s in segments).strip()
        # An empty transcript makes CosyVoice/MaskGCT produce garbage rather than
        # error. Fall back to a neutral prompt and say so, loudly.
        if not text:
            # 2026-09-15 fix: this used to warn and then return the empty string
            # anyway, which crashes CosyVoice/MaskGCT deep inside their own text
            # normalizers (wetext's token_parser.load does `assert len(input) > 0`)
            # -- not on utterance 0, so the seed/config guard never catches it.
            # Substitute a short, content-neutral, definitely-non-empty prompt so
            # the cloner backend has something to normalize.
            text = "This is a voice sample."
            print(f"[transcribe] WARNING empty transcript for {wav_path} -- "
                  f"substituting neutral fallback prompt {text!r} so the cloner "
                  f"backend does not crash on an empty string.")
        return text


# --------------------------------------------------------------------------------
# backend: F5-TTS  (mel infilling -- measured 0.9300)
# --------------------------------------------------------------------------------

def load_f5tts(device, args):
    from f5_tts.api import F5TTS
    print("[f5tts] loading (first run downloads weights)...")
    return F5TTS(device=device)


def clone_f5tts(model, speaker_audio_16k, gen_text, tmp_dir, tag, ref_text, args):
    ref_path = write_ref_wav(speaker_audio_16k, tmp_dir, tag)
    # ref_text="" -> F5-TTS auto-transcribes with faster-whisper internally.
    wav, sr, _spect = model.infer(ref_file=ref_path, ref_text=ref_text or "", gen_text=gen_text)
    return to_16k_tensor(wav, sr, speaker_audio_16k.device)


# --------------------------------------------------------------------------------
# backend: CosyVoice 2  (VoiceMark published 0.964)
# --------------------------------------------------------------------------------

def load_cosyvoice(device, args):
    """
    Repo-based install, not a plain pip package:

        git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
        cd CosyVoice && git submodule update --init --recursive
        pip install -r requirements.txt
        python -c "from huggingface_hub import snapshot_download; \
          snapshot_download('FunAudioLLM/CosyVoice2-0.5B', \
          local_dir='pretrained_models/CosyVoice2-0.5B')"

    third_party/Matcha-TTS must be on sys.path BEFORE importing cosyvoice, or the
    import fails on a missing matcha module.
    """
    root = args.cosyvoice_root
    if not root or not os.path.isdir(root):
        raise SystemExit(
            "--cosyvoice_root must point at the cloned CosyVoice repo "
            f"(got {root!r}). See the docstring of load_cosyvoice for setup.")
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.join(root, "third_party", "Matcha-TTS"))

    model_dir = args.cosyvoice_model_dir or os.path.join(
        root, "pretrained_models", "CosyVoice2-0.5B")
    if not os.path.isdir(model_dir):
        raise SystemExit(f"CosyVoice model dir not found: {model_dir}")

    # The class name moved between releases: CosyVoice2 in the 2.x line, AutoModel in
    # newer builds. Try both rather than pinning a release the user may not have.
    cls, err = None, None
    for name in ("CosyVoice2", "AutoModel", "CosyVoice"):
        try:
            mod = __import__("cosyvoice.cli.cosyvoice", fromlist=[name])
            cls = getattr(mod, name)
            print(f"[cosyvoice] using {name} from cosyvoice.cli.cosyvoice")
            break
        except (ImportError, AttributeError) as e:
            err = e
    if cls is None:
        raise SystemExit(f"Could not import a CosyVoice class: {err}")

    try:
        model = cls(model_dir, load_jit=False, load_trt=False, fp16=False)
    except TypeError:
        model = cls(model_dir)  # AutoModel-style signature
    print(f"[cosyvoice] loaded {model_dir}; sample_rate={getattr(model, 'sample_rate', 24000)}")
    return model


def clone_cosyvoice(model, speaker_audio_16k, gen_text, tmp_dir, tag, ref_text, args):
    """
    inference_zero_shot(tts_text, prompt_text, prompt_wav_path) -> generator of dicts
    with key 'tts_speech'. Pass the PATH, not a pre-loaded tensor: CosyVoice2's
    frontend_zero_shot() calls its own internal load_wav() on this argument TWICE --
    once at 16 kHz for the LLM prompt tokens, once at 24 kHz for the flow-matching
    prompt mel. Handing it an already-loaded tensor makes that second internal
    load_wav() call torchaudio.load() on a tensor instead of a file, which fails
    deep inside TorchCodec with an unrelated-looking "video_tensor must be kUInt8".
    Confirmed against the official README example, which passes a path string.
    """
    ref_path = write_ref_wav(speaker_audio_16k, tmp_dir, tag)
    if not ref_text:
        raise SystemExit("CosyVoice requires a prompt transcript; --auto_transcribe is off "
                         "and --ref_text is empty.")

    chunks = []
    try:
        gen = model.inference_zero_shot(gen_text, ref_text, ref_path, stream=False)
    except TypeError:
        gen = model.inference_zero_shot(gen_text, ref_text, ref_path)
    for out in gen:
        chunks.append(out["tts_speech"].reshape(-1))
    if not chunks:
        raise RuntimeError("CosyVoice returned no audio chunks")

    wav = torch.cat(chunks)
    sr = int(getattr(model, "sample_rate", 24000))
    return to_16k_tensor(wav, sr, speaker_audio_16k.device)


# --------------------------------------------------------------------------------
# backend: MaskGCT  (VoiceMark published 0.957)
# --------------------------------------------------------------------------------

def load_maskgct(device, args):
    """
    Amphion monorepo, not a pip package. espeak-ng is a HARD system dependency (used by
    phonemizer's EspeakBackend, which the g2p tokenizer constructs unconditionally --
    even for English-only text, PhonemeBpeTokenizer.__init__ builds all six language
    backends):

        apt-get install -y espeak-ng
        git clone https://github.com/open-mmlab/Amphion.git

    DO NOT `pip install -r Amphion/models/tts/maskgct/requirements.txt` as-is: it pins
    torch==2.0.1, numpy==1.26.0, transformers==4.41.2 (would replace the working stack).

    Its language-dependency list needs trimming, not skipping wholesale -- traced through
    the ACTUAL chain (maskgct_utils -> g2p_generation -> g2p/__init__ -> cleaners, which
    unconditionally imports ALL SIX per-language modules regardless of what language the
    text is in):
      NEEDED:     pyopenjtalk + pykakasi (via cleaners -> japanese.py, fetched verbatim --
                  do not trust a prior summary of this file's imports again), jieba +
                  cn2an + pypinyin (via cleaners -> mandarin.py), unidecode + inflect
                  (via english.py), phonemizer + LangSegment (g2p/utils/g2p.py,
                  g2p/g2p/__init__.py)
      NOT needed: g2p_en (english.py uses unidecode/inflect, not g2p_en) -- the only
                  package from the upstream requirements.txt actually confirmed unused
    pyopenjtalk is the one real build risk (C++ extension); pykakasi and everything
    else here is pure Python. If pyopenjtalk has no prebuilt wheel for this Python
    version, expect the same class of failure openai-whisper==20231117 hit for CosyVoice.

    Weights (five safetensors components) download from amphion/MaskGCT on first use.
    --amphion_root must be the repo root, because maskgct_utils imports by the
    'models.tts.maskgct...' package path, and utils.util (for load_config) sits at
    the repo root too.
    """
    root = args.amphion_root
    if not root or not os.path.isdir(root):
        raise SystemExit(
            "--amphion_root must point at the cloned Amphion repo "
            f"(got {root!r}). See the docstring of load_maskgct for setup.")
    sys.path.insert(0, root)

    # Amphion's own code assumes cwd == repo root for a handful of relative resource
    # paths -- e.g. g2p/g2p/mandarin.py reads
    # "./models/tts/maskgct/g2p/sources/g2p_chinese_model/polychar.txt" AT IMPORT TIME
    # and calls exit() if it's not found there (surfaces as NameError outside a REPL,
    # since exit() isn't a builtin in a plain script). Same convention VoiceMark's own
    # vendored SBW model uses -- see backbone.py's _prev_cwd chdir pattern, which this
    # mirrors. chdir for the import + build + checkpoint-loading only, then always
    # restore, so this script's OWN relative paths (--data_root, --save_clones_dir,
    # --tmp_dir) keep resolving against the invocation directory afterward.
    _prev_cwd = os.getcwd()
    try:
        os.chdir(root)

        # This script's OWN top-of-file sys.path setup adds .../src/models (for
        # `from backbone import ...`) and .../src (which ALSO contains a models/
        # subdirectory) to sys.path for this process's whole lifetime. Amphion's repo
        # root ALSO has a top-level models/ directory (models/tts/maskgct/...). Two
        # real 'models' packages compete on sys.path -- whichever gets resolved and
        # cached in sys.modules FIRST wins for every subsequent `from models.X import
        # Y` in this process, regardless of what gets inserted at sys.path[0]
        # afterward (Python checks sys.modules before ever searching sys.path again).
        # Evict any stale entry so this import resolves fresh, now that Amphion's root
        # is at sys.path[0].
        for _mod_name in list(sys.modules):
            if _mod_name == "models" or _mod_name.startswith("models."):
                del sys.modules[_mod_name]

        from huggingface_hub import hf_hub_download
        import safetensors.torch
        from models.tts.maskgct.maskgct_utils import (
            build_semantic_model, build_semantic_codec, build_acoustic_codec,
            build_t2s_model, build_s2a_model, MaskGCT_Inference_Pipeline,
        )
        # Verify we actually got AMPHION's models package, not a same-named one from
        # elsewhere on sys.path -- fail loudly here rather than with a confusing
        # AttributeError deep inside build_semantic_model() a moment later.
        import models as _amphion_models_check
        _root_norm = os.path.normpath(root)
        _resolved = [os.path.normpath(p) for p in getattr(_amphion_models_check, "__path__", [])]
        if not any(p.startswith(_root_norm) for p in _resolved):
            raise RuntimeError(
                f"'models' resolved to {_resolved}, not under --amphion_root ({root}). "
                f"A same-named 'models' package elsewhere on sys.path is shadowing "
                f"Amphion's. sys.path[:6]={sys.path[:6]}")
        del _amphion_models_check, _root_norm, _resolved

        # NOT omegaconf -- Amphion has its own config loader (utils/util.py) that
        # returns a JsonHParams object with the same cfg.model.semantic_codec
        # attribute access OmegaConf would give, but this is what maskgct_utils'
        # build_* functions were actually written against. Same name-collision risk
        # as 'models' above ('utils' is an extremely common top-level name) -- evict
        # defensively here too.
        for _mod_name in list(sys.modules):
            if _mod_name == "utils" or _mod_name.startswith("utils."):
                del sys.modules[_mod_name]
        from utils.util import load_config

        cfg = load_config(os.path.join(root, "models/tts/maskgct/config/maskgct.json"))

        semantic_model, semantic_mean, semantic_std = build_semantic_model(device)
        semantic_codec = build_semantic_codec(cfg.model.semantic_codec, device)
        codec_encoder, codec_decoder = build_acoustic_codec(cfg.model.acoustic_codec, device)
        t2s_model = build_t2s_model(cfg.model.t2s_model, device)
        s2a_model_1layer = build_s2a_model(cfg.model.s2a_model.s2a_1layer, device)
        s2a_model_full = build_s2a_model(cfg.model.s2a_model.s2a_full, device)

        files = {
            "semantic_codec": "semantic_codec/model.safetensors",
            "codec_encoder": "acoustic_codec/model.safetensors",
            "codec_decoder": "acoustic_codec/model_1.safetensors",
            "t2s_model": "t2s_model/model.safetensors",
            "s2a_1layer": "s2a_model/s2a_model_1layer/model.safetensors",
            "s2a_full": "s2a_model/s2a_model_full/model.safetensors",
        }
        ckpt = {k: hf_hub_download("amphion/MaskGCT", filename=v) for k, v in files.items()}
        safetensors.torch.load_model(semantic_codec, ckpt["semantic_codec"])
        safetensors.torch.load_model(codec_encoder, ckpt["codec_encoder"])
        safetensors.torch.load_model(codec_decoder, ckpt["codec_decoder"])
        safetensors.torch.load_model(t2s_model, ckpt["t2s_model"])
        safetensors.torch.load_model(s2a_model_1layer, ckpt["s2a_1layer"])
        safetensors.torch.load_model(s2a_model_full, ckpt["s2a_full"])

        pipeline = MaskGCT_Inference_Pipeline(
            semantic_model, semantic_codec, codec_encoder, codec_decoder,
            t2s_model, s2a_model_1layer, s2a_model_full,
            semantic_mean, semantic_std, device,
        )
    finally:
        os.chdir(_prev_cwd)

    print("[maskgct] pipeline built")
    return pipeline


def clone_maskgct(model, speaker_audio_16k, gen_text, tmp_dir, tag, ref_text, args):
    """
    maskgct_inference(prompt_wav_path, prompt_text, target_text, "en", "en",
                      target_len=...) -> numpy audio @ 24 kHz.
    target_len=None lets the model choose its own duration.
    """
    # abspath BEFORE any chdir below -- otherwise this relative path would resolve
    # against the wrong directory once cwd changes.
    ref_path = os.path.abspath(write_ref_wav(speaker_audio_16k, tmp_dir, tag))
    if not ref_text:
        raise SystemExit("MaskGCT requires a prompt transcript; --auto_transcribe is off "
                         "and --ref_text is empty.")
    # Text-to-phoneme conversion (g2p/chn_eng_g2p) runs PER CALL here, not just at
    # pipeline-build time -- if it reads any relative-path resource the same way
    # mandarin.py's module-level polychar.txt load did, this needs cwd == Amphion
    # root too. Defensive, cheap: same chdir/restore as load_maskgct.
    _prev_cwd = os.getcwd()
    try:
        if args.amphion_root:
            os.chdir(args.amphion_root)
        wav = model.maskgct_inference(
            ref_path, ref_text, gen_text, "en", "en", target_len=args.target_len)
    finally:
        os.chdir(_prev_cwd)
    return to_16k_tensor(wav, 24000, speaker_audio_16k.device)


BACKENDS = {
    "f5tts":     (load_f5tts,     clone_f5tts,     "mel infilling",                 0.979),
    "cosyvoice": (load_cosyvoice, clone_cosyvoice, "flow matching on prompt mel",   0.964),
    "maskgct":   (load_maskgct,   clone_maskgct,   "masked RVQ acoustic infilling", 0.957),
}


# --------------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cloner", type=str, required=True, choices=sorted(BACKENDS),
                   help="Run ONE per environment. Do not co-install these.")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Omit for PRETRAINED VoiceMark (their released weights) -- that is "
                        "the arm that tests their published claim directly.")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--diagnostic", action="store_true",
                   help="One utterance, then stop. ALWAYS run this first: it catches a "
                        "broken install or a bad transcript in ~2 minutes instead of "
                        "after an hour of GPU time.")
    p.add_argument("--check_env", action="store_true",
                   help="Build the VoiceMark backbone and exit. Run this FIRST after "
                        "installing a cloner's requirements: it catches a dependency "
                        "downgrade in seconds, before any model download.")
    p.add_argument("--n_utterances", type=int, default=100)
    p.add_argument("--save_clones_dir", type=str, default=None)

    p.add_argument("--input_wav_dir", type=str, default=None,
                   help="PRE-MADE-AUDIO MODE. Read sample{i}_{wav_suffix}.wav from here "
                        "instead of watermarking in this process. Written by "
                        "demucs_fallback_eval.py --save_clones_dir (audio/ subdir).")
    p.add_argument("--wav_suffix", type=str, default="watermarked",
                   help="Arm to read: 'clean' (unprotected control) | 'watermarked' "
                        "(watermark + PGD) | 'denoised' (protected, after DEMUCS).")
    p.add_argument("--message_seed_base", type=int, default=123,
                   help="Must match the generating script (demucs_fallback_eval.py uses "
                        "123 + index). A mismatch yields chance accuracy silently; the "
                        "source-ACC guard aborts on utterance 0 if it does not match.")

    p.add_argument("--gen_text", type=str,
                   default="This is a test sentence for voice cloning.")
    p.add_argument("--ref_text", type=str, default="",
                   help="Fixed prompt transcript. Leave empty and use --auto_transcribe.")
    p.add_argument("--gen_text_from_transcript", action="store_true",
                   help="CARRIER-PROBE (2026-09-12): use each utterance's own "
                        "LibriSpeech ground-truth transcript as BOTH the synthesis "
                        "target (gen_text) and the prompt transcript (ref_text), "
                        "instead of the fixed --gen_text sentence. Requires "
                        "LibriSpeech input (ignored in --input_wav_dir mode, where no "
                        "transcript is available). Makes the clone say the same words "
                        "as the reference, so a downstream frame-by-frame latent "
                        "comparison (carrier_probe.py) is not confounded by content "
                        "mismatch. Default off -- every other use of this script for "
                        "ACC-style measurement is unaffected.")
    p.add_argument("--auto_transcribe", action="store_true",
                   help="Transcribe each reference with faster-whisper. REQUIRED for "
                        "cosyvoice and maskgct (they do not infer the prompt text); "
                        "optional for f5tts, which does it internally.")
    p.add_argument("--whisper_size", type=str, default="base.en")
    p.add_argument("--target_len", type=float, default=None, help="MaskGCT only.")

    p.add_argument("--cosyvoice_root", type=str, default=None)
    p.add_argument("--cosyvoice_model_dir", type=str, default=None)
    p.add_argument("--amphion_root", type=str, default=None)

    p.add_argument("--tmp_dir", type=str, default="./tmp_cloner")
    p.add_argument("--data_root", type=str, default="./data/librispeech")
    p.add_argument("--n_speakers", type=int, default=60)
    p.add_argument("--utterances_per_speaker", type=int, default=15)
    p.add_argument("--n_eval_speakers", type=int, default=20)
    p.add_argument("--eval_utterances_per_speaker", type=int, default=5)
    p.add_argument("--crop_seconds", type=float, default=3.0)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--msgproc_lora_r", type=int, default=None,
                    help="Only needed to load a checkpoint trained with "
                         "train_route2_clone_aware.py --msgproc_lora_r -- MUST match "
                         "the value used at training time or load_state_dict will hit "
                         "a shape mismatch on msg_processor's LoRA tensors.")
    args = p.parse_args()

    preflight()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_fn, clone_fn, mechanism, published = BACKENDS[args.cloner]

    if args.cloner in ("cosyvoice", "maskgct") and not (args.auto_transcribe or args.ref_text):
        raise SystemExit(
            f"--cloner {args.cloner} needs the prompt transcript. Pass --auto_transcribe "
            f"(recommended: same faster-whisper path F5-TTS uses internally) or --ref_text.")

    transcribe = Transcriber(args.whisper_size, device) if args.auto_transcribe else None

    # --check_env exists because the expensive failure mode in this project is a
    # DEPENDENCY BREAK, not a logic bug: installing a cloner's requirements.txt can
    # downgrade a package the VoiceMark backbone needs (this already happened once --
    # `pip install denoiser` pulled omegaconf below 2.0 and broke checkpoint
    # unpickling). Building the backbone alone takes seconds; discovering the break
    # after a 10-minute model download and an hour of cloning does not.
    if args.check_env:
        backbone = build_backbone(args.checkpoint, args.lora_r, args.lora_alpha, msgproc_lora_r=args.msgproc_lora_r)
        backbone.model.to(device)
        wav = torch.randn(1, 1, 16000, device=device) * 0.01
        msg = random_message(16, 1, device, seed=0)
        acc = detect_acc(backbone, wav, msg)
        print(f"[check_env] backbone builds, detector runs (noise ACC {acc:.3f}, ~0.5 expected)")
        print(f"[check_env] torch {torch.__version__}, cuda={torch.cuda.is_available()}")
        if args.auto_transcribe:
            from faster_whisper import WhisperModel  # noqa: F401
            print("[check_env] faster-whisper importable")
        print("[check_env] OK -- the cloner install did not break the backbone.")
        return

    # The dataset is only needed when this process does the watermarking. In
    # pre-made-audio mode the WAVs are the input, so requiring LibriSpeech here would
    # block the protected/denoised arms in an environment that has no copy of it.
    loader = None
    if args.diagnostic or not args.input_wav_dir:
        eval_ds = LibriSpeechSubset(
            root=args.data_root, n_speakers=args.n_speakers,
            utterances_per_speaker=args.utterances_per_speaker,
            n_eval_speakers=args.n_eval_speakers,
            eval_utterances_per_speaker=args.eval_utterances_per_speaker,
            sample_rate=16000, crop_seconds=args.crop_seconds, split="eval",
        )
        loader = DataLoader(eval_ds, batch_size=1, shuffle=False,
                            collate_fn=collate_librispeech)

    backbone = build_backbone(args.checkpoint, args.lora_r, args.lora_alpha, msgproc_lora_r=args.msgproc_lora_r)
    backbone.model.to(device)
    model = load_fn(device, args)

    def trim_trailing_silence(waveform, eps: float = 1e-4):
        """
        CARRIER-PROBE (2026-09-12, third fix). Cuts trailing near-zero samples --
        the _crop_or_pad padding LibriSpeechSubset appends when an utterance is
        shorter than --crop_seconds -- off a reference waveform. Without this,
        --crop_seconds 20 pads most (shorter) LibriSpeech utterances with several
        seconds of trailing silence, which carrier_probe.py's frame-count-mismatch
        check misreads as a content mismatch. Same fix already applied to
        gen_samples_yourtts.py/gen_samples_xtts.py -- see that file's docstring.
        """
        w = waveform.squeeze(0) if waveform.dim() > 1 else waveform
        nz = (w.abs() > eps).nonzero()
        if nz.numel() == 0:
            return waveform
        last = nz[-1].item()
        trimmed = w[: last + 1]
        return trimmed.unsqueeze(0) if waveform.dim() > 1 else trimmed

    def clone_one(recon_wm, tag, own_transcript=None):
        gen_text = args.gen_text
        ref_text = args.ref_text
        if args.gen_text_from_transcript and own_transcript:
            # Known ground truth -- use it for both roles, skip whisper entirely.
            gen_text = own_transcript
            ref_text = own_transcript
        elif transcribe is not None:
            ref_text = transcribe(write_ref_wav(recon_wm, args.tmp_dir, tag))
        return clone_fn(model, recon_wm, gen_text, args.tmp_dir, tag, ref_text, args)

    if args.diagnostic:
        print("\n" + "=" * 64 + f"\nDIAGNOSTIC ({args.cloner}, 1 utterance)\n" + "=" * 64)
        batch = next(iter(loader))
        clean = trim_trailing_silence(batch["waveform"][0]).unsqueeze(0).to(device)
        msg = random_message(16, 1, device, seed=args.message_seed_base)
        with torch.no_grad():
            recon_wm = backbone.forward_full(clean, msg)["recon_wm"]
        diag_transcript = (batch.get("transcript") or [None])[0]
        cloned = clone_one(recon_wm, "diag", own_transcript=diag_transcript)
        print(f"  recon_wm {tuple(recon_wm.shape)}  clone {tuple(cloned.shape)}")
        print(f"  ACC on watermarked source : {detect_acc(backbone, recon_wm, msg):.4f}  (expect ~0.99)")
        print(f"  ACC on {args.cloner} clone     : {detect_acc(backbone, cloned, msg):.4f}  (0.5 = chance)")
        print("\n  n=1 is indicative only. If shapes are sane, the clone is non-trivial in")
        print("  length, and source ACC is ~0.99, run for real.")
        return

    accs_src, accs_clone = [], []
    print(f"\n{'=' * 78}")
    print(f"WATERMARK SURVIVAL THROUGH CLONING | cloner={args.cloner} ({mechanism})")
    print(f"checkpoint: {'PRETRAINED VoiceMark' if args.checkpoint is None else args.checkpoint}")
    print(f"VoiceMark published for this cloner: {published}")
    print(f"{'=' * 78}")

    if args.input_wav_dir:
        import glob, re
        pattern = f"sample*_{args.wav_suffix}.wav"
        wavs = sorted(
            glob.glob(os.path.join(args.input_wav_dir, pattern)),
            key=lambda q: int(re.search(r"sample(\d+)_", os.path.basename(q)).group(1)),
        )
        if not wavs:
            raise SystemExit(
                f"No {pattern} in {args.input_wav_dir}\n"
                f"These come from demucs_fallback_eval.py --save_clones_dir (audio/ subdir).")
        print(f"[input] PRE-MADE-AUDIO MODE ('{args.wav_suffix}' arm): {len(wavs)} WAVs")
        source = [("wav", w) for w in wavs]
    else:
        print("[input] watermarking LibriSpeech eval audio in-process")
        source = [("batch", b) for b in loader]

    for i, (kind, item) in enumerate(source):
        if i >= args.n_utterances:
            break
        msg = random_message(16, 1, device, seed=args.message_seed_base + i)

        if kind == "wav":
            import soundfile as sf
            arr, sr = sf.read(item)
            assert sr == 16000, f"{item} is {sr} Hz, expected 16000"
            recon_wm = torch.as_tensor(arr, dtype=torch.float32).reshape(1, 1, -1).to(device)
        else:
            clean = trim_trailing_silence(item["waveform"][0]).unsqueeze(0).to(device)
            with torch.no_grad():
                recon_wm = backbone.forward_full(clean, msg)["recon_wm"]

        loop_transcript = (item.get("transcript") or [None])[0] if kind == "batch" else None
        # 2026-09-15 fix: a cloner backend (CosyVoice/MaskGCT, via wetext's text
        # normalizer) can raise on a single pathological utterance -- not just a
        # literally-empty transcript (handled separately in Transcriber.__call__),
        # but also a short/unusual one that its OWN internal reordering reduces to
        # zero tokens. Rather than enumerate every string a third-party normalizer
        # chokes on, skip this utterance loudly and keep the unattended run alive;
        # n_completed (see _dump) already tolerates fewer results than n_utterances.
        try:
            cloned = clone_one(recon_wm, f"u{i}", own_transcript=loop_transcript)
        except Exception as e:
            print(f"  [{i}] SKIPPED -- {args.cloner} cloning failed: "
                  f"{type(e).__name__}: {e}", flush=True)
            continue

        a_src = detect_acc(backbone, recon_wm, msg)
        a_cln = detect_acc(backbone, cloned, msg)
        accs_src.append(a_src)
        accs_clone.append(a_cln)
        print(f"  [{i}] acc_source={a_src:.4f}  acc_{args.cloner}_clone={a_cln:.4f}", flush=True)

        # SEED-MISMATCH GUARD. acc_source is detection on the INPUT audio itself, so it
        # must be ~0.99 when the regenerated message matches the embedded one. Near 0.5
        # means the seed convention differs and every clone number below is meaningless.
        # Fail loudly rather than let that be read as an architecture result.
        # 'clean' is the unprotected control -- it carries NO watermark, so a_src near
        # 0.5 is the correct result there, not a seed mismatch. Guard the marked arms only.
        if args.input_wav_dir and args.wav_suffix != "clean" and i == 0 and a_src < 0.85:
            raise SystemExit(
                f"\nABORT: acc_source={a_src:.4f} on the INPUT audio (expected ~0.99).\n"
                f"The message regenerated with seed {args.message_seed_base}+{i} does not "
                f"match the one embedded in {args.input_wav_dir}.\n"
                f"Fix --message_seed_base to match the generating script.")

        if args.save_clones_dir:
            import soundfile as sf
            os.makedirs(args.save_clones_dir, exist_ok=True)
            sf.write(os.path.join(args.save_clones_dir, f"sample{i}_reference.wav"),
                     recon_wm[0].detach().cpu().reshape(-1).numpy(), 16000)
            sf.write(os.path.join(args.save_clones_dir, f"sample{i}_clone_{args.cloner}.wav"),
                     cloned[0].detach().cpu().reshape(-1).numpy(), 16000)

        if args.output and (i + 1) % 5 == 0:
            _dump(args, mechanism, published, accs_src, accs_clone)

    m_src = sum(accs_src) / len(accs_src)
    m_cln = sum(accs_clone) / len(accs_clone)
    print(f"\n{'=' * 78}")
    print(f"RESULT  cloner={args.cloner}  n={len(accs_clone)}")
    print(f"{'=' * 78}")
    print(f"  ACC on watermarked source audio : {m_src:.4f}   (sanity -- the mark is present)")
    print(f"  ACC on {args.cloner} clone [KEY]      : {m_cln:.4f}   (0.5 = chance)")
    print(f"\n  Conditioning-bandwidth ladder, same detector and payload throughout:")
    print(f"    YourTTS   fixed d-vector          0.5337")
    print(f"    XTTS-v2   audio-prompt tokens     0.6119")
    print(f"    F5-TTS    mel infilling           0.9300")
    print(f"    {args.cloner:9s} {mechanism:24s} {m_cln:.4f}   <- this run")
    print(f"\n  VoiceMark published for {args.cloner}: {published}")
    print(f"\n  >=0.90 -> lands with F5-TTS at the high-bandwidth end. The architecture")
    print(f"            account holds on a third/fourth independent model.")
    print(f"  ~0.6-0.9 -> a gradient, not two clusters. Report the ladder as continuous")
    print(f"            and relate ACC to how much reference detail each model carries.")
    print(f"  <0.6   -> DO NOT report as an architecture result yet. Check first:")
    print(f"            transcript quality, clone length, resampling, and whether this")
    print(f"            model's prompt path resynthesises the reference before use.")

    if args.output:
        _dump(args, mechanism, published, accs_src, accs_clone)
        print(f"\n  saved -> {args.output}")


def _dump(args, mechanism, published, accs_src, accs_clone):
    with open(args.output, "w") as f:
        json.dump({
            "label": f"{args.cloner}_watermark_survival",
            "cloner": args.cloner,
            "conditioning_mechanism": mechanism,
            "voicemark_published": published,
            "checkpoint": args.checkpoint,
            "arm": args.wav_suffix if args.input_wav_dir else "watermarked_here",
            "input_wav_dir": args.input_wav_dir,
            "message_seed_base": args.message_seed_base,
            "auto_transcribe": args.auto_transcribe,
            "n_completed": len(accs_clone),
            "reference_points": {
                "yourtts_dvector": 0.5337,
                "xtts_prompt_tokens": 0.6119,
                "f5tts_mel_infill": 0.9300,
            },
            "results": {
                "acc_source_mean": sum(accs_src) / len(accs_src),
                "acc_clone_mean": sum(accs_clone) / len(accs_clone),
                "acc_source_values": accs_src,
                "acc_clone_values": accs_clone,
            },
        }, f, indent=2)


if __name__ == "__main__":
    main()
