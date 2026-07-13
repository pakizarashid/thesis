"""
src/models/backbone.py

Wraps VoiceMark's SBW model (SpeechTokenizer RVQ codec + WMEmbedder + WMDetector)
and its pretrained msstftd discriminator, loaded from the official checkpoint.

Confirmed from checkpoint inspection (Week 1 audit):
  - model_state_dict contains ONLY 'msg_processor' (WMEmbedder) and 'detector'
    (WMDetector) weights. st_model (SpeechTokenizer) is NOT in this checkpoint --
    it's loaded separately from speechtokenizer/pretrained_model/ and was frozen
    throughout VoiceMark's own training. This confirms our freeze plan: st_model
    stays frozen, adapters attach to msg_processor / detector paths only.
  - adversaries_state_dict['msstftd'] contains a fully-trained multi-scale STFT
    discriminator (91 tensors, 5 sub-discriminators, filters=32) with live Adam
    optimizer state (lr=5e-5, betas=[0.5, 0.9]). This matches
    speechtokenizer.discriminators.MultiScaleSTFTDiscriminator(filters=32) from
    the official speechtokenizer PyPI package -- confirmed via its documented
    usage (discriminators = {'mstftd': MultiScaleSTFTDiscriminator(32)}) and its
    5-entry default n_ffts/hop_lengths/win_lengths lists matching the 5
    discriminators.0..4 seen in the checkpoint.
  - epoch: 46 (paper reports 30 epochs trained; this checkpoint is a later/
    continued run, not necessarily the exact paper checkpoint).

IMPORTANT -- two different packages share the name 'speechtokenizer' in this
project: the PIP-INSTALLED speechtokenizer package (used only for its
discriminators.py, which VoiceMark's vendored copy does not include) and the
VENDORED copy at external/voicemark/speechtokenizer/ (used by SBW's st_model,
and required for compatibility with the VoiceMark checkpoint's architecture).
Because both register themselves under sys.modules['speechtokenizer'], whichever
one is imported first in a given process "wins" for the rest of that process
unless we explicitly purge the cache between loads. All loading in this file is
done through functions that purge sys.modules and scope sys.path around each
import, so VoiceMarkBackbone() and VoiceMarkDiscriminator() are safe to
construct in EITHER order, or repeatedly, within the same process.

Weight keys in the checkpoint are saved with a 'module.' prefix (trained under
DataParallel/DDP). This loader strips that prefix, matching WatermarkSolver's
load_model logic in the original infer.py.
"""

import os
import sys
import torch
import torch.nn as nn

VOICEMARK_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "external", "voicemark"
)
DEFAULT_CHECKPOINT = os.path.join(VOICEMARK_ROOT, "voicemark.pth")


def _purge_module_cache(*prefixes: str):
    """Remove any cached sys.modules entries whose name equals or starts with
    one of the given prefixes (e.g. 'speechtokenizer', 'models')."""
    for mod_name in list(sys.modules.keys()):
        for prefix in prefixes:
            if mod_name == prefix or mod_name.startswith(prefix + "."):
                del sys.modules[mod_name]
                break


def _strip_module_prefix(state_dict: dict) -> dict:
    """Strip 'module.' prefix left over from DataParallel/DDP training."""
    return {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }


def _import_vendored_sbw():
    """
    Import SBW from external/voicemark/models.py, ensuring the VENDORED
    speechtokenizer package resolves (not any pip-installed one that might be
    cached from a prior _import_pip_msstftd_discriminator() call in this
    process).
    """
    _purge_module_cache("speechtokenizer", "models")
    if VOICEMARK_ROOT not in sys.path:
        sys.path.insert(0, VOICEMARK_ROOT)
    else:
        # ensure it's at the FRONT, in case something else was prepended since
        sys.path.remove(VOICEMARK_ROOT)
        sys.path.insert(0, VOICEMARK_ROOT)
    from models import SBW  # noqa: E402  (import intentionally deferred)
    return SBW


def _import_pip_msstftd_discriminator():
    """
    Import MultiScaleSTFTDiscriminator from the PIP-INSTALLED speechtokenizer
    package, ensuring VOICEMARK_ROOT is NOT on sys.path while doing so (so
    Python can't accidentally resolve 'speechtokenizer' to the vendored copy,
    which has no discriminators.py at all).
    """
    _purge_module_cache("speechtokenizer")
    had_voicemark_root = VOICEMARK_ROOT in sys.path
    if had_voicemark_root:
        sys.path.remove(VOICEMARK_ROOT)
    try:
        import speechtokenizer.discriminators as _disc  # pip package
        msstftd_cls = _disc.MultiScaleSTFTDiscriminator
    finally:
        _purge_module_cache("speechtokenizer")
        if had_voicemark_root and VOICEMARK_ROOT not in sys.path:
            sys.path.insert(0, VOICEMARK_ROOT)
    return msstftd_cls


class VoiceMarkBackbone(nn.Module):
    """
    Loads the pretrained SBW model. st_model (SpeechTokenizer / RVQ codec) is
    ALWAYS frozen -- it was never part of VoiceMark's own training. msg_processor
    (embedder) and detector are loaded pretrained and frozen by default; call
    `unfreeze()` after wrapping them with your adapter layers to control exactly
    which parameters get gradients.
    """

    def __init__(
        self,
        checkpoint_path: str = DEFAULT_CHECKPOINT,
        strict: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = device

        SBW = _import_vendored_sbw()

        # models.py's SBW.__init__ hardcodes RELATIVE paths for the SpeechTokenizer
        # checkpoint (e.g. "speechtokenizer/pretrained_model/...json"), which only
        # resolve correctly if the process's cwd is external/voicemark itself.
        # We chdir there just for construction, then restore, rather than editing
        # the vendored file.
        _prev_cwd = os.getcwd()
        try:
            os.chdir(VOICEMARK_ROOT)
            self.model = SBW()
        finally:
            os.chdir(_prev_cwd)

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.epoch = ckpt.get("epoch")

        raw_state = ckpt["model_state_dict"]
        state = _strip_module_prefix(raw_state)

        if not strict:
            model_sd = self.model.state_dict()
            state = {
                k: v
                for k, v in state.items()
                if k in model_sd and model_sd[k].shape == v.shape
            }

        missing, unexpected = self.model.load_state_dict(state, strict=False)
        # st_model keys will show up as "missing" here -- that's expected, since
        # SBW's __init__ already loads st_model from its own pretrained checkpoint
        # (speechtokenizer/pretrained_model/) independently of this state dict.
        st_model_missing = [k for k in missing if k.startswith("st_model.")]
        other_missing = [k for k in missing if not k.startswith("st_model.")]
        if other_missing:
            print(f"[VoiceMarkBackbone] WARNING - unexpected missing keys (not st_model): {other_missing}")
        if unexpected:
            print(f"[VoiceMarkBackbone] WARNING - unexpected keys in checkpoint: {unexpected}")
        print(
            f"[VoiceMarkBackbone] Loaded checkpoint from epoch {self.epoch}. "
            f"st_model loaded separately ({len(st_model_missing)} of its keys "
            f"correctly absent from this checkpoint)."
        )

        self.freeze_backbone()
        self.model.to(self.device)

    def freeze_backbone(self):
        """Freeze everything. Call before wrapping msg_processor/detector with adapters."""
        for p in self.model.parameters():
            p.requires_grad = False

    def unfreeze(self, submodule_names):
        """
        Unfreeze specific named submodules (e.g. ['msg_processor', 'detector'])
        for full fine-tuning. Prefer attaching adapters over this for Stage 1 --
        this is provided for ablations / sanity checks only.
        """
        for name in submodule_names:
            submodule = getattr(self.model, name, None)
            if submodule is None:
                raise ValueError(f"No submodule named '{name}' on SBW model")
            for p in submodule.parameters():
                p.requires_grad = True

    def forward(self, speech_input: torch.Tensor, message: torch.Tensor):
        """
        speech_input: [batch, 1, T] waveform tensor
        message: [batch, nbits] int tensor (0/1)
        Returns dict with 'recon' (unwatermarked reconstruction) and
        'recon_wm' (watermarked reconstruction).
        """
        return self.model(speech_input, message=message)

    def forward_full(self, speech_input: torch.Tensor, message: torch.Tensor):
        """
        Calls st_model.forward() DIRECTLY (bypassing SBW.forward, which discards
        two of the four returned tensors) to expose 'acoustic' and 'acoustic_wm'
        -- the pre-watermark and post-watermark speaker-specific latents needed
        for Lcos. Confirmed from speechtokenizer/model.py:

            acoustic = e - quantized_list[0]                          # pre-watermark
            acoustic_wm = sum(msg_processor(x, message) for x in subset)  # post-watermark
            return (o, o_wm, acoustic, acoustic_wm)

        Returns dict: {'recon', 'recon_wm', 'acoustic', 'acoustic_wm'}.
        """
        o, o_wm, acoustic, acoustic_wm = self.model.st_model(
            speech_input, msg_processor=self.model.msg_processor, message=message
        )
        wav_length = min(speech_input.size(-1), o_wm.size(-1))
        return {
            "recon": o[..., :wav_length],
            "recon_wm": o_wm[..., :wav_length],
            "acoustic": acoustic,
            "acoustic_wm": acoustic_wm,
        }

    def detect_watermark(self, x: torch.Tensor, return_logits: bool = False):
        return self.model.detect_watermark(x, return_logits=return_logits)


class VoiceMarkDiscriminator(nn.Module):
    """
    Loads the pretrained msstftd multi-scale STFT discriminator from
    adversaries_state_dict, using speechtokenizer.discriminators.MultiScaleSTFTDiscriminator
    (pip package) for VoiceMark's Ladv loss, instead of training a fresh
    discriminator from scratch.
    """

    def __init__(
        self,
        checkpoint_path: str = DEFAULT_CHECKPOINT,
        filters: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = device

        MultiScaleSTFTDiscriminator = _import_pip_msstftd_discriminator()
        self.discriminator = MultiScaleSTFTDiscriminator(filters=filters)

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        adv = ckpt["adversaries_state_dict"]["msstftd"]
        self.optimizer_state = adv.get("optimizer")
        weight_state = _strip_module_prefix(
            {k: v for k, v in adv.items() if k != "optimizer"}
        )
        # Checkpoint also carries an 'adversary.' prefix on discriminator keys
        # (e.g. 'adversary.discriminators.0.convs.0.conv.weight') that the pip
        # package's own state_dict() doesn't use -- strip that too.
        weight_state = {
            (k[len("adversary."):] if k.startswith("adversary.") else k): v
            for k, v in weight_state.items()
        }

        missing, unexpected = self.discriminator.load_state_dict(weight_state, strict=False)
        if missing:
            print(f"[VoiceMarkDiscriminator] WARNING - missing keys: {missing}")
        if unexpected:
            print(f"[VoiceMarkDiscriminator] WARNING - unexpected keys: {unexpected}")
        if not missing and not unexpected:
            print(
                f"[VoiceMarkDiscriminator] Loaded all {len(weight_state)} discriminator "
                f"weight tensors cleanly (filters={filters})."
            )

        self.discriminator.to(self.device)

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        """
        Confirmed signature from speechtokenizer.discriminators source:
            forward(self, y, y_hat) -> (logits, logits_fake, fmaps, fmaps_fake)
        where each 'logits'/'logits_fake' is a List[Tensor] (one per
        sub-discriminator) and each 'fmaps'/'fmaps_fake' is a
        List[List[Tensor]] (per sub-discriminator, per conv layer).
        y = real audio, y_hat = fake/generated audio. Single call computes
        both, sharing no weights but avoiding two separate Python-level calls.
        """
        return self.discriminator(y, y_hat)

    def build_optimizer(self, lr: float = None, betas=(0.5, 0.9)):
        """
        Builds a fresh Adam optimizer for the discriminator, optionally seeded
        with the checkpoint's original lr if none is given. NOTE: optimizer
        momentum/state itself is NOT restored (Adam's per-parameter state
        buffers aren't reliably remappable across a strict/non-strict reload
        without matching param ordering exactly) -- only the hyperparameters.
        If you need exact resumption, use torch.optim.Adam.load_state_dict
        with the raw self.optimizer_state dict and verify param order matches.
        """
        if lr is None and self.optimizer_state is not None:
            lr = self.optimizer_state["param_groups"][0]["lr"]
        elif lr is None:
            lr = 5e-5
        return torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)


if __name__ == "__main__":
    print("=== Loading VoiceMarkBackbone ===")
    backbone = VoiceMarkBackbone()
    print(backbone.model)

    print("\n=== Loading VoiceMarkDiscriminator ===")
    discriminator = VoiceMarkDiscriminator()
    print(discriminator.discriminator)

    print("\n=== Sanity: constructing backbone again after discriminator ===")
    backbone2 = VoiceMarkBackbone()
    print("OK - no import collision across repeated/interleaved construction.")