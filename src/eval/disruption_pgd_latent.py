"""
src/eval/disruption_pgd_latent.py

LAYER 1 of the new contribution: moves the joint disruption+watermark PGD
objective OUT of waveform space (disruption_pgd.py adds `delta` to `recon_wm`,
the DECODED audio) and INTO the codec's speaker-specific RVQ latent space
(this script adds `delta` to `acoustic_wm`, the PRE-DECODE speaker latent
`forward_full()` already exposes). The codec decoder then runs exactly once,
turning the modified latent into the published waveform -- there is no
separate "decode, then re-perturb the decoded audio" step.

WHY THIS MIGHT MATTER (the actual hypothesis under test, stated up front so
the framing can't drift after results come in -- see the project's own
"measure before trusting" discipline elsewhere in this repo):

AudioPure is a diffusion-based purifier -- architecturally, that class of
defense is specifically good at removing small, roughly-additive,
roughly-Gaussian waveform perturbations (that's what a denoising diffusion
process is built to do). An L-infinity waveform perturbation (what
disruption_pgd.py and SafeSpeech's own mechanism both use) is close to the
textbook case purification is designed against -- plausibly why THREE
independent attempts to fix AudioPure-survival from the detector side
(train_stage3_audiopure_robust.py, all three runs) all failed identically:
the vulnerability may not be fixable from the detector at all if the
protection itself lives in the wrong representation. A perturbation applied
instead to the pre-decode speaker latent, then decoded ONCE by the same
frozen codec that already knows how to turn watermark-bearing latents into
natural audio, produces a STRUCTURAL change to the decoded waveform --
closer to "a different speaker said this" than to "noise was added to this"
-- which a denoiser tuned to remove small Gaussian-like perturbations may not
target well. This is a hypothesis, not a promise. Both outcomes are legitimate
thesis findings -- see the "success/failure criteria" note in the project doc
`latent-space-contribution-plan-2026-09-08.md` before running the full sweep.

IMPLEMENTATION -- HOW delta REACHES THE LATENT WITHOUT EDITING THE VENDORED
external/voicemark/models.py (not committed to this repo, and not
modifiable from this environment):

backbone.py's own docstring (verified against speechtokenizer/model.py)
documents the exact internal computation:
    acoustic = e - quantized_list[0]                              # pre-watermark
    acoustic_wm = sum(msg_processor(x, message) for x in subset)  # post-watermark
    return (o, o_wm, acoustic, acoustic_wm)
i.e. `msg_processor` is called once per RVQ layer in `subset` (layers 2-8),
and the per-layer outputs are SUMMED to form `acoustic_wm`. Since summation is
linear, adding `delta` to the output of exactly ONE of those calls is
mathematically identical to adding `delta` to the total `acoustic_wm` -- so a
thin Python-level WRAPPER around `msg_processor` (added to its output on
exactly its first call within a given forward pass, a no-op on any later
calls in the same pass) gets `delta` into the latent with zero changes to any
vendored file, and gradients flow back to `delta` through the SAME decode
path Lcos/Lmel already prove is differentiable during Stage 1 training. This
is an assumption about call order/count, not a certainty (models.py's exact
internals were never independently re-verified this session, since
huggingface.co is unreachable from this environment) -- ALWAYS run
--diagnostic first, which explicitly checks that a known nonzero delta
changes `acoustic_wm` by (approximately) that same delta, before trusting a
sweep. If that check fails, the call-count assumption is wrong and this file
needs a one-line fix (e.g. add on every call divided by the number of calls,
if `subset` turns out to have more than one element contributing per pass).

WHAT'S NEW VS disruption_pgd.py, PER YOUR OWN REVIEW OF THAT SCRIPT:
  1. delta lives in latent space (acoustic_wm), not waveform space (recon_wm).
  2. epsilon is a RELATIVE L2 bound (a fraction of ||acoustic_wm||), not an
     absolute waveform-amplitude L-infinity bound -- latent magnitude scale
     is uncalibrated, so a relative bound is the only sane starting point.
     Diagnostic mode prints the raw latent norm and resulting delta norm.
  3. The PGD objective now includes THREE terms, not two: sim disruption
     (attacker's clone should have low speaker similarity -- unchanged from
     disruption_pgd.py), watermark preservation (Ldec on the perturbed,
     decoded, PUBLISHED audio -- unchanged), and -- NEW -- the pivotal
     mel-distance loss (compute_pivotal_disruption_loss), which
     disruption_pgd.py only ever MEASURED for reporting, never optimized.
     Pivotal loss operates on the CLONE's mel-spectrogram vs. the clean
     original's, so pushing it up directly targets making the ATTACKER's
     synthesis sound bad/unnatural/wrong -- a distinct objective from SIM
     (speaker-identity distance) -- without touching the PUBLISHED audio's
     own quality, which is governed by --lambda_wm and the epsilon bound as
     before. See --lambda_pivotal.
  4. --check_audiopure: optionally purifies the protected audio through
     AudioPure and reports watermark ACC after purification, in the SAME
     run. Nothing in this repo has ever measured whether disruption-PGD
     protection (waveform OR latent) survives AudioPure -- this was flagged
     explicitly in README's "Known limitations" as an untested inference,
     not a measurement. This flag closes that gap directly, for whichever
     mechanism this script is run with.

Usage (same three-phase discipline as disruption_pgd.py -- diagnostic first):
    # 1. Diagnostic: 1 batch, 1 PGD step, verifies delta reaches the latent
    #    and gradient reaches delta. No --output needed.
    python src/eval/disruption_pgd_latent.py \\
        --checkpoint ./checkpoints/stage1_final_scaleup_recalibrated/recalibrated_final.pt \\
        --diagnostic

    # 2. Small calibration sweep (n=25) across a few epsilon_rel values --
    #    same spirit as disruption_pgd.py's epsilon sweep, just in relative
    #    latent-space units instead of absolute waveform amplitude:
    python src/eval/disruption_pgd_latent.py \\
        --checkpoint ./checkpoints/stage1_final_scaleup_recalibrated/recalibrated_final.pt \\
        --epsilon_rel 0.05 --lambda_wm 1.0 --lambda_pivotal 1.0 \\
        --n_eval_speakers 5 --eval_utterances_per_speaker 5 \\
        --output results/results_latentpgd_sim_eps005_lwm1_lpiv1_n25_run1.json

    # 3. Once an operating point is chosen, run it at n=100 AND with the
    #    AudioPure check on, in one pass -- this is the measurement that
    #    actually tests Layer 1's hypothesis:
    python src/eval/disruption_pgd_latent.py \\
        --checkpoint ./checkpoints/stage1_final_scaleup_recalibrated/recalibrated_final.pt \\
        --epsilon_rel <chosen> --lambda_wm 1.0 --lambda_pivotal 1.0 \\
        --n_eval_speakers 20 --eval_utterances_per_speaker 5 \\
        --check_audiopure --repo_root . \\
        --output results/results_latentpgd_sim_eps<chosen>_lwm1_lpiv1_n100_run1.json
"""

import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "losses"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backbone import VoiceMarkBackbone
from adapters import apply_lora_adapters
from surrogate_vc import load_yourtts_surrogate
from safespeech_losses import SafeSpeechMelSpectrogram, compute_pivotal_disruption_loss, compute_sim_disruption_loss
from librispeech import LibriSpeechSubset, collate_librispeech
from vctk import VCTKSubset, collate_vctk
from libritts import LibriTTSSubset, collate_libritts
from torch.utils.data import DataLoader
from train_stage2 import compute_detection_accuracy
from voicemark_losses import compute_ldec


# ---------------------------------------------------------------------------
# The msg_processor wrapper that gets `delta` into the latent (see module
# docstring for why this works without touching the vendored models.py).
# ---------------------------------------------------------------------------

class LatentDeltaMsgProcessor:
    """
    Wraps a (frozen, possibly LoRA-adapted) msg_processor. `st_model.forward()`
    calls this once per RVQ layer in `subset`, summing the results into
    `acoustic_wm` (confirmed via backbone.py's own docstring). Adding `delta`
    on exactly the FIRST such call within a forward pass is equivalent to
    adding it once to the summed total -- call `.reset()` before every fresh
    forward pass (this script does so automatically in `latent_pgd_perturb`).

    NOT an nn.Module on purpose -- st_model.forward() only needs this to be
    callable as msg_processor(x, message); wrapping it in plain Python avoids
    any risk of double-registering the underlying module's parameters. The
    underlying module's own parameters (and gradient flow through them) are
    untouched -- this script never trains msg_processor, exactly like
    disruption_pgd.py never trains the backbone.
    """

    def __init__(self, base_msg_processor):
        self.base = base_msg_processor
        self.delta = None       # set fresh before each forward pass
        self._used = False

    def set_delta(self, delta: torch.Tensor):
        self.delta = delta
        self._used = False

    def __call__(self, x, message):
        out = self.base(x, message)
        if self.delta is not None and not self._used:
            out = out + self.delta.to(out.dtype)
            self._used = True
        return out


def build_backbone(checkpoint_path: str, r: int, alpha: int):
    """
    Same discipline as disruption_pgd.py's build_backbone: every backbone
    parameter is frozen after loading. This script never updates any weight
    -- the only tensor ever optimized is `delta` in latent_pgd_perturb().
    """
    backbone = VoiceMarkBackbone()
    apply_lora_adapters(backbone, r=r, alpha=alpha)

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        lora_state = ckpt["lora_state_dict"]
        missing, unexpected = backbone.model.load_state_dict(lora_state, strict=False)
        non_lora_missing = [k for k in missing if "_lora." in k]
        if non_lora_missing:
            print(f"[build_backbone] WARNING - {len(non_lora_missing)} LoRA keys missing: "
                  f"{non_lora_missing[:3]}...")
        print(f"[build_backbone] Loaded LoRA weights from {checkpoint_path} (epoch {ckpt.get('epoch')})")
    else:
        print("[build_backbone] No checkpoint given -- LoRA at zero-init (== pretrained VoiceMark).")

    backbone.model.eval()
    for p in backbone.model.parameters():
        p.requires_grad_(False)
    print("[build_backbone] All backbone params frozen. Only `delta` (in latent space) is ever optimized.")
    return backbone


def random_message(nbits: int, batch_size: int, device, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.randint(0, 2, (batch_size, nbits), generator=gen, device=device)


def compute_sim(surrogate, clean_audio: torch.Tensor, cloned_output: torch.Tensor) -> float:
    with torch.no_grad():
        emb_clean = surrogate.compute_speaker_embedding(clean_audio)
        emb_cloned = surrogate.compute_speaker_embedding(cloned_output)
        sim = F.cosine_similarity(emb_clean, emb_cloned, dim=-1)
    return sim.mean().item()


def detect_acc(backbone, wm_audio: torch.Tensor, message: torch.Tensor) -> float:
    with torch.no_grad():
        detect_feat = backbone.model.st_model.forward_feature(wm_audio)
        _logits, chunk_logits = backbone.model.detector(detect_feat)
    return compute_detection_accuracy(chunk_logits, message)


def latent_pgd_perturb(backbone, surrogate, mel_fn, clean_audio: torch.Tensor, message: torch.Tensor, text: str,
                        epsilon_rel: float, step_size_rel: float, n_steps: int, random_start: bool,
                        lambda_wm: float = 1.0, lambda_pivotal: float = 0.0, verbose: bool = False):
    """
    The Layer 1 mechanism. Mirrors disruption_pgd.py's pgd_perturb() closely
    (same "sign-gradient descent on a single free tensor, everything else
    frozen and detached" structure) -- the only thing that moves is WHERE
    delta is applied and WHEN decode happens relative to it.

    Step 1: one no-grad forward pass through the REAL msg_processor gets the
    baseline acoustic_wm (this is what would ship with zero protection) and
    its norm, which defines the epsilon ball (epsilon_rel * ||acoustic_wm||).
    Step 2: swap in LatentDeltaMsgProcessor, call st_model.forward() directly
    (bypassing backbone.forward_full's convenience wrapper, since we need to
    pass OUR wrapper as the msg_processor argument) with delta attached.
    Step 3: decode happens INSIDE that one call -- o_wm here already reflects
    delta's effect, with gradient flowing all the way back to delta.
    Step 4: clone the decoded (perturbed) audio through the surrogate, and
    compute all three loss terms on it / on the perturbed audio itself.

    Returns (recon_wm_unprotected, perturbed_final, delta, acoustic_wm_norm) -- all detached.
    """
    st_model = backbone.model.st_model
    real_msg_processor = backbone.model.msg_processor

    with torch.no_grad():
        _o, o_wm_baseline, _acoustic, acoustic_wm_baseline = st_model(
            clean_audio, msg_processor=real_msg_processor, message=message
        )
        recon_wm_baseline = o_wm_baseline.detach()
        latent_norm = acoustic_wm_baseline.norm(p=2, dim=tuple(range(1, acoustic_wm_baseline.dim())), keepdim=True).detach()
        epsilon = epsilon_rel * latent_norm          # per-sample L2 ball radius, absolute units
        step_size = step_size_rel * latent_norm

    wrapper = LatentDeltaMsgProcessor(real_msg_processor)

    if random_start:
        delta = torch.randn_like(acoustic_wm_baseline)
        delta = delta * (epsilon / delta.norm(p=2, dim=tuple(range(1, delta.dim())), keepdim=True).clamp_min(1e-12))
    else:
        delta = torch.zeros_like(acoustic_wm_baseline)
    delta = delta.detach().requires_grad_(True)

    for step in range(n_steps):
        wrapper.set_delta(delta)
        wav_len = clean_audio.size(-1)

        # FIX (2026-09-09): unlike disruption_pgd.py -- which only ever needs
        # backward through the DETECTOR, since delta is added AFTER decode
        # there -- this script needs backward through st_model's own
        # forward() too, because delta now enters BEFORE decode, and
        # st_model contains an RNN internally (the same one disruption_pgd.py's
        # detector call already had to work around). cuDNN's fused RNN kernel
        # refuses backward in eval mode, and the flag has to be set at
        # FORWARD time (when the graph is built), not just when .grad() is
        # later called -- so the whole per-step forward (st_model call,
        # surrogate clone, detector call) now lives inside this context, not
        # just the detector piece. First run's traceback confirmed this: the
        # call-count diagnostic (a no_grad, forward-only call) passed fine;
        # only the backward-requiring step failed.
        with torch.backends.cudnn.flags(enabled=False):
            o, o_wm, acoustic, acoustic_wm = st_model(
                clean_audio, msg_processor=wrapper, message=message
            )
            perturbed = o_wm[..., :wav_len] if o_wm.size(-1) >= wav_len else o_wm

            cloned_output = surrogate.clone_voice(perturbed, text=text)
            emb_clean = surrogate.compute_speaker_embedding(clean_audio)
            emb_cloned = surrogate.compute_speaker_embedding(cloned_output)
            sim_loss = compute_sim_disruption_loss(emb_clean.detach(), emb_cloned)

            pivotal_loss = compute_pivotal_disruption_loss(mel_fn, clean_audio, cloned_output)
            # We want to MAXIMIZE pivotal distance (attacker's clone should sound
            # wrong) but the PGD step below MINIMIZES the combined objective --
            # negate here, matching safespeech_losses.py's own sign convention.
            pivotal_term = -pivotal_loss

            detect_feat = st_model.forward_feature(perturbed)
            _logits, chunk_logits = backbone.model.detector(detect_feat)
            wm_loss = compute_ldec(chunk_logits, message)

            grad_sim = torch.autograd.grad(sim_loss, delta, retain_graph=True, create_graph=False)[0]
            grad_piv = torch.autograd.grad(pivotal_term, delta, retain_graph=True, create_graph=False)[0] \
                if lambda_pivotal != 0.0 else torch.zeros_like(delta)
            grad_wm = torch.autograd.grad(wm_loss, delta, retain_graph=False, create_graph=False)[0]
        grad = grad_sim + lambda_pivotal * grad_piv + lambda_wm * grad_wm
        grad_norm = grad.norm().item()

        if verbose:
            print(f"  [latent_pgd step {step}] sim_loss={sim_loss.item():.4f} "
                  f"pivotal_loss={pivotal_loss.item():.4f} wm_loss={wm_loss.item():.4f} "
                  f"|grad_sim|={grad_sim.norm().item():.4e} |grad_piv|={grad_piv.norm().item():.4e} "
                  f"|grad_wm|={grad_wm.norm().item():.4e} |grad_combined|={grad_norm:.4e} "
                  f"delta_l2={delta.norm().item():.6f} eps_l2(sample0)={epsilon.flatten()[0].item():.6f}")
        if step == 0 and grad_norm == 0.0:
            print("  [latent_pgd_perturb] WARNING: zero gradient reached delta at step 0 -- "
                  "either the wrapper isn't actually in the decode path (call-count assumption "
                  "wrong, see module docstring) or epsilon_rel is too small. Do not trust a full "
                  "sweep until this is nonzero.")

        with torch.no_grad():
            delta = delta - step_size * (grad / grad.norm(p=2, dim=tuple(range(1, grad.dim())), keepdim=True).clamp_min(1e-12))
            delta_norm = delta.norm(p=2, dim=tuple(range(1, delta.dim())), keepdim=True)
            over = (delta_norm > epsilon).float()
            delta = delta * (1 - over) + delta * (epsilon / delta_norm.clamp_min(1e-12)) * over
        delta = delta.detach().requires_grad_(True)

    with torch.no_grad():
        wrapper.set_delta(delta)
        _o, o_wm_final, _acoustic, _acoustic_wm_final = st_model(
            clean_audio, msg_processor=wrapper, message=message
        )
        wav_len = clean_audio.size(-1)
        perturbed_final = o_wm_final[..., :wav_len] if o_wm_final.size(-1) >= wav_len else o_wm_final

    return recon_wm_baseline, perturbed_final.detach(), delta.detach(), latent_norm.detach()


def run_latent_pgd_eval(backbone, surrogate, mel_fn, eval_loader, device, text: str,
                         epsilon_rel: float, step_size_rel: float, n_steps: int, random_start: bool,
                         lambda_wm: float, lambda_pivotal: float, seed: int = 123) -> dict:
    metrics = {k: [] for k in [
        "sim_before", "sim_after", "pivotal_before", "pivotal_after",
        "detection_acc_before", "detection_acc_after",
        "delta_l2", "latent_l2", "delta_relative_norm",
    ]}

    for batch_idx, batch in enumerate(eval_loader):
        clean_audio = batch["waveform"].to(device)
        message = random_message(16, clean_audio.shape[0], device, seed=seed + batch_idx)

        recon_wm, perturbed_final, delta, latent_norm = latent_pgd_perturb(
            backbone, surrogate, mel_fn, clean_audio, message, text,
            epsilon_rel, step_size_rel, n_steps, random_start,
            lambda_wm=lambda_wm, lambda_pivotal=lambda_pivotal,
        )

        with torch.no_grad():
            cloned_before = surrogate.clone_voice(recon_wm, text=text)
            cloned_after = surrogate.clone_voice(perturbed_final, text=text)

        metrics["sim_before"].append(compute_sim(surrogate, clean_audio, cloned_before))
        metrics["sim_after"].append(compute_sim(surrogate, clean_audio, cloned_after))
        metrics["pivotal_before"].append(compute_pivotal_disruption_loss(mel_fn, clean_audio, cloned_before).item())
        metrics["pivotal_after"].append(compute_pivotal_disruption_loss(mel_fn, clean_audio, cloned_after).item())
        metrics["detection_acc_before"].append(detect_acc(backbone, recon_wm, message))
        metrics["detection_acc_after"].append(detect_acc(backbone, perturbed_final, message))
        metrics["delta_l2"].append(delta.norm().item())
        metrics["latent_l2"].append(latent_norm.mean().item())
        metrics["delta_relative_norm"].append((delta.norm() / latent_norm.mean().clamp_min(1e-12)).item())

        print(f"[batch {batch_idx}] sim {metrics['sim_before'][-1]:.4f} -> {metrics['sim_after'][-1]:.4f} | "
              f"pivotal {metrics['pivotal_before'][-1]:.4f} -> {metrics['pivotal_after'][-1]:.4f} | "
              f"acc {metrics['detection_acc_before'][-1]:.4f} -> {metrics['detection_acc_after'][-1]:.4f} | "
              f"delta_rel_norm={metrics['delta_relative_norm'][-1]:.4f}")

    return {k: sum(v) / len(v) for k, v in metrics.items()}, metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)

    p.add_argument("--epsilon_rel", type=float, default=0.05,
                    help="L2 perturbation budget as a FRACTION of ||acoustic_wm||, per sample. "
                         "Latent magnitude is uncalibrated -- start small (0.02-0.1) and sweep, "
                         "same discipline as disruption_pgd.py's absolute epsilon sweep.")
    p.add_argument("--step_size_rel", type=float, default=None,
                    help="Defaults to epsilon_rel/4 (same heuristic as disruption_pgd.py).")
    p.add_argument("--n_steps", type=int, default=10)
    p.add_argument("--random_start", action="store_true", default=True)
    p.add_argument("--no_random_start", dest="random_start", action="store_false")
    p.add_argument("--lambda_wm", type=float, default=1.0,
                    help="Weight on watermark-preservation (Ldec on perturbed/published audio). "
                         "disruption_pgd.py's own sweep found 1.0 reaches the ACC ceiling with no "
                         "measurable disruption cost -- same default here, RE-VERIFY for latent space.")
    p.add_argument("--lambda_pivotal", type=float, default=1.0,
                    help="NEW vs. disruption_pgd.py: weight on the pivotal mel-distance term, "
                         "computed on the ATTACKER'S CLONE (not the published audio). This is the "
                         "'make the attacker's synthesis sound wrong, not just differently-voiced' "
                         "term disruption_pgd.py only ever measured, never optimized. Set to 0.0 to "
                         "reproduce a pure SIM+watermark objective (closest analogue to disruption_pgd.py).")

    p.add_argument("--diagnostic", action="store_true",
                    help="1 batch, 1 step, verbose. ALSO explicitly checks the call-count assumption "
                         "(see module docstring) by comparing acoustic_wm with delta=0 vs a known "
                         "nonzero test delta. Run this FIRST, always.")
    p.add_argument("--save_samples_dir", type=str, default=None)
    p.add_argument("--n_samples", type=int, default=4)

    p.add_argument("--check_audiopure", action="store_true",
                    help="Also purify the final protected audio through AudioPure's DiffWave "
                         "denoiser and report watermark ACC after purification, in this same run. "
                         "Closes the 'never actually tested' gap flagged in README's Known "
                         "limitations for waveform-space PGD too -- use --repo_root to point at the "
                         "checkout root if not running from repo root.")
    p.add_argument("--repo_root", type=str, default=".")
    p.add_argument("--reverse_timestep", type=int, default=25)

    p.add_argument("--output", type=str, default=None)
    p.add_argument("--dataset", type=str, default="librispeech", choices=["librispeech", "vctk", "libritts"])
    p.add_argument("--vctk_root", type=str,
                    default="/kaggle/input/datasets/pratt3000/vctk-corpus/VCTK-Corpus/VCTK-Corpus")
    p.add_argument("--libritts_root", type=str,
                    default="/kaggle/input/datasets/pratt3000/libritts/LibriTTS")
    p.add_argument("--data_root", type=str, default="./data/librispeech")
    p.add_argument("--n_speakers", type=int, default=10)
    p.add_argument("--utterances_per_speaker", type=int, default=10)
    p.add_argument("--n_eval_speakers", type=int, default=5)
    p.add_argument("--eval_utterances_per_speaker", type=int, default=5)
    p.add_argument("--crop_seconds", type=float, default=3.0)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--surrogate_sample_rate", type=int, default=16000)
    p.add_argument("--surrogate_text", type=str, default="This is a test sentence for voice cloning.")
    args = p.parse_args()

    if not args.diagnostic and not args.save_samples_dir and args.output is None:
        p.error("--output is required unless --diagnostic or --save_samples_dir is set")

    step_size_rel = args.step_size_rel if args.step_size_rel is not None else args.epsilon_rel / 4.0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset == "librispeech":
        eval_ds = LibriSpeechSubset(
            root=args.data_root, n_speakers=args.n_speakers, utterances_per_speaker=args.utterances_per_speaker,
            n_eval_speakers=args.n_eval_speakers, eval_utterances_per_speaker=args.eval_utterances_per_speaker,
            sample_rate=16000, crop_seconds=args.crop_seconds, split="eval",
        )
        collate_fn = collate_librispeech
    elif args.dataset == "vctk":
        eval_ds = VCTKSubset(
            root=args.vctk_root, n_speakers=args.n_speakers, utterances_per_speaker=args.utterances_per_speaker,
            n_eval_speakers=args.n_eval_speakers, eval_utterances_per_speaker=args.eval_utterances_per_speaker,
            sample_rate=16000, crop_seconds=args.crop_seconds, split="eval",
        )
        collate_fn = collate_vctk
    else:
        eval_ds = LibriTTSSubset(
            root=args.libritts_root, n_speakers=args.n_speakers, utterances_per_speaker=args.utterances_per_speaker,
            n_eval_speakers=args.n_eval_speakers, eval_utterances_per_speaker=args.eval_utterances_per_speaker,
            sample_rate=16000, crop_seconds=args.crop_seconds, split="eval",
        )
        collate_fn = collate_libritts

    print(f"[main] LATENT-SPACE PGD | dataset={args.dataset} epsilon_rel={args.epsilon_rel} "
          f"step_size_rel={step_size_rel} n_steps={args.n_steps} lambda_wm={args.lambda_wm} "
          f"lambda_pivotal={args.lambda_pivotal}")
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)

    backbone = build_backbone(args.checkpoint, args.lora_r, args.lora_alpha)
    print("[main] Loading YourTTS surrogate (frozen)...")
    surrogate = load_yourtts_surrogate(device=device)
    mel_fn = SafeSpeechMelSpectrogram(sampling_rate=args.surrogate_sample_rate).to(device)

    if args.diagnostic:
        print("\n" + "=" * 60 + "\nDIAGNOSTIC MODE: 1 batch, 1 latent-PGD step\n" + "=" * 60)
        batch = next(iter(eval_loader))
        clean_audio = batch["waveform"].to(device)
        message = random_message(16, clean_audio.shape[0], device, seed=123)

        # Call-count sanity check FIRST, before the real PGD step: with a
        # known nonzero test delta, does acoustic_wm actually change by
        # (approximately) that delta? This is the load-bearing assumption
        # the whole mechanism depends on -- see module docstring.
        print("\n--- Call-count assumption check ---")
        st_model = backbone.model.st_model
        real_mp = backbone.model.msg_processor
        with torch.no_grad():
            _o, _owm, _ac, acoustic_wm_zero = st_model(clean_audio, msg_processor=real_mp, message=message)
            test_wrapper = LatentDeltaMsgProcessor(real_mp)
            test_delta = torch.ones_like(acoustic_wm_zero) * 0.01 * acoustic_wm_zero.norm().item() / acoustic_wm_zero.numel() ** 0.5
            test_wrapper.set_delta(test_delta)
            _o, _owm, _ac, acoustic_wm_test = st_model(clean_audio, msg_processor=test_wrapper, message=message)
            observed_diff = (acoustic_wm_test - acoustic_wm_zero)
            expected_diff = test_delta
            rel_err = (observed_diff - expected_diff).norm().item() / expected_diff.norm().clamp_min(1e-12).item()
        print(f"  ||observed_diff - expected_delta|| / ||expected_delta|| = {rel_err:.4f} "
              f"(should be near 0.0 -- e.g. < 0.05 -- if the 'add once, summed linearly' "
              f"assumption holds; if this is large, msg_processor is called more than once "
              f"per forward pass in a way that isn't a simple sum, or something else in the "
              f"decode path is nonlinear w.r.t. this term -- STOP and inspect before sweeping)")

        print("\n--- Real PGD step (verbose) ---")
        recon_wm, perturbed, delta, latent_norm = latent_pgd_perturb(
            backbone, surrogate, mel_fn, clean_audio, message, args.surrogate_text,
            args.epsilon_rel, step_size_rel, n_steps=1, random_start=args.random_start,
            lambda_wm=args.lambda_wm, lambda_pivotal=args.lambda_pivotal, verbose=True,
        )
        acc_before = detect_acc(backbone, recon_wm, message)
        acc_after = detect_acc(backbone, perturbed, message)
        print(f"\ndetection_acc: {acc_before:.4f} -> {acc_after:.4f} (1 step -- expect small movement)")
        print(f"latent_norm(sample0)={latent_norm.flatten()[0].item():.4f}, "
              f"delta_l2(sample0)={delta[0].norm().item():.6f}")
        print("[main] Diagnostic complete. If the call-count check above was near 0.0 and gradient "
              "was nonzero, proceed to a small sweep (drop --diagnostic, add --output, small n).")
        return

    if args.save_samples_dir:
        os.makedirs(args.save_samples_dir, exist_ok=True)
        sample_loader = DataLoader(eval_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
        print(f"\nSaving {args.n_samples} samples to {args.save_samples_dir}/ "
              f"(epsilon_rel={args.epsilon_rel}, lambda_wm={args.lambda_wm}, lambda_pivotal={args.lambda_pivotal})")
        for i, batch in enumerate(sample_loader):
            if i >= args.n_samples:
                break
            clean_audio = batch["waveform"].to(device)
            message = random_message(16, clean_audio.shape[0], device, seed=123 + i)
            recon_wm, perturbed_final, delta, _ = latent_pgd_perturb(
                backbone, surrogate, mel_fn, clean_audio, message, args.surrogate_text,
                args.epsilon_rel, step_size_rel, args.n_steps, args.random_start,
                lambda_wm=args.lambda_wm, lambda_pivotal=args.lambda_pivotal,
            )
            with torch.no_grad():
                cloned_after = surrogate.clone_voice(perturbed_final, text=args.surrogate_text)
            acc_before = detect_acc(backbone, recon_wm, message)
            acc_after = detect_acc(backbone, perturbed_final, message)

            def save_wav(path, waveform, sample_rate=16000):
                sf.write(path, waveform.detach().cpu().squeeze().numpy(), sample_rate)

            save_wav(os.path.join(args.save_samples_dir, f"sample{i}_clean.wav"), clean_audio[0])
            save_wav(os.path.join(args.save_samples_dir, f"sample{i}_watermarked.wav"), perturbed_final[0])
            save_wav(os.path.join(args.save_samples_dir, f"sample{i}_cloned.wav"), cloned_after[0])
            print(f"  sample{i}: acc {acc_before:.4f} -> {acc_after:.4f}, delta_l2={delta.norm().item():.5f}")
        print(f"\n[main] Done. Next: python src/eval/quality_metrics.py --sample_dir {args.save_samples_dir} "
              f"--n_samples {args.n_samples} --skip_wer   # PESQ/STOI on the PUBLISHED audio")
        print(f"Then: python src/eval/audio_diff_analysis.py --sample_dir {args.save_samples_dir} --sample_idx 0")
        return

    label = f"latentpgd_epsrel{args.epsilon_rel}_n{args.n_steps}_lwm{args.lambda_wm}_lpiv{args.lambda_pivotal}"
    print(f"\n{'=' * 60}\nEvaluating: {label}\n{'=' * 60}")

    means, _all = run_latent_pgd_eval(
        backbone, surrogate, mel_fn, eval_loader, device, args.surrogate_text,
        args.epsilon_rel, step_size_rel, args.n_steps, args.random_start,
        lambda_wm=args.lambda_wm, lambda_pivotal=args.lambda_pivotal,
    )

    print(f"\nSIM: {means['sim_before']:.4f} -> {means['sim_after']:.4f} "
          f"(delta={means['sim_before'] - means['sim_after']:+.4f}, positive = more disrupted)")
    print(f"Pivotal (attacker's clone vs. clean): {means['pivotal_before']:.4f} -> {means['pivotal_after']:.4f} "
          f"(positive change = clone sounds MORE wrong, which is what we want)")
    print(f"Detection ACC: {means['detection_acc_before']:.4f} -> {means['detection_acc_after']:.4f}")
    print(f"Mean delta/latent relative L2 norm: {means['delta_relative_norm']:.4f} "
          f"(should track close to --epsilon_rel={args.epsilon_rel})")

    # SAVE THE DISRUPTION RESULTS NOW, before touching AudioPure at all. This
    # is the fix for the 2026-09-09 incident: a crash loading the AudioPure
    # denoiser (corrupt/truncated checkpoint) took the whole n=100 run down
    # with it, including the SIM/pivotal/ACC numbers that had already
    # finished successfully. Writing here means those numbers are safe on
    # disk regardless of what happens next; the AudioPure block below only
    # ever ADDS to this file (via the re-open-and-update at the end), never
    # gates it.
    base_results = {
        "sim_before": means["sim_before"], "sim_after": means["sim_after"],
        "sim_drop": means["sim_before"] - means["sim_after"],
        "pivotal_before": means["pivotal_before"], "pivotal_after": means["pivotal_after"],
        "detection_acc_before": means["detection_acc_before"],
        "detection_acc_after": means["detection_acc_after"],
        "detection_acc_drop": means["detection_acc_before"] - means["detection_acc_after"],
        "delta_relative_norm_mean": means["delta_relative_norm"],
        "n_trials": len(eval_ds),
        "sim": means["sim_after"],
        "audiopure_acc_after_mean": None,
    }
    if args.output:
        out = {
            "label": label, "checkpoint": args.checkpoint, "dataset": args.dataset,
            "mechanism": "latent_space_pgd",
            "pgd": {"epsilon_rel": args.epsilon_rel, "step_size_rel": step_size_rel, "n_steps": args.n_steps,
                     "random_start": args.random_start, "lambda_wm": args.lambda_wm, "lambda_pivotal": args.lambda_pivotal},
            "results": base_results,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[main] Saved disruption results to {args.output} (audiopure_acc_after_mean pending, "
              f"will be added below if --check_audiopure succeeds)")

    audiopure_result = None
    if args.check_audiopure:
        print(f"\n{'=' * 60}\nAudioPure survival check (NEW -- never measured before for PGD-protected audio)\n{'=' * 60}")
        try:
            sys.path.insert(0, os.path.dirname(__file__))
            from audiopure_eval import build_audiopure_denoiser
            denoiser = build_audiopure_denoiser(args.repo_root, reverse_timestep=args.reverse_timestep)
            acc_after_purify = []
            for batch_idx, batch in enumerate(eval_loader):
                clean_audio = batch["waveform"].to(device)
                message = random_message(16, clean_audio.shape[0], device, seed=123 + batch_idx)
                # FIX (2026-09-09): latent_pgd_perturb() must NOT be called
                # inside torch.no_grad() -- confirmed by an actual n=100 run
                # (epsilon_rel=0.10): the disruption pass itself (unwrapped,
                # same as here) completed cleanly for all 100 utterances, but
                # this block crashed immediately with "element 0 of tensors
                # does not require grad and does not have a grad_fn" the
                # moment it reached AudioPure, because PGD runs its own steps
                # internally via torch.autograd.grad(), which needs a live
                # grad_fn -- the outer no_grad() strips that from every
                # tensor before latent_pgd_perturb() ever gets to use it.
                # Only the purify+detect step below (no further optimization,
                # detached from the PGD graph) needs no_grad.
                _rw, perturbed_final, _d, _ln = latent_pgd_perturb(
                    backbone, surrogate, mel_fn, clean_audio, message, args.surrogate_text,
                    args.epsilon_rel, step_size_rel, args.n_steps, args.random_start,
                    lambda_wm=args.lambda_wm, lambda_pivotal=args.lambda_pivotal,
                )
                with torch.no_grad():
                    purified = denoiser(perturbed_final.detach())
                    acc = detect_acc(backbone, purified, message)
                acc_after_purify.append(acc)
                print(f"  [audiopure batch {batch_idx}] watermark ACC after purification of protected audio: {acc:.4f}")
            audiopure_result = sum(acc_after_purify) / len(acc_after_purify)
            print(f"\nMean watermark ACC after AudioPure purification of LATENT-PROTECTED audio: {audiopure_result:.4f} "
                  f"(compare against ~0.49-0.53 for unprotected/waveform-PGD-protected audio -- see README Section 4)")
        except Exception as e:
            print(f"\n[main] WARNING - AudioPure check FAILED ({type(e).__name__}: {e}). "
                  f"Disruption results above are already saved to {args.output} regardless "
                  f"(audiopure_acc_after_mean stays null in that file) -- fix whatever broke "
                  f"and re-run with ONLY --check_audiopure's underlying issue addressed; the "
                  f"expensive n={len(eval_ds)} disruption pass does not need to be repeated.")

    if args.output and audiopure_result is not None:
        # Re-open and update just the one field, rather than re-deriving
        # everything -- keeps this block a pure "add audiopure result if we
        # got one" step with no risk of drifting from what was already saved.
        with open(args.output) as f:
            out = json.load(f)
        out["results"]["audiopure_acc_after_mean"] = audiopure_result
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[main] Updated {args.output} with audiopure_acc_after_mean={audiopure_result:.4f}")


if __name__ == "__main__":
    main()
