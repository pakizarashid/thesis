"""
src/eval/disruption_pgd.py

The PGD-hybrid disruption mechanism -- the untested "genuinely different
mechanism" approved to replace/extend embedder-weight training after six
converging negative results (lambda scale, loss re-weighting, LoRA capacity
4x on attention, training duration, mel-mode vs sim-mode objective, and FFN
capacity extension -- see STAGE2_WRITEUP.md Sections 4-10). All six of those
share ONE structural property: they all train a small number of SHARED
weights (a LoRA adapter, at most 786,432 params for the FFN-capacity variant)
via standard Adam to produce a single embedder that must generalize its
disruption effect across every input utterance at once. SafeSpeech's own real
mechanism (external/safespeech/protect.py) is structurally different: it runs
epsilon-bounded PGD (sign-gradient descent) DIRECTLY on the waveform, fresh
for every utterance, with no shared weights and no generalization requirement
at all -- every utterance gets its own bespoke perturbation, solved to
convergence independently. That is a categorically larger optimization
surface (T waveform samples with box constraints per utterance) than 786K
shared parameters that have to work for every utterance simultaneously. This
script tests that distinction directly rather than assuming it.

MECHANISM: after the (frozen) embedder produces recon_wm, add a small
epsilon-ball-bounded additive perturbation `delta` directly to the waveform,
solved via iterative PGD against compute_sim_disruption_loss for THAT batch
specifically. No weights are trained or saved -- `delta` (or equivalently
recon_wm + delta) is the artifact, recomputed fresh every time this script
runs, at both "protect" time and (if you were to deploy this) inference time.
This is closer to how adversarial examples are actually generated than
anything tested so far in this project.

WHAT THIS DOES NOT CHANGE: the watermark itself. recon_wm (produced by
msg_processor, whatever checkpoint you point --checkpoint at) is untouched --
delta is added ON TOP of it. Because delta also perturbs the exact signal
Ldec/the detector reads, detection ACC is NOT guaranteed to survive just
because nothing was "trained" -- this script measures detection_acc_before
vs detection_acc_after explicitly, every run, so dual-defense survival is
verified rather than assumed.

CONFIRMED FINDING (first full run, sim-only, lambda_wm=0): SIM dropped
0.4578 -> 0.1963 at epsilon=0.01 -- a categorically larger effect than any
of the six embedder-weight-training attempts (which plateaued around
0.45-0.47, i.e. no measurable effect at all). This validates the mechanism-
mismatch diagnosis directly: unconstrained per-utterance waveform-space
optimization DOES disrupt cloning, where shared-weight training did not.
Stacking PGD on top of the best embedder-trained checkpoint (stage2_capacity_ffn)
produced statistically indistinguishable numbers (SIM 0.4583 -> 0.1861) --
the embedder training track contributes nothing once PGD is applied; it's
subsumed, not complementary. BUT sim-only PGD also dragged detection ACC from
~99.5% to ~59% (near chance for a per-bit metric) -- it disrupts the exact
signal the detector reads with no incentive not to. --lambda_wm (see below)
adds a second term to the PGD objective specifically to fix this: the actual
open question now is the SIM-vs-ACC trade-off curve as lambda_wm increases
from 0, not whether PGD disrupts (already confirmed it does).

CAN BE STACKED: --checkpoint accepts ANY existing checkpoint (stage1 baseline,
a plain stage2 sim-mode checkpoint, or a stage2_capacity_ffn checkpoint via
--include_ffn). This gives you two cheap ablations for free: (1) PGD alone on
top of the stage1 baseline -- does the new mechanism work without any
embedder-level disruption training at all; (2) PGD stacked on top of your best
existing stage2 checkpoint -- does it add anything beyond what's already
there. Run both; they answer different questions.

DISCIPLINE: same "measure before trusting" pattern as gradient_diagnostic.py
before train_stage2_capacity.py. Before running the full --n_steps sweep over
your whole eval set, run with --diagnostic first: 1 batch, 1 PGD step, prints
whether gradient actually reaches `delta` and by how much. If it's ~0, the
mechanism is broken (e.g. epsilon so small the clamp kills it, or an autograd
break) and a full sweep would just burn GPU hours confirming what the
diagnostic already told you in 30 seconds.

Usage:
    # 1. Cheap sanity check first (no --output needed, exits immediately):
    python src/eval/disruption_pgd.py \\
        --checkpoint ./checkpoints/stage1_full_recalibrated_v3/recalibrated_final.pt \\
        --diagnostic

    # 2. Full run once the diagnostic looks sane. NOTE: filename contains "sim" and
    #    not "audioseal"/"audiopure"/"fpr"/"quality" on purpose -- aggregate_results.py's
    #    _classify() pattern-matches on the FILENAME to route rows into its
    #    "Disruption (SIM)" table section; keep that substring in whatever you name these.
    python src/eval/disruption_pgd.py \\
        --checkpoint ./checkpoints/stage1_full_recalibrated_v3/recalibrated_final.pt \\
        --epsilon 0.01 --n_steps 10 \\
        --output results/results_pgd_sim_stage1base_eps01_run1.json

    # 3. Stacked ablation, on top of the best existing stage2 capacity checkpoint:
    python src/eval/disruption_pgd.py \\
        --checkpoint ./checkpoints/stage2_capacity_ffn/stage2_capacity_epoch29.pt \\
        --include_ffn --capacity_lora_r 32 \\
        --epsilon 0.01 --n_steps 10 \\
        --output results/results_pgd_sim_stacked_capacity_eps01_run1.json
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


def build_backbone(checkpoint_path: str, r: int, alpha: int, include_ffn: bool, capacity_lora_r: int):
    """
    Mirrors disruption_effectiveness_capacity.py's build_backbone, generalized
    to handle BOTH plain (stage1 / plain stage2) and capacity (stage2_capacity_ffn)
    checkpoints via one --include_ffn flag, since this script's whole point is
    to be runnable against either kind of base checkpoint for the two ablations
    described in the module docstring.

    Every backbone parameter (base weights AND any LoRA deltas) is explicitly
    set to requires_grad=False after loading -- this script NEVER updates any
    weight. The only tensor PGD ever optimizes is `delta` in pgd_perturb().
    This is the actual mechanism shift the "mechanism mismatch" diagnosis
    called for: moving the entire disruption signal off of shared weights
    (however many, however capacious) and onto a fresh, unconstrained,
    per-utterance tensor in waveform space.
    """
    backbone = VoiceMarkBackbone()
    if include_ffn:
        apply_lora_adapters(
            backbone, r=r, alpha=alpha, targets=("msg_processor", "detector"),
            include_ffn=True, ffn_r=capacity_lora_r, ffn_targets=("msg_processor",),
        )
    else:
        apply_lora_adapters(backbone, r=r, alpha=alpha)

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        lora_state = ckpt["lora_state_dict"]
        missing, unexpected = backbone.model.load_state_dict(lora_state, strict=False)
        non_lora_missing = [k for k in missing if "_lora." in k]
        if non_lora_missing:
            print(f"[build_backbone] WARNING - {len(non_lora_missing)} LoRA keys missing: "
                  f"{non_lora_missing[:3]}... (checkpoint/--include_ffn/--capacity_lora_r mismatch?)")
        avg_acc = ckpt.get("avg_acc")
        avg_acc_str = f"{avg_acc:.4f}" if avg_acc is not None else "N/A"
        print(f"[build_backbone] Loaded LoRA weights from {checkpoint_path} "
              f"(epoch {ckpt.get('epoch')}, train-time avg_acc={avg_acc_str}, include_ffn={include_ffn})")
    else:
        print("[build_backbone] No checkpoint given -- LoRA at zero-init (== pretrained VoiceMark). "
              "This isolates the PGD mechanism completely from any embedder-level training.")

    backbone.model.eval()
    n_before = sum(p.numel() for p in backbone.model.parameters() if p.requires_grad)
    for p in backbone.model.parameters():
        p.requires_grad_(False)
    print(f"[build_backbone] Froze all {n_before:,} previously-trainable backbone params "
          f"(base weights were already frozen). Backbone has 0 trainable params for this "
          f"script -- PGD only ever optimizes the per-utterance `delta` tensor below.")

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


def pgd_perturb(backbone, surrogate, clean_audio: torch.Tensor, message: torch.Tensor, text: str,
                 epsilon: float, step_size: float, n_steps: int, random_start: bool,
                 lambda_wm: float = 0.0, verbose: bool = False):
    """
    The actual mechanism. recon_wm is computed once (frozen backbone, no_grad)
    and detached -- it is the fixed base every step perturbs. `delta` is the
    ONLY tensor with requires_grad=True anywhere in this function. Each step:
    add delta to recon_wm, clamp to valid waveform range, clone through the
    frozen surrogate, compute sim_disruption_loss against the (fixed, also
    detached) clean-audio embedding, take one sign-gradient DESCENT step on
    delta (we want to MINIMIZE sim_loss directly -- lower cosine similarity =
    more disruption -- so this is a minimization PGD, not the more commonly
    described maximization/attack-loss PGD; SafeSpeech's own protect.py is
    also a minimization PGD against its retargeted pivotal loss, for the same
    reason), then project back onto the epsilon L-infinity ball.

    lambda_wm > 0 adds a SECOND term to the per-step objective: compute_ldec
    (the exact same cross-entropy the detector is trained with) evaluated on
    the PERTURBED audio against `message`, minimized jointly with sim_loss.
    This is the fix for the dual-defense collapse the first full run
    surfaced: sim-only PGD (lambda_wm=0, the original behavior) disrupts SIM
    hard but with no incentive to avoid the exact signal the detector reads,
    it also drags detection ACC down toward chance. Reusing compute_ldec
    (rather than inventing a new proxy) keeps this consistent with what the
    detector was actually trained against -- same discipline as reusing
    compute_sim_disruption_loss/compute_pivotal_disruption_loss verbatim
    elsewhere in this project rather than approximating them.

    Gradient norms for BOTH terms at delta are computed and (if verbose)
    printed SEPARATELY every step -- same "measure before trusting" practice
    used to catch the original 1000x kl_to_noise/pivotal imbalance in
    gradient_diagnostic.py. Pick lambda_wm from the ratio you observe in
    --diagnostic mode, don't guess it.

    Returns (recon_wm, perturbed_final, delta) -- all detached.
    """
    with torch.no_grad():
        out = backbone.forward_full(clean_audio, message)
        recon_wm = out["recon_wm"].detach()
        emb_clean = surrogate.compute_speaker_embedding(clean_audio).detach()

    if random_start:
        delta = (torch.rand_like(recon_wm) * 2 - 1) * epsilon
    else:
        delta = torch.zeros_like(recon_wm)
    delta = delta.detach().requires_grad_(True)

    for step in range(n_steps):
        perturbed = torch.clamp(recon_wm + delta, -1.0, 1.0)

        cloned_output = surrogate.clone_voice(perturbed, text=text)
        emb_cloned = surrogate.compute_speaker_embedding(cloned_output)
        sim_loss = compute_sim_disruption_loss(emb_clean, emb_cloned)

        # backbone.model is in eval() mode (build_backbone calls .eval() -- correct
        # for a frozen inference pipeline). st_model/detector contain an RNN
        # somewhere internally (see the earlier weight_norm deprecation warning at
        # construction -- SpeechTokenizer-family codecs commonly use one), and
        # cuDNN's fused RNN kernel refuses to run backward when the module is in
        # eval mode ("cudnn RNN backward can only be called in training mode") --
        # this is a documented cuDNN/PyTorch restriction, not a bug in this model.
        # Every OTHER place this exact call sequence runs with gradients enabled
        # (train_stage2.py/train_stage2_capacity.py's training loops) always has
        # backbone.model.train() active, which is why they never hit this. We are
        # NOT willing to flip the whole backbone to train() here even briefly --
        # if anything in st_model/detector tracks running stats (BatchNorm) that
        # would silently corrupt them for every subsequent eval call in this
        # process. Instead, disable cuDNN's RNN fast path for just this forward
        # pass, forcing PyTorch's native (non-fused) RNN implementation, which
        # supports backward regardless of train/eval mode and has NO side effects
        # on any other module's state -- only this one forward+backward is slower.
        with torch.backends.cudnn.flags(enabled=False):
            detect_feat = backbone.model.st_model.forward_feature(perturbed)
            _logits, chunk_logits = backbone.model.detector(detect_feat)
            wm_loss = compute_ldec(chunk_logits, message)

        grad_sim = torch.autograd.grad(sim_loss, delta, retain_graph=True, create_graph=False)[0]
        grad_wm = torch.autograd.grad(wm_loss, delta, retain_graph=False, create_graph=False)[0]
        grad = grad_sim + lambda_wm * grad_wm
        grad_norm = grad.norm().item()

        if verbose:
            ratio = grad_sim.norm().item() / max(grad_wm.norm().item(), 1e-12)
            print(f"  [pgd step {step}] sim_loss={sim_loss.item():.4f} wm_loss={wm_loss.item():.4f} "
                  f"|grad_sim|={grad_sim.norm().item():.4e} |grad_wm|={grad_wm.norm().item():.4e} "
                  f"(sim:wm raw ratio={ratio:.2f}, i.e. lambda_wm~={ratio:.2f} would roughly balance them) "
                  f"|grad_combined|={grad_norm:.4e} delta_linf={delta.abs().max().item():.6f}")
        if step == 0 and grad_norm == 0.0:
            print(f"  [pgd_perturb] WARNING: zero gradient reached delta at step 0 -- "
                  f"either the losses are saturated at this epsilon/init, or the autograd "
                  f"path from perturbed-audio through the surrogate/detector is broken. Do not "
                  f"trust a full sweep until this is nonzero (see surrogate_vc.py's own "
                  f"__main__ smoke test, which confirms the path CAN carry gradient).")

        with torch.no_grad():
            delta = delta - step_size * grad.sign()
            delta = torch.clamp(delta, -epsilon, epsilon)
        delta = delta.detach().requires_grad_(True)

    with torch.no_grad():
        perturbed_final = torch.clamp(recon_wm + delta, -1.0, 1.0)

    return recon_wm, perturbed_final, delta.detach()


def run_pgd_eval(backbone, surrogate, eval_loader, device, mel_fn, text: str,
                  epsilon: float, step_size: float, n_steps: int, random_start: bool,
                  lambda_wm: float = 0.0, seed: int = 123) -> dict:
    metrics = {k: [] for k in [
        "sim_before", "sim_after", "pivotal_before", "pivotal_after",
        "detection_acc_before", "detection_acc_after",
        "perturbation_linf", "perturbation_snr_db",
    ]}

    for batch_idx, batch in enumerate(eval_loader):
        clean_audio = batch["waveform"].to(device)
        message = random_message(16, clean_audio.shape[0], device, seed=seed + batch_idx)

        recon_wm, perturbed_final, delta = pgd_perturb(
            backbone, surrogate, clean_audio, message, text,
            epsilon, step_size, n_steps, random_start, lambda_wm=lambda_wm,
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

        linf = delta.abs().max().item()
        rms_signal = recon_wm.pow(2).mean().sqrt().item()
        rms_delta = delta.pow(2).mean().sqrt().item()
        snr_db = 20.0 * torch.log10(torch.tensor(rms_signal / max(rms_delta, 1e-12))).item()
        metrics["perturbation_linf"].append(linf)
        metrics["perturbation_snr_db"].append(snr_db)

        print(f"[batch {batch_idx}] sim {metrics['sim_before'][-1]:.4f} -> {metrics['sim_after'][-1]:.4f} | "
              f"acc {metrics['detection_acc_before'][-1]:.4f} -> {metrics['detection_acc_after'][-1]:.4f} | "
              f"delta_linf={linf:.5f} snr={snr_db:.1f}dB")

    return {k: sum(v) / len(v) for k, v in metrics.items()}, metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None,
                    help="Base watermark checkpoint. PGD adds delta ON TOP of this checkpoint's "
                         "recon_wm -- it does not require embedder-level disruption training. "
                         "Omit to test PGD alone against the pretrained (zero-init LoRA) VoiceMark.")
    p.add_argument("--include_ffn", action="store_true",
                    help="Set if --checkpoint is a stage2_capacity_ffn checkpoint (trained with "
                         "train_stage2_capacity.py). Leave unset for stage1 / plain stage2 checkpoints.")
    p.add_argument("--capacity_lora_r", type=int, default=32)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)

    p.add_argument("--epsilon", type=float, default=0.01,
                    help="L-infinity perturbation budget in raw waveform amplitude "
                         "(watermarked audio is float roughly in [-1,1]). 0.01 is a conservative "
                         "starting budget -- verify imperceptibility empirically (PESQ/STOI via "
                         "disruption_effectiveness*.py's sibling audio-quality scripts, or by "
                         "listening) rather than trusting this number on its own.")
    p.add_argument("--step_size", type=float, default=None,
                    help="PGD step size. Defaults to epsilon/4 (standard PGD heuristic: "
                         "step_size * n_steps should comfortably exceed epsilon so the sign-step "
                         "can reach the ball boundary from a random start).")
    p.add_argument("--n_steps", type=int, default=10)
    p.add_argument("--random_start", action="store_true", default=True)
    p.add_argument("--no_random_start", dest="random_start", action="store_false")
    p.add_argument("--lambda_wm", type=float, default=0.0,
                    help="Weight on the watermark-preservation term (compute_ldec on the PERTURBED "
                         "audio vs message), added to the PGD objective alongside sim_loss. Default "
                         "0.0 reproduces the original sim-only behavior (which the first full run "
                         "showed disrupts SIM hard but also drags detection ACC toward chance -- see "
                         "module docstring). Run --diagnostic first: it prints the raw |grad_sim| : "
                         "|grad_wm| ratio at delta, which is the value to start lambda_wm at (same "
                         "balancing logic as lambda_disrupt_max in gradient_diagnostic.py).")

    p.add_argument("--diagnostic", action="store_true",
                    help="Smoke-test mode: 1 batch, 1 PGD step, verbose per-step printing, exits "
                         "WITHOUT requiring/writing --output. Run this FIRST, same discipline as "
                         "gradient_diagnostic.py before train_stage2_capacity.py's full run.")
    p.add_argument("--save_samples_dir", type=str, default=None,
                    help="Instead of the aggregate eval, save --n_samples individual utterances as "
                         "sampleN_clean.wav / sampleN_watermarked.wav / sampleN_cloned.wav into this "
                         "directory -- SAME naming convention save_audio_samples.py uses, so the "
                         "existing quality_metrics.py (real PESQ/STOI/SI-SNR/WER, not just the SNR "
                         "proxy this script's aggregate JSON reports) and audio_diff_analysis.py (the "
                         "listenable/visual proof) both work UNMODIFIED against this directory. "
                         "'watermarked' here is recon_wm + delta -- the actual audio that would ship -- "
                         "not the pre-perturbation recon_wm, since the imperceptibility question that "
                         "matters is whether the PUBLISHED (post-PGD) audio stays transparent. Exits "
                         "after saving; does not require/write --output.")
    p.add_argument("--n_samples", type=int, default=4, help="Used only with --save_samples_dir.")
    p.add_argument("--output", type=str, default=None, help="Required unless --diagnostic/--save_samples_dir.")

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

    step_size = args.step_size if args.step_size is not None else args.epsilon / 4.0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset == "librispeech":
        eval_ds = LibriSpeechSubset(
            root=args.data_root, n_speakers=args.n_speakers,
            utterances_per_speaker=args.utterances_per_speaker,
            n_eval_speakers=args.n_eval_speakers,
            eval_utterances_per_speaker=args.eval_utterances_per_speaker,
            sample_rate=16000, crop_seconds=args.crop_seconds, split="eval",
        )
        collate_fn = collate_librispeech
    elif args.dataset == "vctk":
        eval_ds = VCTKSubset(
            root=args.vctk_root, n_speakers=args.n_speakers,
            utterances_per_speaker=args.utterances_per_speaker,
            n_eval_speakers=args.n_eval_speakers,
            eval_utterances_per_speaker=args.eval_utterances_per_speaker,
            sample_rate=16000, crop_seconds=args.crop_seconds, split="eval",
        )
        collate_fn = collate_vctk
    else:
        eval_ds = LibriTTSSubset(
            root=args.libritts_root, n_speakers=args.n_speakers,
            utterances_per_speaker=args.utterances_per_speaker,
            n_eval_speakers=args.n_eval_speakers,
            eval_utterances_per_speaker=args.eval_utterances_per_speaker,
            sample_rate=16000, crop_seconds=args.crop_seconds, split="eval",
        )
        collate_fn = collate_libritts

    print(f"[main] Evaluating on dataset: {args.dataset} | epsilon={args.epsilon} "
          f"step_size={step_size} n_steps={args.n_steps} random_start={args.random_start}")
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, drop_last=False)

    backbone = build_backbone(args.checkpoint, args.lora_r, args.lora_alpha,
                               args.include_ffn, args.capacity_lora_r)
    print("[main] Loading YourTTS surrogate (frozen)...")
    surrogate = load_yourtts_surrogate(device=device)
    mel_fn = SafeSpeechMelSpectrogram(sampling_rate=args.surrogate_sample_rate).to(device)

    if args.diagnostic:
        print("\n" + "=" * 60 + "\nDIAGNOSTIC MODE: 1 batch, 1 PGD step\n" + "=" * 60)
        batch = next(iter(eval_loader))
        clean_audio = batch["waveform"].to(device)
        message = random_message(16, clean_audio.shape[0], device, seed=123)
        recon_wm, perturbed, delta = pgd_perturb(
            backbone, surrogate, clean_audio, message, args.surrogate_text,
            args.epsilon, step_size, n_steps=1, random_start=args.random_start,
            lambda_wm=args.lambda_wm, verbose=True,
        )
        acc_before = detect_acc(backbone, recon_wm, message)
        acc_after = detect_acc(backbone, perturbed, message)
        print(f"\ndetection_acc: {acc_before:.4f} -> {acc_after:.4f} (single step -- expect small "
              f"movement either way at n_steps=1; this is checking the GRADIENT REACHES delta and "
              f"detection doesn't collapse outright, not measuring final disruption strength).")
        print("[main] Diagnostic complete. If grad_norm above was nonzero and detection_acc_after "
              "didn't collapse to ~0, proceed to a full run (drop --diagnostic, add --output).")
        return

    if args.save_samples_dir:
        os.makedirs(args.save_samples_dir, exist_ok=True)
        sample_loader = DataLoader(eval_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
        print(f"\n{'=' * 60}\nSaving {args.n_samples} samples to {args.save_samples_dir}/ "
              f"(lambda_wm={args.lambda_wm}, epsilon={args.epsilon})\n{'=' * 60}")
        for i, batch in enumerate(sample_loader):
            if i >= args.n_samples:
                break
            clean_audio = batch["waveform"].to(device)
            message = random_message(16, clean_audio.shape[0], device, seed=123 + i)

            recon_wm, perturbed_final, delta = pgd_perturb(
                backbone, surrogate, clean_audio, message, args.surrogate_text,
                args.epsilon, step_size, args.n_steps, args.random_start, lambda_wm=args.lambda_wm,
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
            print(f"  sample{i}: acc {acc_before:.4f} -> {acc_after:.4f}, "
                  f"delta_linf={delta.abs().max().item():.5f}")

        print(f"\n[main] Done. {args.n_samples} triplets saved in {args.save_samples_dir}/.")
        print(f"Next: python src/eval/quality_metrics.py --sample_dir {args.save_samples_dir} "
              f"--n_samples {args.n_samples} --skip_wer   # real PESQ/STOI/SI-SNR on the PUBLISHED "
              f"(post-PGD) audio vs clean")
        print(f"Then (per sample_idx):  python src/eval/audio_diff_analysis.py "
              f"--sample_dir {args.save_samples_dir} --sample_idx 0")
        print("And actually listen to sampleN_clean.wav vs sampleN_watermarked.wav -- "
              "no metric substitutes for that.")
        return

    label = (f"pgd_eps{args.epsilon}_n{args.n_steps}_lwm{args.lambda_wm}"
              + (f"_on_{args.checkpoint}" if args.checkpoint else "_on_baseline"))
    print(f"\n{'=' * 60}\nEvaluating: {label}\n{'=' * 60}")

    means, _all_values = run_pgd_eval(
        backbone, surrogate, eval_loader, device, mel_fn, args.surrogate_text,
        args.epsilon, step_size, args.n_steps, args.random_start, lambda_wm=args.lambda_wm,
    )

    print(f"\nSIM: {means['sim_before']:.4f} -> {means['sim_after']:.4f} "
          f"(delta={means['sim_before'] - means['sim_after']:+.4f}, positive = more disrupted)")
    print(f"Pivotal mel distance: {means['pivotal_before']:.4f} -> {means['pivotal_after']:.4f}")
    print(f"Detection ACC: {means['detection_acc_before']:.4f} -> {means['detection_acc_after']:.4f} "
          f"(drop={means['detection_acc_before'] - means['detection_acc_after']:+.4f})")
    print(f"Perturbation: mean L-inf={means['perturbation_linf']:.5f}, "
          f"mean SNR={means['perturbation_snr_db']:.1f} dB")

    with open(args.output, "w") as f:
        json.dump({
            "label": label, "checkpoint": args.checkpoint, "dataset": args.dataset,
            "include_ffn": args.include_ffn, "capacity_lora_r": args.capacity_lora_r if args.include_ffn else None,
            "pgd": {"epsilon": args.epsilon, "step_size": step_size, "n_steps": args.n_steps,
                     "random_start": args.random_start, "lambda_wm": args.lambda_wm},
            "results": {
                "sim_before": means["sim_before"], "sim_after": means["sim_after"],
                "sim_drop": means["sim_before"] - means["sim_after"],
                "pivotal_before": means["pivotal_before"], "pivotal_after": means["pivotal_after"],
                "detection_acc_before": means["detection_acc_before"],
                "detection_acc_after": means["detection_acc_after"],
                "detection_acc_drop": means["detection_acc_before"] - means["detection_acc_after"],
                "perturbation_linf_mean": means["perturbation_linf"],
                "perturbation_snr_db_mean": means["perturbation_snr_db"],
                "n_trials": len(eval_ds),
                # Backward-compat key so this drops into aggregate_results.py's
                # existing "Disruption (SIM)" classification/table unmodified.
                "sim": means["sim_after"],
            },
        }, f, indent=2)
    print(f"[main] Saved results to {args.output}")


if __name__ == "__main__":
    main()
