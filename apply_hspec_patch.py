"""
apply_hspec_patch.py -- H-SPEC (2026-09-12) + H-DIRECT (2026-09-12 follow-up) code update.

H-SPEC adds SafeSpeech's KL-to-noise / L1-to-noise SPEC terms and a clone-aware
watermark-preservation term to the existing PGD mechanism, all opt-in via
new --lambda_kl / --lambda_l1n / --lambda_wm_clone flags (default 0.0 =
byte-for-byte original behavior). H-SPEC ran 2026-09-12: null, adverse-direction
result (SIM rose rather than fell for both n=20 configs) -- see
pre-registration-hspec-2026-09-12.md. That motivated H-DIRECT: the same SPEC
terms (compute_kl_to_noise / compute_l1_to_noise), applied instead directly to
the PERTURBED INPUT's own mel spectrogram (before the surrogate clones it),
bypassing the YourTTS surrogate for this term entirely -- opt-in via new
--lambda_kl_direct / --lambda_l1n_direct flags (default 0.0 = no change to
H-SPEC/original behavior). Both patches thread through the core PGD script and
the Stage-0 eps-sweep script that calls it.

Usage (run once, from the repo root, right after cloning):
    python apply_hspec_patch.py src/eval/disruption_pgd.py src/eval/demucs_fallback_eval.py

Each edit is checked to occur EXACTLY ONCE in its target file before being
applied, and the four edit groups (EDITS_PGD, EDITS_HDIRECT_PGD, EDITS_STAGE0,
EDITS_HDIRECT_STAGE0) are applied strictly in that order so each group's
anchors are matched against the file state left by the previous group. If a
file has drifted from what this patch expects, it fails loudly naming which
edit didn't match, and leaves that file untouched.
"""
import sys

import sys

EDITS_PGD = [
("""from safespeech_losses import SafeSpeechMelSpectrogram, compute_pivotal_disruption_loss, compute_sim_disruption_loss""",
 """from safespeech_losses import (SafeSpeechMelSpectrogram, compute_pivotal_disruption_loss,
                                compute_sim_disruption_loss, compute_kl_to_noise, compute_l1_to_noise)"""),

("""def pgd_perturb(backbone, surrogate, clean_audio: torch.Tensor, message: torch.Tensor, text: str,
                 epsilon: float, step_size: float, n_steps: int, random_start: bool,
                 lambda_wm: float = 0.0, verbose: bool = False):
    \"\"\"
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
    \"\"\"
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

    return recon_wm, perturbed_final, delta.detach()""",
 """def pgd_perturb(backbone, surrogate, clean_audio: torch.Tensor, message: torch.Tensor, text: str,
                 epsilon: float, step_size: float, n_steps: int, random_start: bool,
                 lambda_wm: float = 0.0, mel_fn=None, lambda_kl: float = 0.0,
                 lambda_l1n: float = 0.0, lambda_wm_clone: float = 0.0,
                 noise_seed: int = None, verbose: bool = False):
    \"\"\"
    H-SPEC (2026-09-12 pre-registration): adds SafeSpeech's own SPEC KL/L1-
    to-noise terms (computed on the surrogate-cloned output vs a fixed random
    noise target, copied from their published method) and a clone-aware
    watermark-preservation term, on top of the original sim_loss + lambda_wm
    mechanism. All new lambdas default to 0.0 -- with all three at 0, this
    function reproduces the original sim_loss/lambda_wm behavior exactly.

    lambda_wm_clone: watermark cross-entropy on the SURROGATE-CLONED output
    (not the pre-clone perturbed audio, which is what the original lambda_wm
    protects) -- gives the objective a gradient signal for watermark survival
    THROUGH cloning, which lambda_wm alone never had. Only teaches survival
    through THIS surrogate's (YourTTS's) own cloning pathway -- the
    architecture where the watermark survives worst natively (ACC ceiling
    ~0.53) -- so do not expect this alone to fix F5-TTS transfer.

    lambda_kl / lambda_l1n: SafeSpeech's SPEC terms, pushing the surrogate's
    cloned output toward noise-like mel statistics rather than attacking one
    encoder's embedding space -- their paper credits exactly this for
    zero-shot transfer to architectures never optimized against (including
    F5-TTS), under a single-surrogate black-box setup structurally identical
    to this project's YourTTS surrogate. mel_fn is REQUIRED if either is
    nonzero. random_noise is fixed once per utterance (not resampled every
    step); noise_seed makes it reproducible.

    Returns (recon_wm, perturbed_final, delta) -- all detached.
    \"\"\"
    if (lambda_kl > 0 or lambda_l1n > 0) and mel_fn is None:
        raise ValueError("pgd_perturb: lambda_kl/lambda_l1n > 0 requires mel_fn (pass the "
                          "SafeSpeechMelSpectrogram instance already built in main()).")

    with torch.no_grad():
        out = backbone.forward_full(clean_audio, message)
        recon_wm = out["recon_wm"].detach()
        emb_clean = surrogate.compute_speaker_embedding(clean_audio).detach()

    if random_start:
        delta = (torch.rand_like(recon_wm) * 2 - 1) * epsilon
    else:
        delta = torch.zeros_like(recon_wm)
    delta = delta.detach().requires_grad_(True)

    random_noise = None
    noise_gen = None
    if noise_seed is not None:
        noise_gen = torch.Generator(device=delta.device).manual_seed(noise_seed)

    for step in range(n_steps):
        perturbed = torch.clamp(recon_wm + delta, -1.0, 1.0)

        cloned_output = surrogate.clone_voice(perturbed, text=text)
        emb_cloned = surrogate.compute_speaker_embedding(cloned_output)
        sim_loss = compute_sim_disruption_loss(emb_clean, emb_cloned)

        kl_loss = None
        l1n_loss = None
        if lambda_kl > 0 or lambda_l1n > 0:
            if random_noise is None or random_noise.shape != cloned_output.shape:
                if noise_gen is not None:
                    random_noise = torch.randn(cloned_output.shape, generator=noise_gen,
                                                device=cloned_output.device)
                else:
                    random_noise = torch.randn_like(cloned_output)
            if lambda_kl > 0:
                kl_loss = compute_kl_to_noise(mel_fn, cloned_output, random_noise)
            if lambda_l1n > 0:
                l1n_loss = compute_l1_to_noise(mel_fn, cloned_output, random_noise)

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
        wm_loss = None
        wm_loss_clone = None
        with torch.backends.cudnn.flags(enabled=False):
            if lambda_wm > 0 or verbose:
                detect_feat = backbone.model.st_model.forward_feature(perturbed)
                _logits, chunk_logits = backbone.model.detector(detect_feat)
                wm_loss = compute_ldec(chunk_logits, message)
            if lambda_wm_clone > 0 or verbose:
                detect_feat_clone = backbone.model.st_model.forward_feature(cloned_output)
                _logits_c, chunk_logits_clone = backbone.model.detector(detect_feat_clone)
                wm_loss_clone = compute_ldec(chunk_logits_clone, message)

        grad = torch.autograd.grad(sim_loss, delta, retain_graph=True, create_graph=False)[0]
        grad_sim_for_log = grad.clone() if verbose else None

        def _add_term(loss, lam, retain=True):
            nonlocal grad
            if loss is None:
                return None
            g = torch.autograd.grad(loss, delta, retain_graph=retain, create_graph=False)[0]
            grad = grad + lam * g
            return g

        grad_wm = _add_term(wm_loss, lambda_wm)
        grad_wm_clone = _add_term(wm_loss_clone, lambda_wm_clone)
        grad_kl = _add_term(kl_loss, lambda_kl)
        grad_l1n = _add_term(l1n_loss, lambda_l1n, retain=False)
        grad_norm = grad.norm().item()

        if verbose:
            parts = [f"sim_loss={sim_loss.item():.4f} |grad_sim|={grad_sim_for_log.norm().item():.4e}"]
            for name, loss_val, g in [("wm", wm_loss, grad_wm), ("wm_clone", wm_loss_clone, grad_wm_clone),
                                       ("kl", kl_loss, grad_kl), ("l1n", l1n_loss, grad_l1n)]:
                if loss_val is not None:
                    gnorm = g.norm().item() if g is not None else 0.0
                    ratio = grad_sim_for_log.norm().item() / max(gnorm, 1e-12)
                    parts.append(f"{name}_loss={loss_val.item():.4f} |grad_{name}|={gnorm:.4e} "
                                 f"(sim:{name} ratio={ratio:.2f})")
            print(f"  [pgd step {step}] " + " | ".join(parts) +
                  f" | |grad_combined|={grad_norm:.4e} delta_linf={delta.abs().max().item():.6f}")
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

    return recon_wm, perturbed_final, delta.detach()"""),

("""def run_pgd_eval(backbone, surrogate, eval_loader, device, mel_fn, text: str,
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
        )""",
 """def run_pgd_eval(backbone, surrogate, eval_loader, device, mel_fn, text: str,
                  epsilon: float, step_size: float, n_steps: int, random_start: bool,
                  lambda_wm: float = 0.0, lambda_kl: float = 0.0, lambda_l1n: float = 0.0,
                  lambda_wm_clone: float = 0.0, seed: int = 123) -> dict:
    metrics = {k: [] for k in [
        "sim_before", "sim_after", "pivotal_before", "pivotal_after",
        "detection_acc_before", "detection_acc_after",
        "detection_acc_clone_before", "detection_acc_clone_after",
        "perturbation_linf", "perturbation_snr_db",
    ]}

    for batch_idx, batch in enumerate(eval_loader):
        clean_audio = batch["waveform"].to(device)
        message = random_message(16, clean_audio.shape[0], device, seed=seed + batch_idx)

        recon_wm, perturbed_final, delta = pgd_perturb(
            backbone, surrogate, clean_audio, message, text,
            epsilon, step_size, n_steps, random_start, lambda_wm=lambda_wm,
            mel_fn=mel_fn, lambda_kl=lambda_kl, lambda_l1n=lambda_l1n,
            lambda_wm_clone=lambda_wm_clone, noise_seed=seed + batch_idx,
        )"""),

("""        metrics["detection_acc_before"].append(detect_acc(backbone, recon_wm, message))
        metrics["detection_acc_after"].append(detect_acc(backbone, perturbed_final, message))

        linf = delta.abs().max().item()
        rms_signal = recon_wm.pow(2).mean().sqrt().item()
        rms_delta = delta.pow(2).mean().sqrt().item()
        snr_db = 20.0 * torch.log10(torch.tensor(rms_signal / max(rms_delta, 1e-12))).item()
        metrics["perturbation_linf"].append(linf)
        metrics["perturbation_snr_db"].append(snr_db)

        print(f"[batch {batch_idx}] sim {metrics['sim_before'][-1]:.4f} -> {metrics['sim_after'][-1]:.4f} | "
              f"acc {metrics['detection_acc_before'][-1]:.4f} -> {metrics['detection_acc_after'][-1]:.4f} | "
              f"delta_linf={linf:.5f} snr={snr_db:.1f}dB")""",
 """        metrics["detection_acc_before"].append(detect_acc(backbone, recon_wm, message))
        metrics["detection_acc_after"].append(detect_acc(backbone, perturbed_final, message))
        metrics["detection_acc_clone_before"].append(detect_acc(backbone, cloned_before, message))
        metrics["detection_acc_clone_after"].append(detect_acc(backbone, cloned_after, message))

        linf = delta.abs().max().item()
        rms_signal = recon_wm.pow(2).mean().sqrt().item()
        rms_delta = delta.pow(2).mean().sqrt().item()
        snr_db = 20.0 * torch.log10(torch.tensor(rms_signal / max(rms_delta, 1e-12))).item()
        metrics["perturbation_linf"].append(linf)
        metrics["perturbation_snr_db"].append(snr_db)

        print(f"[batch {batch_idx}] sim {metrics['sim_before'][-1]:.4f} -> {metrics['sim_after'][-1]:.4f} | "
              f"acc(published) {metrics['detection_acc_before'][-1]:.4f} -> {metrics['detection_acc_after'][-1]:.4f} | "
              f"acc(clone) {metrics['detection_acc_clone_before'][-1]:.4f} -> {metrics['detection_acc_clone_after'][-1]:.4f} | "
              f"delta_linf={linf:.5f} snr={snr_db:.1f}dB")"""),

("""    p.add_argument("--lambda_wm", type=float, default=0.0,
                    help="Weight on the watermark-preservation term (compute_ldec on the PERTURBED "
                         "audio vs message), added to the PGD objective alongside sim_loss. Default "
                         "0.0 reproduces the original sim-only behavior (which the first full run "
                         "showed disrupts SIM hard but also drags detection ACC toward chance -- see "
                         "module docstring). Run --diagnostic first: it prints the raw |grad_sim| : "
                         "|grad_wm| ratio at delta, which is the value to start lambda_wm at (same "
                         "balancing logic as lambda_disrupt_max in gradient_diagnostic.py).")""",
 """    p.add_argument("--lambda_wm", type=float, default=0.0,
                    help="Weight on the watermark-preservation term (compute_ldec on the PERTURBED "
                         "audio vs message), added to the PGD objective alongside sim_loss. Default "
                         "0.0 reproduces the original sim-only behavior (which the first full run "
                         "showed disrupts SIM hard but also drags detection ACC toward chance -- see "
                         "module docstring). Run --diagnostic first: it prints the raw |grad_sim| : "
                         "|grad_wm| ratio at delta, which is the value to start lambda_wm at (same "
                         "balancing logic as lambda_disrupt_max in gradient_diagnostic.py).")
    p.add_argument("--lambda_wm_clone", type=float, default=0.0,
                    help="H-SPEC (2026-09-12). Weight on compute_ldec evaluated on the SURROGATE-"
                         "CLONED output, not the pre-clone perturbed audio -- gives watermark survival "
                         "THROUGH cloning a gradient signal, which --lambda_wm alone never had. "
                         "Default 0.0 = no change to existing behavior.")
    p.add_argument("--lambda_kl", type=float, default=0.0,
                    help="H-SPEC (2026-09-12). Weight on compute_kl_to_noise (SafeSpeech's SPEC "
                         "KL-divergence-to-noise term) evaluated on cloned_output vs fixed random "
                         "noise -- the term their paper credits for zero-shot transfer to unseen "
                         "architectures including F5-TTS. Default 0.0 = no change to existing behavior.")
    p.add_argument("--lambda_l1n", type=float, default=0.0,
                    help="H-SPEC (2026-09-12). Weight on compute_l1_to_noise (SafeSpeech's SPEC "
                         "L1-to-noise term), same rationale as --lambda_kl. Default 0.0 = no change.")"""),

("""            recon_wm, perturbed, delta = pgd_perturb(
                backbone, surrogate, clean_audio, message, args.surrogate_text,
                args.epsilon, step_size, n_steps=1, random_start=args.random_start,
                lambda_wm=args.lambda_wm, verbose=True,
            )""".replace("                ", "        ") if False else
 """        recon_wm, perturbed, delta = pgd_perturb(
            backbone, surrogate, clean_audio, message, args.surrogate_text,
            args.epsilon, step_size, n_steps=1, random_start=args.random_start,
            lambda_wm=args.lambda_wm, verbose=True,
        )""",
 """        recon_wm, perturbed, delta = pgd_perturb(
            backbone, surrogate, clean_audio, message, args.surrogate_text,
            args.epsilon, step_size, n_steps=1, random_start=args.random_start,
            lambda_wm=args.lambda_wm, mel_fn=mel_fn, lambda_kl=args.lambda_kl,
            lambda_l1n=args.lambda_l1n, lambda_wm_clone=args.lambda_wm_clone,
            noise_seed=123, verbose=True,
        )"""),

("""            recon_wm, perturbed_final, delta = pgd_perturb(
                backbone, surrogate, clean_audio, message, args.surrogate_text,
                args.epsilon, step_size, args.n_steps, args.random_start, lambda_wm=args.lambda_wm,
            )
            with torch.no_grad():
                cloned_after = surrogate.clone_voice(perturbed_final, text=args.surrogate_text)""",
 """            recon_wm, perturbed_final, delta = pgd_perturb(
                backbone, surrogate, clean_audio, message, args.surrogate_text,
                args.epsilon, step_size, args.n_steps, args.random_start, lambda_wm=args.lambda_wm,
                mel_fn=mel_fn, lambda_kl=args.lambda_kl, lambda_l1n=args.lambda_l1n,
                lambda_wm_clone=args.lambda_wm_clone, noise_seed=123 + i,
            )
            with torch.no_grad():
                cloned_after = surrogate.clone_voice(perturbed_final, text=args.surrogate_text)"""),

("""    label = (f"pgd_eps{args.epsilon}_n{args.n_steps}_lwm{args.lambda_wm}"
              + (f"_on_{args.checkpoint}" if args.checkpoint else "_on_baseline"))
    print(f"\\n{'=' * 60}\\nEvaluating: {label}\\n{'=' * 60}")

    means, _all_values = run_pgd_eval(
        backbone, surrogate, eval_loader, device, mel_fn, args.surrogate_text,
        args.epsilon, step_size, args.n_steps, args.random_start, lambda_wm=args.lambda_wm,
    )

    print(f"\\nSIM: {means['sim_before']:.4f} -> {means['sim_after']:.4f} "
          f"(delta={means['sim_before'] - means['sim_after']:+.4f}, positive = more disrupted)")
    print(f"Pivotal mel distance: {means['pivotal_before']:.4f} -> {means['pivotal_after']:.4f}")
    print(f"Detection ACC: {means['detection_acc_before']:.4f} -> {means['detection_acc_after']:.4f} "
          f"(drop={means['detection_acc_before'] - means['detection_acc_after']:+.4f})")
    print(f"Perturbation: mean L-inf={means['perturbation_linf']:.5f}, "
          f"mean SNR={means['perturbation_snr_db']:.1f} dB")""",
 """    label = (f"pgd_eps{args.epsilon}_n{args.n_steps}_lwm{args.lambda_wm}"
              + (f"_lwmc{args.lambda_wm_clone}" if args.lambda_wm_clone else "")
              + (f"_lkl{args.lambda_kl}" if args.lambda_kl else "")
              + (f"_ll1n{args.lambda_l1n}" if args.lambda_l1n else "")
              + (f"_on_{args.checkpoint}" if args.checkpoint else "_on_baseline"))
    print(f"\\n{'=' * 60}\\nEvaluating: {label}\\n{'=' * 60}")

    means, _all_values = run_pgd_eval(
        backbone, surrogate, eval_loader, device, mel_fn, args.surrogate_text,
        args.epsilon, step_size, args.n_steps, args.random_start, lambda_wm=args.lambda_wm,
        lambda_kl=args.lambda_kl, lambda_l1n=args.lambda_l1n, lambda_wm_clone=args.lambda_wm_clone,
    )

    print(f"\\nSIM: {means['sim_before']:.4f} -> {means['sim_after']:.4f} "
          f"(delta={means['sim_before'] - means['sim_after']:+.4f}, positive = more disrupted)")
    print(f"Pivotal mel distance: {means['pivotal_before']:.4f} -> {means['pivotal_after']:.4f}")
    print(f"Detection ACC (published, pre-clone): {means['detection_acc_before']:.4f} -> "
          f"{means['detection_acc_after']:.4f} "
          f"(drop={means['detection_acc_before'] - means['detection_acc_after']:+.4f})")
    print(f"Detection ACC (through YourTTS clone): {means['detection_acc_clone_before']:.4f} -> "
          f"{means['detection_acc_clone_after']:.4f} "
          f"(drop={means['detection_acc_clone_before'] - means['detection_acc_clone_after']:+.4f}) "
          f"-- THIS is the number the eps sweep showed collapsing; compare against the sim-only "
          f"baseline you already have at the same epsilon.")
    print(f"Perturbation: mean L-inf={means['perturbation_linf']:.5f}, "
          f"mean SNR={means['perturbation_snr_db']:.1f} dB")"""),

("""    base_results = {
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
        "audiopure_acc_after_mean": None,
    }
    if args.output:
        out = {
            "label": label, "checkpoint": args.checkpoint, "dataset": args.dataset,
            "include_ffn": args.include_ffn, "capacity_lora_r": args.capacity_lora_r if args.include_ffn else None,
            "pgd": {"epsilon": args.epsilon, "step_size": step_size, "n_steps": args.n_steps,
                     "random_start": args.random_start, "lambda_wm": args.lambda_wm},
            "results": base_results,
        }""",
 """    base_results = {
        "sim_before": means["sim_before"], "sim_after": means["sim_after"],
        "sim_drop": means["sim_before"] - means["sim_after"],
        "pivotal_before": means["pivotal_before"], "pivotal_after": means["pivotal_after"],
        "detection_acc_before": means["detection_acc_before"],
        "detection_acc_after": means["detection_acc_after"],
        "detection_acc_drop": means["detection_acc_before"] - means["detection_acc_after"],
        "detection_acc_clone_before": means["detection_acc_clone_before"],
        "detection_acc_clone_after": means["detection_acc_clone_after"],
        "detection_acc_clone_drop": means["detection_acc_clone_before"] - means["detection_acc_clone_after"],
        "perturbation_linf_mean": means["perturbation_linf"],
        "perturbation_snr_db_mean": means["perturbation_snr_db"],
        "n_trials": len(eval_ds),
        # Backward-compat key so this drops into aggregate_results.py's
        # existing "Disruption (SIM)" classification/table unmodified.
        "sim": means["sim_after"],
        "audiopure_acc_after_mean": None,
    }
    if args.output:
        out = {
            "label": label, "checkpoint": args.checkpoint, "dataset": args.dataset,
            "include_ffn": args.include_ffn, "capacity_lora_r": args.capacity_lora_r if args.include_ffn else None,
            "pgd": {"epsilon": args.epsilon, "step_size": step_size, "n_steps": args.n_steps,
                     "random_start": args.random_start, "lambda_wm": args.lambda_wm,
                     "lambda_wm_clone": args.lambda_wm_clone, "lambda_kl": args.lambda_kl,
                     "lambda_l1n": args.lambda_l1n},
            "results": base_results,
        }"""),
]



EDITS_STAGE0 = [
("""from disruption_pgd import build_backbone, random_message, detect_acc, compute_sim, pgd_perturb""",
 """from disruption_pgd import build_backbone, random_message, detect_acc, compute_sim, pgd_perturb
from safespeech_losses import SafeSpeechMelSpectrogram"""),

("""    p.add_argument("--lambda_wm", type=float, default=1.0,
                   help="1.0 = the combined system (watermark preserved during PGD). "
                        "0.0 = SafeSpeech-style arm, no watermark-preservation term.")

    p.add_argument("--surrogate_text", type=str, default="This is a test sentence for voice cloning.")""",
 """    p.add_argument("--lambda_wm", type=float, default=1.0,
                   help="1.0 = the combined system (watermark preserved during PGD). "
                        "0.0 = SafeSpeech-style arm, no watermark-preservation term.")
    p.add_argument("--lambda_wm_clone", type=float, default=0.0,
                   help="H-SPEC (2026-09-12). Watermark loss on the surrogate-CLONED output "
                        "rather than the pre-clone audio. Default 0.0 = no change.")
    p.add_argument("--lambda_kl", type=float, default=0.0,
                   help="H-SPEC (2026-09-12). SafeSpeech's SPEC KL-to-noise term on the "
                        "surrogate-cloned output. Default 0.0 = no change.")
    p.add_argument("--lambda_l1n", type=float, default=0.0,
                   help="H-SPEC (2026-09-12). SafeSpeech's SPEC L1-to-noise term, same "
                        "rationale as --lambda_kl. Default 0.0 = no change.")

    p.add_argument("--surrogate_text", type=str, default="This is a test sentence for voice cloning.")"""),

("""    device = "cuda" if torch.cuda.is_available() else "cpu"
    step_size = args.step_size if args.step_size is not None else args.epsilon / 4""",
 """    device = "cuda" if torch.cuda.is_available() else "cpu"
    step_size = args.step_size if args.step_size is not None else args.epsilon / 4
    mel_fn = SafeSpeechMelSpectrogram(sampling_rate=16000).to(device)"""),

("""        recon_wm, perturbed, _d = pgd_perturb(
            backbone, surrogate, clean_audio, message, args.surrogate_text,
            args.epsilon, step_size, args.n_steps, args.random_start, lambda_wm=args.lambda_wm,
        )
        den = denoise(perturbed.detach())""",
 """        recon_wm, perturbed, _d = pgd_perturb(
            backbone, surrogate, clean_audio, message, args.surrogate_text,
            args.epsilon, step_size, args.n_steps, args.random_start, lambda_wm=args.lambda_wm,
            mel_fn=mel_fn, lambda_kl=args.lambda_kl, lambda_l1n=args.lambda_l1n,
            lambda_wm_clone=args.lambda_wm_clone, noise_seed=123,
        )
        den = denoise(perturbed.detach())"""),

("""        recon_wm, perturbed, _d = pgd_perturb(
            backbone, surrogate, clean_audio, message, args.surrogate_text,
            args.epsilon, step_size, args.n_steps, args.random_start, lambda_wm=args.lambda_wm,
        )
        perturbed = perturbed.detach()""",
 """        recon_wm, perturbed, _d = pgd_perturb(
            backbone, surrogate, clean_audio, message, args.surrogate_text,
            args.epsilon, step_size, args.n_steps, args.random_start, lambda_wm=args.lambda_wm,
            mel_fn=mel_fn, lambda_kl=args.lambda_kl, lambda_l1n=args.lambda_l1n,
            lambda_wm_clone=args.lambda_wm_clone, noise_seed=123 + i,
        )
        perturbed = perturbed.detach()"""),

("""    print(f"DEMUCS FALLBACK EVAL | backend={backend} | eps={args.epsilon} | "
          f"lambda_wm={args.lambda_wm} | n={args.n_utterances}")""",
 """    print(f"DEMUCS FALLBACK EVAL | backend={backend} | eps={args.epsilon} | "
          f"lambda_wm={args.lambda_wm} lambda_kl={args.lambda_kl} lambda_l1n={args.lambda_l1n} "
          f"lambda_wm_clone={args.lambda_wm_clone} | n={args.n_utterances}")"""),

("""                "pgd": {"epsilon": args.epsilon, "n_steps": args.n_steps,
                        "lambda_wm": args.lambda_wm, "step_size": step_size},""",
 """                "pgd": {"epsilon": args.epsilon, "n_steps": args.n_steps,
                        "lambda_wm": args.lambda_wm, "step_size": step_size,
                        "lambda_kl": args.lambda_kl, "lambda_l1n": args.lambda_l1n,
                        "lambda_wm_clone": args.lambda_wm_clone},"""),
]

EDITS_HDIRECT_PGD = [
    # signature: add lambda_kl_direct/lambda_l1n_direct params
    ('def pgd_perturb(backbone, surrogate, clean_audio: torch.Tensor, message: torch.Tensor, text: str,\n                 epsilon: float, step_size: float, n_steps: int, random_start: bool,\n                 lambda_wm: float = 0.0, mel_fn=None, lambda_kl: float = 0.0,\n                 lambda_l1n: float = 0.0, lambda_wm_clone: float = 0.0,\n                 noise_seed: int = None, verbose: bool = False):\n',
     'def pgd_perturb(backbone, surrogate, clean_audio: torch.Tensor, message: torch.Tensor, text: str,\n                 epsilon: float, step_size: float, n_steps: int, random_start: bool,\n                 lambda_wm: float = 0.0, mel_fn=None, lambda_kl: float = 0.0,\n                 lambda_l1n: float = 0.0, lambda_wm_clone: float = 0.0,\n                 lambda_kl_direct: float = 0.0, lambda_l1n_direct: float = 0.0,\n                 noise_seed: int = None, verbose: bool = False):\n'),
    # docstring: explain H-DIRECT rationale
    ("    to this project's YourTTS surrogate. mel_fn is REQUIRED if either is\n    nonzero. random_noise is fixed once per utterance (not resampled every\n    step); noise_seed makes it reproducible.\n\n    Returns (recon_wm, perturbed_final, delta) -- all detached.\n",
     "    to this project's YourTTS surrogate. mel_fn is REQUIRED if either is\n    nonzero. random_noise is fixed once per utterance (not resampled every\n    step); noise_seed makes it reproducible.\n\n    H-DIRECT (2026-09-12, follow-up to H-SPEC): lambda_kl_direct / lambda_l1n_direct\n    apply the SAME SafeSpeech SPEC terms directly to the PERTURBED INPUT's own mel\n    spectrogram (`perturbed`, i.e. recon_wm + delta -- BEFORE the surrogate clones it),\n    instead of to the surrogate's generated output. Motivation: the project's own\n    five-architecture ladder (research-question-answered doc) established that F5-TTS\n    does not regenerate the reference through a decoder -- it RETAINS the reference mel\n    almost verbatim. H-SPEC's kl_loss/l1n_loss (above) push the YourTTS SURROGATE's\n    generated clone toward noise -- a signal that only matters if F5-TTS's own\n    conditioning behaves like the surrogate's, which the eps-sweep and H-SPEC results\n    both suggest it does not. Pushing the INPUT itself toward noise-like mel statistics\n    requires no surrogate forward pass for this term at all, so it cannot fail to\n    transfer to F5-TTS for the same reason H-SPEC's kl_loss did -- it attacks the thing\n    F5-TTS actually copies, not a YourTTS-mediated proxy for it. Still a hypothesis, not\n    a promise: this may simply behave like brute-force epsilon (same quality cost), which\n    is a legitimate, reportable outcome too. random_noise_direct is independent of\n    random_noise (different shape: perturbed's, not cloned_output's) but drawn from the\n    same noise_gen when noise_seed is set, so a given noise_seed remains fully\n    reproducible across all four SPEC-style terms.\n\n    Returns (recon_wm, perturbed_final, delta) -- all detached.\n"),
    # validate mel_fn requirement covers new lambdas
    ('    if (lambda_kl > 0 or lambda_l1n > 0) and mel_fn is None:\n        raise ValueError("pgd_perturb: lambda_kl/lambda_l1n > 0 requires mel_fn (pass the "\n                          "SafeSpeechMelSpectrogram instance already built in main()).")\n',
     '    if (lambda_kl > 0 or lambda_l1n > 0 or lambda_kl_direct > 0 or lambda_l1n_direct > 0) and mel_fn is None:\n        raise ValueError("pgd_perturb: lambda_kl/lambda_l1n/lambda_kl_direct/lambda_l1n_direct > 0 "\n                          "requires mel_fn (pass the SafeSpeechMelSpectrogram instance already "\n                          "built in main()).")\n'),
    # init random_noise_direct alongside random_noise
    ('    random_noise = None\n    noise_gen = None\n    if noise_seed is not None:\n        noise_gen = torch.Generator(device=delta.device).manual_seed(noise_seed)\n',
     '    random_noise = None\n    random_noise_direct = None\n    noise_gen = None\n    if noise_seed is not None:\n        noise_gen = torch.Generator(device=delta.device).manual_seed(noise_seed)\n'),
    # compute kl_loss_direct/l1n_loss_direct on perturbed input
    ('        kl_loss = None\n        l1n_loss = None\n        if lambda_kl > 0 or lambda_l1n > 0:\n            if random_noise is None or random_noise.shape != cloned_output.shape:\n                if noise_gen is not None:\n                    random_noise = torch.randn(cloned_output.shape, generator=noise_gen,\n                                                device=cloned_output.device)\n                else:\n                    random_noise = torch.randn_like(cloned_output)\n            if lambda_kl > 0:\n                kl_loss = compute_kl_to_noise(mel_fn, cloned_output, random_noise)\n            if lambda_l1n > 0:\n                l1n_loss = compute_l1_to_noise(mel_fn, cloned_output, random_noise)\n',
     '        kl_loss = None\n        l1n_loss = None\n        if lambda_kl > 0 or lambda_l1n > 0:\n            if random_noise is None or random_noise.shape != cloned_output.shape:\n                if noise_gen is not None:\n                    random_noise = torch.randn(cloned_output.shape, generator=noise_gen,\n                                                device=cloned_output.device)\n                else:\n                    random_noise = torch.randn_like(cloned_output)\n            if lambda_kl > 0:\n                kl_loss = compute_kl_to_noise(mel_fn, cloned_output, random_noise)\n            if lambda_l1n > 0:\n                l1n_loss = compute_l1_to_noise(mel_fn, cloned_output, random_noise)\n\n        kl_loss_direct = None\n        l1n_loss_direct = None\n        if lambda_kl_direct > 0 or lambda_l1n_direct > 0:\n            if random_noise_direct is None or random_noise_direct.shape != perturbed.shape:\n                if noise_gen is not None:\n                    random_noise_direct = torch.randn(perturbed.shape, generator=noise_gen,\n                                                       device=perturbed.device)\n                else:\n                    random_noise_direct = torch.randn_like(perturbed)\n            if lambda_kl_direct > 0:\n                kl_loss_direct = compute_kl_to_noise(mel_fn, perturbed, random_noise_direct)\n            if lambda_l1n_direct > 0:\n                l1n_loss_direct = compute_l1_to_noise(mel_fn, perturbed, random_noise_direct)\n'),
    # accumulate grad_kl_direct/grad_l1n_direct
    ('        grad_wm = _add_term(wm_loss, lambda_wm)\n        grad_wm_clone = _add_term(wm_loss_clone, lambda_wm_clone)\n        grad_kl = _add_term(kl_loss, lambda_kl)\n        grad_l1n = _add_term(l1n_loss, lambda_l1n, retain=False)\n        grad_norm = grad.norm().item()\n',
     '        grad_wm = _add_term(wm_loss, lambda_wm)\n        grad_wm_clone = _add_term(wm_loss_clone, lambda_wm_clone)\n        grad_kl = _add_term(kl_loss, lambda_kl)\n        grad_l1n = _add_term(l1n_loss, lambda_l1n)\n        grad_kl_direct = _add_term(kl_loss_direct, lambda_kl_direct)\n        grad_l1n_direct = _add_term(l1n_loss_direct, lambda_l1n_direct, retain=False)\n        grad_norm = grad.norm().item()\n'),
    # verbose diagnostic prints new terms
    ('        if verbose:\n            parts = [f"sim_loss={sim_loss.item():.4f} |grad_sim|={grad_sim_for_log.norm().item():.4e}"]\n            for name, loss_val, g in [("wm", wm_loss, grad_wm), ("wm_clone", wm_loss_clone, grad_wm_clone),\n                                       ("kl", kl_loss, grad_kl), ("l1n", l1n_loss, grad_l1n)]:\n',
     '        if verbose:\n            parts = [f"sim_loss={sim_loss.item():.4f} |grad_sim|={grad_sim_for_log.norm().item():.4e}"]\n            for name, loss_val, g in [("wm", wm_loss, grad_wm), ("wm_clone", wm_loss_clone, grad_wm_clone),\n                                       ("kl", kl_loss, grad_kl), ("l1n", l1n_loss, grad_l1n),\n                                       ("kl_direct", kl_loss_direct, grad_kl_direct),\n                                       ("l1n_direct", l1n_loss_direct, grad_l1n_direct)]:\n'),
]

EDITS_HDIRECT_STAGE0 = [
    # CLI: add --lambda_kl_direct/--lambda_l1n_direct
    ('    p.add_argument("--lambda_l1n", type=float, default=0.0,\n                   help="H-SPEC (2026-09-12). SafeSpeech\'s SPEC L1-to-noise term, same "\n                        "rationale as --lambda_kl. Default 0.0 = no change.")\n',
     '    p.add_argument("--lambda_l1n", type=float, default=0.0,\n                   help="H-SPEC (2026-09-12). SafeSpeech\'s SPEC L1-to-noise term, same "\n                        "rationale as --lambda_kl. Default 0.0 = no change.")\n    p.add_argument("--lambda_kl_direct", type=float, default=0.0,\n                   help="H-DIRECT (2026-09-12, follow-up to H-SPEC). SafeSpeech\'s SPEC "\n                        "KL-to-noise term applied to the PERTURBED INPUT\'s own mel "\n                        "(before cloning), not the surrogate\'s cloned output -- bypasses "\n                        "the YourTTS surrogate for this term entirely, so it cannot fail to "\n                        "transfer to F5-TTS the way lambda_kl did. Default 0.0 = no change.")\n    p.add_argument("--lambda_l1n_direct", type=float, default=0.0,\n                   help="H-DIRECT (2026-09-12). SafeSpeech\'s SPEC L1-to-noise term, same "\n                        "input-mel-direct rationale as --lambda_kl_direct. Default 0.0 = no change.")\n'),
    # diagnostic pgd_perturb call: thread new lambdas
    ('            mel_fn=mel_fn, lambda_kl=args.lambda_kl, lambda_l1n=args.lambda_l1n,\n            lambda_wm_clone=args.lambda_wm_clone, noise_seed=123,\n        )\n',
     '            mel_fn=mel_fn, lambda_kl=args.lambda_kl, lambda_l1n=args.lambda_l1n,\n            lambda_wm_clone=args.lambda_wm_clone,\n            lambda_kl_direct=args.lambda_kl_direct, lambda_l1n_direct=args.lambda_l1n_direct,\n            noise_seed=123,\n        )\n'),
    # print header: show new lambdas
    ('    print(f"DEMUCS FALLBACK EVAL | backend={backend} | eps={args.epsilon} | "\n          f"lambda_wm={args.lambda_wm} lambda_kl={args.lambda_kl} lambda_l1n={args.lambda_l1n} "\n          f"lambda_wm_clone={args.lambda_wm_clone} | n={args.n_utterances}")\n',
     '    print(f"DEMUCS FALLBACK EVAL | backend={backend} | eps={args.epsilon} | "\n          f"lambda_wm={args.lambda_wm} lambda_kl={args.lambda_kl} lambda_l1n={args.lambda_l1n} "\n          f"lambda_wm_clone={args.lambda_wm_clone} lambda_kl_direct={args.lambda_kl_direct} "\n          f"lambda_l1n_direct={args.lambda_l1n_direct} | n={args.n_utterances}")\n'),
    # full-run pgd_perturb call: thread new lambdas
    ('            mel_fn=mel_fn, lambda_kl=args.lambda_kl, lambda_l1n=args.lambda_l1n,\n            lambda_wm_clone=args.lambda_wm_clone, noise_seed=123 + i,\n        )\n',
     '            mel_fn=mel_fn, lambda_kl=args.lambda_kl, lambda_l1n=args.lambda_l1n,\n            lambda_wm_clone=args.lambda_wm_clone,\n            lambda_kl_direct=args.lambda_kl_direct, lambda_l1n_direct=args.lambda_l1n_direct,\n            noise_seed=123 + i,\n        )\n'),
    # output JSON: record new lambdas
    ('                "pgd": {"epsilon": args.epsilon, "n_steps": args.n_steps,\n                        "lambda_wm": args.lambda_wm, "step_size": step_size,\n                        "lambda_kl": args.lambda_kl, "lambda_l1n": args.lambda_l1n,\n                        "lambda_wm_clone": args.lambda_wm_clone},\n',
     '                "pgd": {"epsilon": args.epsilon, "n_steps": args.n_steps,\n                        "lambda_wm": args.lambda_wm, "step_size": step_size,\n                        "lambda_kl": args.lambda_kl, "lambda_l1n": args.lambda_l1n,\n                        "lambda_wm_clone": args.lambda_wm_clone,\n                        "lambda_kl_direct": args.lambda_kl_direct,\n                        "lambda_l1n_direct": args.lambda_l1n_direct},\n'),
]


def _apply(path, edits):
    with open(path, "r") as f:
        content = f.read()
    for i, (old, new) in enumerate(edits):
        count = content.count(old)
        if count != 1:
            print(f"FAILED on {path}, edit {i+1}/{len(edits)}: expected 1 occurrence of the "
                  f"anchor text, found {count}. This file has diverged from what this patch "
                  f"expects -- stopping WITHOUT writing {path} (it is untouched). Paste this "
                  f"message back to Claude with the current content of {path} to regenerate "
                  f"the patch.")
            return False
        content = content.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(content)
    print(f"OK: applied all {len(edits)} edits to {path}.")
    return True


def main():
    if len(sys.argv) != 3:
        print("Usage: python apply_hspec_patch.py <path to disruption_pgd.py> "
              "<path to demucs_fallback_eval.py>")
        sys.exit(1)
    pgd_path, stage0_path = sys.argv[1], sys.argv[2]
    ok = _apply(pgd_path, EDITS_PGD)
    ok = _apply(pgd_path, EDITS_HDIRECT_PGD) if ok else False
    ok = _apply(stage0_path, EDITS_STAGE0) if ok else False
    ok = _apply(stage0_path, EDITS_HDIRECT_STAGE0) if ok else False
    if not ok:
        sys.exit(1)
    print("H-SPEC + H-DIRECT patch fully applied to both files.")


if __name__ == "__main__":
    main()
