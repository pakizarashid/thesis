"""
src/eval/f5_whitebox_pgd.py  (H-WHITEBOX, pre-registered 2026-10-14: claude/pre-registration-hwhitebox-2026-10-14.md)

PGD against F5-TTS ITSELF instead of the YourTTS surrogate.

F5-TTS (f5-tts 1.1.22, the version the signatures were read from) clones by flow-matching mel infilling: the prompt mel is fed to a DiT as `cond`,
the text as `text`, and a neural ODE (Euler, `steps` NFE, classifier-free guidance) turns noise into the mel of prompt+continuation. CFM.sample is
@torch.no_grad, so this module re-implements the same loop WITHOUT no_grad (same transformer call, same Euler update, same CFG combination, same
timestep schedule, same duration rule) and lets the gradient flow back to the waveform:

    waveform (16 kHz) -> rms normalise -> resample 24 kHz -> CFM.mel_spec -> DiT Euler rollout (K steps, CFG, gradient checkpointing)
                       -> generated mel [prompt_frames:]  -> (arm "wspk") Vocos decode -> ECAPA embedding

Two arms (see the pre-registration):
    wmel : maximise the L1 distance between the generated mel for the PROTECTED prompt and for the unprotected prompt (same noise).
           No vocoder, no speaker model in the loss.
    wspk : minimise ECAPA similarity between the Vocos-decoded generated region and the clean utterance (ECAPA is in the loss).
Both keep the VoiceMark watermark term exactly as pgd_perturb does (compute_ldec on the detector with cuDNN disabled).

Gradient balance. pgd_perturb sums raw gradients (grad_sim + lambda_wm * grad_wm) and takes the sign, so only the RATIO of the two gradients
matters. A different disruption loss has a different scale, so the disruption gradient is rescaled to  dis_to_wm_ratio * ||grad_wm||  per step.
`reference_grad_norms` measures ||grad_sim|| / ||grad_wm|| for the existing YourTTS term, so that ratio can be copied and the A0 control and the
white-box arms spend their step on the same balance.
"""
import torch
import torch.nn.functional as F
import torchaudio


def load_ecapa_differentiable(device):
    from speechbrain.inference.speaker import EncoderClassifier
    return EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb",
                                          run_opts={"device": device})


def ecapa_embed(model, audio, native_sr):
    if audio.dim() == 3:
        audio = audio.squeeze(1)
    if native_sr != 16000:
        audio = torchaudio.functional.resample(audio, native_sr, 16000)
    return model.encode_batch(audio).squeeze(1)


def load_xvector(device):
    from speechbrain.inference.speaker import EncoderClassifier
    return EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb",
                                          run_opts={"device": device})


def load_wavlm_sv_differentiable(device):
    """WavLM-base-plus-sv called WITHOUT no_grad (H-MULTIVERIFIER: in the loss, so it can no longer be the held-out scorer)."""
    from transformers import WavLMForXVector
    m = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def wavlm_embed(model, audio, native_sr):
    """Same waveform normalisation as the HF feature extractor ((x - mean) / sqrt(var + 1e-7)), but differentiable. Unit-norm embedding [B, D]."""
    if audio.dim() == 3:
        audio = audio.squeeze(1)
    if native_sr != 16000:
        audio = torchaudio.functional.resample(audio, native_sr, 16000)
    x = (audio - audio.mean(-1, keepdim=True)) / torch.sqrt(audio.var(-1, keepdim=True, unbiased=False) + 1e-7)
    return F.normalize(model(input_values=x.to(torch.float32)).embeddings, dim=-1)


def make_verifiers(ecapa, xvec, wavlm):
    """The three in-loop verifiers of arm wens: list of (name, fn(wave, sample_rate) -> embedding [B, D]). Equal weights, no tuning."""
    return [("ecapa", lambda a, sr: ecapa_embed(ecapa, a, sr)),
            ("wavlm", lambda a, sr: wavlm_embed(wavlm, a, sr)),
            ("xvec", lambda a, sr: ecapa_embed(xvec, a, sr))]


class F5Attacker:
    """Differentiable re-implementation of CFM.sample for a single utterance (batch 1)."""

    def __init__(self, f5, steps=6, cfg=2.0, sway=-1.0, speed=1.0, checkpoint=True):
        import f5_tts.model.cfm as cfm_mod
        import f5_tts.model.utils as U
        import f5_tts.infer.utils_infer as UI
        self.f5, self.model, self.vocoder = f5, f5.ema_model, f5.vocoder
        self.steps, self.cfg, self.sway, self.speed = steps, cfg, sway, speed
        self.device = next(self.model.parameters()).device
        self.dtype = next(self.model.parameters()).dtype
        self.sr = getattr(UI, "target_sample_rate", 24000)
        self.hop = getattr(UI, "hop_length", 256)
        self.target_rms = 0.1
        self.convert = U.convert_char_to_pinyin
        self.list_str_to_idx, self.list_str_to_tensor = U.list_str_to_idx, U.list_str_to_tensor
        self.get_epss = getattr(cfm_mod, "get_epss_timesteps", None) or getattr(U, "get_epss_timesteps", None)
        for p in list(self.model.parameters()) + list(self.vocoder.parameters()):
            p.requires_grad_(False)
        self.model.eval()
        if checkpoint and hasattr(self.model.transformer, "checkpoint_activations"):
            self.model.transformer.checkpoint_activations = True

    # ---- input side -------------------------------------------------------------------------------------------------------------------
    def prompt_audio_24k(self, audio, native_sr=16000):
        a = audio.reshape(1, -1).to(self.device, self.dtype)
        rms = a.pow(2).mean().sqrt()
        a = a * torch.where(rms < self.target_rms, self.target_rms / (rms + 1e-9), torch.ones_like(rms))   # same rule as infer_batch_process
        return torchaudio.functional.resample(a, native_sr, self.sr)

    def prompt_mel(self, audio, native_sr=16000):
        mel = self.model.mel_spec(self.prompt_audio_24k(audio, native_sr))    # [1, d, n]
        return mel.permute(0, 2, 1)                                           # [1, n, d]

    def prepare(self, n_audio_24k_samples, ref_text, gen_text):
        """Text list + duration exactly as utils_infer.infer_batch_process computes them (speed 1.0)."""
        if len(ref_text[-1].encode("utf-8")) == 1:
            ref_text = ref_text + " "
        text_list = self.convert([ref_text + gen_text])
        ref_audio_len = n_audio_24k_samples // self.hop
        duration = ref_audio_len + int(ref_audio_len / len(ref_text.encode("utf-8")) * len(gen_text.encode("utf-8")) / self.speed)
        return text_list, duration

    # ---- the differentiable sampler ---------------------------------------------------------------------------------------------------
    def _timesteps(self):
        t = None
        if self.get_epss is not None:
            try:
                t = self.get_epss(self.steps, device=self.device, dtype=self.dtype)
            except Exception:
                t = None
        if t is None:
            t = torch.linspace(0, 1, self.steps + 1, device=self.device, dtype=self.dtype)
        if self.sway is not None:
            t = t + self.sway * (torch.cos(torch.pi / 2 * t) - 1 + t)
        return t

    def generate_mel(self, cond, text_list, duration, seed=None, noise=None):
        """cond [1, n, d] (may carry grad). Returns the full mel [1, dur, d] (prompt part replaced by cond, as CFM.sample does)."""
        m = self.model
        cond = cond.to(self.dtype)
        n, d = cond.shape[1], cond.shape[2]
        text = (self.list_str_to_idx(text_list, m.vocab_char_map) if getattr(m, "vocab_char_map", None) is not None
                else self.list_str_to_tensor(text_list)).to(self.device)
        text_len = int((text != -1).sum(dim=-1).max())
        dur = max(max(text_len, n) + 1, int(duration))
        cond_pad = F.pad(cond, (0, 0, 0, dur - n), value=0.0)
        cond_mask = torch.zeros(1, dur, 1, dtype=torch.bool, device=self.device)
        cond_mask[:, :n] = True
        step_cond = torch.where(cond_mask, cond_pad, torch.zeros_like(cond_pad))
        if noise is None:
            g = torch.Generator(device=self.device).manual_seed(int(seed))
            noise = torch.randn(dur, d, generator=g, device=self.device, dtype=self.dtype)
        y = noise.unsqueeze(0)
        t = self._timesteps()
        for k in range(self.steps):
            pred_cfg = m.transformer(x=y, cond=step_cond, text=text, time=t[k], mask=None, cfg_infer=True, cache=False)
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            v = pred + (pred - null_pred) * self.cfg
            y = y + (t[k + 1] - t[k]) * v
        return torch.where(cond_mask, cond_pad, y)

    def vocode(self, mel):
        """mel [1, frames, d] -> waveform [1, T] at 24 kHz (Vocos decode, gradient carrying)."""
        # Vocos.decode is decorated @torch.inference_mode(), so it returns tensors with NO autograd graph (G6 failure 2026-10-14).
        # Call the same two modules it calls (backbone -> head) outside inference mode.
        feats = mel.permute(0, 2, 1).to(torch.float32)
        if hasattr(self.vocoder, "backbone") and hasattr(self.vocoder, "head"):
            return self.vocoder.head(self.vocoder.backbone(feats))
        return self.vocoder.decode.__wrapped__(self.vocoder, feats)


# ---------------------------------------------------------------------------------------------------------------------------------------
def _random_start(like, epsilon, noise_seed):
    if noise_seed is not None:
        g = torch.Generator(device=like.device).manual_seed(int(noise_seed))      # same recipe as the H-ENSEMBLE generator wrapper -> paired
        return (torch.rand(like.shape, generator=g, device=like.device, dtype=like.dtype) * 2 - 1) * epsilon
    return (torch.rand_like(like) * 2 - 1) * epsilon


def envelope_bound(ref, epsilon, alpha, floor, win=321):
    """Per-sample L-inf bound, same shape as `ref` (the fixed watermarked waveform): epsilon * clamp(env^alpha, floor, 1), env = 20 ms RMS envelope / its max.
    Identical to composability_gen_shaped_audio.envelope_bound (H-BUDGET-SWEEP, notebook 01b variant)."""
    shape = ref.shape
    x2 = ref.detach().reshape(1, 1, -1).pow(2)
    k = torch.ones(1, 1, win, device=ref.device, dtype=ref.dtype) / win
    env = torch.sqrt(F.conv1d(x2, k, padding=win // 2) + 1e-12)[..., : x2.shape[-1]]
    env = env / env.max().clamp_min(1e-9)
    m = env.pow(alpha).clamp(min=floor, max=1.0) if alpha > 0 else torch.ones_like(env)
    return (epsilon * m).reshape(shape)


def whitebox_pgd(backbone, f5a, clean_audio, message, ref_text, gen_text, epsilon, step_size, n_steps, arm,
                 lambda_wm=1.0, dis_to_wm_ratio=1.0, ecapa_model=None, noise_seed=None, f5_seed=7000, verbose=False, verifiers=None,
                 shape_alpha=0.0, shape_floor=0.03):
    """Returns (recon_wm, perturbed_final, delta, diag). arm in {"wmel", "wspk", "wens"}.  wens: mean cosine similarity over several verifiers (list of (name, fn))."""
    from voicemark_losses import compute_ldec
    from safespeech_losses import compute_sim_disruption_loss
    assert arm in ("wmel", "wspk", "wens")
    if arm == "wspk":
        assert ecapa_model is not None
    if arm == "wens":
        assert verifiers and len(verifiers) >= 2, "arm wens needs the verifier list"

    with torch.no_grad():
        recon_wm = backbone.forward_full(clean_audio, message)["recon_wm"].detach()
        a24 = f5a.prompt_audio_24k(recon_wm)
        text_list, duration = f5a.prepare(a24.shape[-1], ref_text, gen_text)
        cond_ref = f5a.prompt_mel(recon_wm)
        n = cond_ref.shape[1]
        y_ref = f5a.generate_mel(cond_ref, text_list, duration, seed=f5_seed)[:, n:] if arm == "wmel" else None
        emb_clean = ecapa_embed(ecapa_model, clean_audio, 16000).detach() if arm == "wspk" else None
        embs_clean = [fn(clean_audio, 16000).detach() for _n, fn in verifiers] if arm == "wens" else None

    shaped = shape_alpha > 0                                                         # H-SHAPED-WENS: time-varying bound that follows the loudness envelope
    bound = envelope_bound(recon_wm, epsilon, shape_alpha, shape_floor) if shaped else epsilon
    step_vec = step_size * (bound / epsilon) if shaped else step_size
    delta = _random_start(recon_wm, bound, noise_seed).detach().requires_grad_(True)
    diag = {"dis_loss": [], "wm_loss": [], "gn_dis": [], "gn_wm": [], "skipped": 0, "mean_bound": float(bound.mean().item()) if shaped else float(epsilon)}

    for step in range(n_steps):
        perturbed = torch.clamp(recon_wm + delta, -1.0, 1.0)
        y_p = f5a.generate_mel(f5a.prompt_mel(perturbed), text_list, duration, seed=f5_seed)[:, n:]
        if arm == "wmel":
            dis_loss = -F.l1_loss(y_p, y_ref)                                        # minimising this MAXIMISES the divergence
        elif arm == "wspk":
            emb = ecapa_embed(ecapa_model, f5a.vocode(y_p), f5a.sr)
            dis_loss = compute_sim_disruption_loss(emb_clean, emb)
        else:
            wave = f5a.vocode(y_p)
            sims = [F.cosine_similarity(ec, fn(wave, f5a.sr)).mean() for ec, (_n, fn) in zip(embs_clean, verifiers)]
            dis_loss = torch.stack(sims).mean()                                      # mean cosine similarity of the rollout to the clean voice, three verifiers
            diag.setdefault("parts", []).append({n_: float(v.item()) for (n_, _f), v in zip(verifiers, sims)})
        g_dis = torch.autograd.grad(dis_loss, delta, retain_graph=False, create_graph=False)[0]
        del y_p

        # The first autograd.grad freed the saved buffers of `perturbed` (the clamp node), so the watermark branch needs its own copy of that node
        # (G7b failure 2026-10-14: "Trying to backward through the graph a second time").
        perturbed = torch.clamp(recon_wm + delta, -1.0, 1.0)
        with torch.backends.cudnn.flags(enabled=False):                              # same cuDNN-RNN workaround as pgd_perturb
            feat = backbone.model.st_model.forward_feature(perturbed)
            _l, chunk_logits = backbone.model.detector(feat)
            wm_loss = compute_ldec(chunk_logits, message)
        g_wm = torch.autograd.grad(wm_loss, delta, retain_graph=False, create_graph=False)[0]

        gn_dis, gn_wm = g_dis.norm().item(), g_wm.norm().item()
        grad = lambda_wm * g_wm + dis_to_wm_ratio * (gn_wm / max(gn_dis, 1e-12)) * g_dis
        diag["dis_loss"].append(float(dis_loss.item())); diag["wm_loss"].append(float(wm_loss.item()))
        diag["gn_dis"].append(gn_dis); diag["gn_wm"].append(gn_wm)
        if not torch.isfinite(grad).all():
            diag["skipped"] += 1
            print(f"  [whitebox step {step}] non-finite gradient, step skipped")
            continue
        if verbose:
            print(f"  [wb step {step}] {arm} dis_loss={dis_loss.item():.4f} wm_loss={wm_loss.item():.4f} |g_dis|={gn_dis:.3e} |g_wm|={gn_wm:.3e} "
                  f"(dis:wm raw {gn_dis / max(gn_wm, 1e-12):.2f}) delta_linf={delta.abs().max().item():.6f}"
                  + (f" parts={ {k: round(v, 3) for k, v in diag['parts'][-1].items()} }" if arm == "wens" else ""))
        if step == 0 and gn_dis == 0.0:
            print("  [whitebox] WARNING: zero disruption gradient at step 0 -- the path waveform -> F5 -> loss is broken; do not trust a full run.")
        with torch.no_grad():
            if shaped:
                delta = torch.max(torch.min(delta - step_vec * grad.sign(), bound), -bound)
            else:
                delta = torch.clamp(delta - step_size * grad.sign(), -epsilon, epsilon)
        delta = delta.detach().requires_grad_(True)

    with torch.no_grad():
        perturbed_final = torch.clamp(recon_wm + delta, -1.0, 1.0)
    return recon_wm, perturbed_final, delta.detach(), diag


def reference_grad_norms(backbone, surrogate, clean_audio, message, text, epsilon, noise_seed):
    """||grad_sim|| and ||grad_wm|| of the EXISTING YourTTS objective at a seeded random start (for the dis_to_wm_ratio calibration)."""
    from voicemark_losses import compute_ldec
    from safespeech_losses import compute_sim_disruption_loss
    with torch.no_grad():
        recon_wm = backbone.forward_full(clean_audio, message)["recon_wm"].detach()
        emb_clean = surrogate.compute_speaker_embedding(clean_audio).detach()
    delta = _random_start(recon_wm, epsilon, noise_seed).detach().requires_grad_(True)
    perturbed = torch.clamp(recon_wm + delta, -1.0, 1.0)
    torch.manual_seed(int(noise_seed)); torch.cuda.manual_seed_all(int(noise_seed))
    cloned = surrogate.clone_voice(perturbed, text=text)
    sim_loss = compute_sim_disruption_loss(emb_clean, surrogate.compute_speaker_embedding(cloned))
    with torch.backends.cudnn.flags(enabled=False):
        feat = backbone.model.st_model.forward_feature(perturbed)
        _l, chunk_logits = backbone.model.detector(feat)
        wm_loss = compute_ldec(chunk_logits, message)
    g_sim = torch.autograd.grad(sim_loss, delta, retain_graph=True)[0]
    g_wm = torch.autograd.grad(wm_loss, delta)[0]
    return g_sim.norm().item(), g_wm.norm().item()
