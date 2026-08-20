"""
src/models/adapters.py

LoRA adapters for VoiceMark's msg_processor (WMEmbedder) and detector
(WMDetector). Both are frozen (see backbone.py); LoRA reparameterizes their
attention projection weights with a low-rank additive delta, so training starts
numerically identical to the pretrained checkpoint (B is zero-initialized) and
drifts gently -- important given Stage 2 adds a much harder loss on top.

Attaches to:
  - msg_processor.transformer_decoder.layers[i].self_attn   (in_proj, out_proj)
  - msg_processor.transformer_decoder.layers[i].multihead_attn (in_proj, out_proj)
  - detector.transformer.layers[i].self_attn                (in_proj, out_proj)

nn.MultiheadAttention packs Q/K/V into a single in_proj_weight of shape
[3*embed_dim, embed_dim] (self-attention) or in_proj may be split for
cross-attention -- we handle both by wrapping the whole in_proj_weight as one
low-rank delta rather than splitting Q/K/V separately. This is simpler and
still standard practice; splitting Q/K/V into separate LoRA deltas is a
reasonable ablation later but adds complexity we don't need for Stage 1.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinearDelta(nn.Module):
    """
    Computes a low-rank delta: delta_W = alpha/r * (B @ A), added to a frozen
    base weight at forward time. Does not modify the base weight in place.

    Usage: wrap an existing frozen nn.Linear-like weight tensor. The caller
    (LoRAMultiheadAttentionWrapper below) is responsible for adding this
    delta to the base weight during forward.
    """

    def __init__(self, in_features: int, out_features: int, r: int = 8, alpha: int = 16):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.A = nn.Parameter(torch.zeros(r, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, r))
        # Standard LoRA init: A ~ kaiming, B = zero => delta starts at exactly 0
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self) -> torch.Tensor:
        """Returns the [out_features, in_features] delta weight matrix."""
        return self.scaling * (self.B @ self.A)


class LoRAMultiheadAttentionWrapper(nn.Module):
    """
    Wraps a frozen nn.MultiheadAttention module, adding LoRA deltas to its
    in_proj_weight and out_proj.weight. The wrapped module's own parameters
    must already be frozen (requires_grad=False) by the caller -- this wrapper
    does not freeze anything itself, it only adds new trainable parameters and
    reimplements the forward pass to inject them.
    """

    def __init__(self, mha: nn.MultiheadAttention, r: int = 8, alpha: int = 16):
        super().__init__()
        self.mha = mha  # frozen, kept as a submodule so its base weights travel with state_dict
        embed_dim = mha.embed_dim

        assert mha.in_proj_weight is not None, (
            "LoRAMultiheadAttentionWrapper assumes unified in_proj_weight "
            "(the default for nn.MultiheadAttention when kdim==vdim==embed_dim, "
            "which holds for both msg_processor and detector's attention layers)."
        )
        self.in_proj_lora = LoRALinearDelta(embed_dim, 3 * embed_dim, r=r, alpha=alpha)
        self.out_proj_lora = LoRALinearDelta(embed_dim, embed_dim, r=r, alpha=alpha)

    def __getattr__(self, name):
        # PyTorch's TransformerEncoder/Decoder container-level code introspects
        # attributes like `.batch_first`, `.num_heads` etc. directly on the
        # self_attn submodule for bookkeeping (e.g. _get_seq_len), bypassing
        # forward() entirely for these lookups. Delegate anything this wrapper
        # doesn't define itself to the wrapped nn.MultiheadAttention.
        #
        # NOTE: this delegation is safe for the actual ATTENTION COMPUTATION
        # too, because PyTorch's fused/nested-tensor fast path -- which would
        # bypass this wrapper's forward() and use the raw frozen weights
        # directly, silently ignoring the LoRA delta -- is already disabled for
        # these particular layers (confirmed by the earlier
        # "use_nested_tensor is False because ... num_heads is odd" warning at
        # construction time). The actual forward call always routes through
        # this class's forward() below, where the LoRA delta is applied.
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        modules = self.__dict__.get("_modules", {})
        mha = modules.get("mha", None)
        if mha is not None and hasattr(mha, name):
            return getattr(mha, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def forward(self, query, key, value, **kwargs):
        # Compose base + LoRA delta weights, then call functional MHA directly
        # (bypassing self.mha.forward, which would use the frozen weights only).
        in_proj_weight = self.mha.in_proj_weight + self.in_proj_lora()
        in_proj_bias = self.mha.in_proj_bias
        out_proj_weight = self.mha.out_proj.weight + self.out_proj_lora()
        out_proj_bias = self.mha.out_proj.bias

        # F.multi_head_attention_forward has NO concept of batch_first -- it
        # always expects (seq, batch, embed). nn.MultiheadAttention.forward
        # normally does this transpose internally before calling the functional
        # form, then transposes the output back. We replicate that here since
        # we're calling the functional form directly.
        batch_first = self.mha.batch_first
        is_batched = query.dim() == 3
        if batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (t.transpose(1, 0) for t in (query, key))
                    value = key
            else:
                query, key, value = (t.transpose(1, 0) for t in (query, key, value))

        attn_output, attn_output_weights = F.multi_head_attention_forward(
            query, key, value,
            embed_dim_to_check=self.mha.embed_dim,
            num_heads=self.mha.num_heads,
            in_proj_weight=in_proj_weight,
            in_proj_bias=in_proj_bias,
            bias_k=self.mha.bias_k,
            bias_v=self.mha.bias_v,
            add_zero_attn=self.mha.add_zero_attn,
            dropout_p=self.mha.dropout,
            out_proj_weight=out_proj_weight,
            out_proj_bias=out_proj_bias,
            training=self.mha.training,
            **kwargs,
        )

        if batch_first and is_batched:
            attn_output = attn_output.transpose(1, 0)

        return attn_output, attn_output_weights


class LoRALinearWrapper(nn.Module):
    """
    Wraps a frozen nn.Linear, adding a LoRA delta to its weight. Same
    zero-init/no-op-at-construction property as LoRAMultiheadAttentionWrapper.

    ADDED to test the capacity-limitation hypothesis raised in
    STAGE2_WRITEUP.md Section 7: the original LoRA wrapping (above) only
    touches nn.MultiheadAttention's in/out projections inside
    transformer_decoder / transformer layers. A standard
    nn.TransformerDecoderLayer / nn.TransformerEncoderLayer also has a
    feedforward block (linear1 -> activation -> linear2), typically with MORE
    parameters than the attention projections, that was previously entirely
    frozen and untouched by any adapter. If the capacity-limitation hypothesis
    is correct, wrapping these FFN layers too (see `include_ffn` below) is the
    most direct, cheap test of it -- cheaper than fully unfreezing the module
    or switching to a non-adapter perturbation mechanism.
    """

    def __init__(self, linear: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.linear = linear  # frozen, kept as submodule so base weights travel with state_dict
        # Named "linear_lora" (not "delta") so its parameter names contain the
        # same "_lora." substring the rest of this file's bookkeeping greps
        # for (see apply_lora_adapters's non_lora_trainable sanity check, and
        # any external checkpoint-filtering code that does `if "_lora." in k`).
        self.linear_lora = LoRALinearDelta(linear.in_features, linear.out_features, r=r, alpha=alpha)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        modules = self.__dict__.get("_modules", {})
        linear = modules.get("linear", None)
        if linear is not None and hasattr(linear, name):
            return getattr(linear, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.linear.weight + self.linear_lora()
        return F.linear(x, weight, self.linear.bias)


def _replace_mha_with_lora(module: nn.Module, r: int, alpha: int, prefix: str = "",
                            include_ffn: bool = False, ffn_names=("linear1", "linear2")):
    """
    Recursively walks `module`, replacing every nn.MultiheadAttention child
    with a LoRAMultiheadAttentionWrapper around it. Returns the list of
    (name, wrapper) pairs created, for parameter-group bookkeeping.

    If include_ffn=True, also wraps any direct nn.Linear children named in
    `ffn_names` (the default 'linear1'/'linear2' matches PyTorch's
    TransformerEncoderLayer/TransformerDecoderLayer feedforward block) with
    LoRALinearWrapper. See LoRALinearWrapper's docstring for why this exists.
    """
    created = []
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.MultiheadAttention):
            wrapper = LoRAMultiheadAttentionWrapper(child, r=r, alpha=alpha)
            setattr(module, name, wrapper)
            created.append((full_name, wrapper))
        elif include_ffn and isinstance(child, nn.Linear) and name in ffn_names:
            wrapper = LoRALinearWrapper(child, r=r, alpha=alpha)
            setattr(module, name, wrapper)
            created.append((full_name, wrapper))
        else:
            created.extend(_replace_mha_with_lora(child, r, alpha, prefix=full_name,
                                                    include_ffn=include_ffn, ffn_names=ffn_names))
    return created


def apply_lora_adapters(backbone, r: int = 8, alpha: int = 16, targets=("msg_processor", "detector"),
                         include_ffn: bool = False, ffn_r: int = None, ffn_targets: tuple = None):
    """
    Applies LoRA wrapping to msg_processor and/or detector on a VoiceMarkBackbone
    instance (backbone.model.msg_processor / backbone.model.detector). Call this
    AFTER VoiceMarkBackbone.__init__ (which freezes everything). Returns the list
    of newly created LoRA parameter names, for building your optimizer's
    parameter group.

    include_ffn: if True, also LoRA-wraps the transformer feedforward Linear
        layers (see LoRALinearWrapper). Backward-compatible default (False)
        reproduces the exact original attention-only behavior.
    ffn_r: optional rank override applied to every LoRA module (attention AND
        FFN alike) on targets listed in `ffn_targets` (defaults to `r` if
        None). Useful for a targeted capacity test -- e.g. keep detector's
        LoRA at r=8 (it doesn't affect recon_wm / the disruption loss at all,
        only msg_processor does) while giving msg_processor a much larger
        rank specifically to test the capacity hypothesis cheaply, without
        inflating detector's unrelated parameter count. NOTE: this overrides
        `r` for the whole target, not just its FFN modules -- there is
        currently no separate attention-only-vs-FFN-only rank control within
        a single target.
    ffn_targets: optional subset of `targets` to apply include_ffn to (e.g.
        ("msg_processor",) only). Defaults to `targets` if None. Restricting
        this to msg_processor only is recommended for disruption-capacity
        experiments -- detector's FFN capacity is irrelevant to recon_wm.

    Example:
        backbone = VoiceMarkBackbone()
        lora_params = apply_lora_adapters(backbone, r=8, alpha=16)
        optimizer = torch.optim.Adam(
            [p for n, p in backbone.named_parameters() if p.requires_grad],
            lr=5e-5,
        )
    """
    if ffn_r is None:
        ffn_r = r
    if ffn_targets is None:
        ffn_targets = targets

    all_created = []
    for target_name in targets:
        submodule = getattr(backbone.model, target_name, None)
        if submodule is None:
            raise ValueError(f"No submodule named '{target_name}' on backbone.model")
        this_include_ffn = include_ffn and (target_name in ffn_targets)
        this_r = ffn_r if this_include_ffn else r
        created = _replace_mha_with_lora(submodule, r=this_r, alpha=alpha, prefix=target_name,
                                          include_ffn=this_include_ffn)
        all_created.extend(created)

    # New LoRA parameters default to CPU regardless of where the frozen backbone
    # already lives (VoiceMarkBackbone.__init__ moves the model to GPU before
    # this function is ever called). Move each wrapper to match.
    try:
        backbone_device = next(backbone.model.parameters()).device
    except StopIteration:
        backbone_device = torch.device("cpu")
    for _, wrapper in all_created:
        wrapper.to(backbone_device)

    # LoRA params are trainable by construction (nn.Parameter defaults to
    # requires_grad=True); everything else on backbone.model should already be
    # frozen by VoiceMarkBackbone.freeze_backbone(). Sanity-check that here.
    trainable = [n for n, p in backbone.model.named_parameters() if p.requires_grad]
    non_lora_trainable = [n for n in trainable if "_lora." not in n]
    if non_lora_trainable:
        print(
            f"[apply_lora_adapters] WARNING - found {len(non_lora_trainable)} trainable "
            f"params that are NOT LoRA params (backbone wasn't fully frozen before "
            f"applying adapters?): {non_lora_trainable[:5]}{'...' if len(non_lora_trainable) > 5 else ''}"
        )

    total_lora_params = sum(
        p.numel() for n, p in backbone.model.named_parameters()
        if p.requires_grad and "_lora." in n
    )
    print(
        f"[apply_lora_adapters] Wrapped {len(all_created)} attention modules across "
        f"{targets} with LoRA (r={r}, alpha={alpha}). "
        f"Total trainable LoRA parameters: {total_lora_params:,}"
    )

    return all_created


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from backbone import VoiceMarkBackbone

    backbone = VoiceMarkBackbone()
    print(f"Trainable params before LoRA: {sum(p.numel() for p in backbone.model.parameters() if p.requires_grad)}")

    apply_lora_adapters(backbone, r=8, alpha=16)

    n_trainable = sum(p.numel() for p in backbone.model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in backbone.model.parameters())
    print(f"Trainable params after LoRA: {n_trainable:,} / {n_total:,} total ({100*n_trainable/n_total:.3f}%)")

    # Forward-pass sanity check: watermarked output should be numerically close
    # to the base model's output at init, since LoRA deltas start at zero.
    dummy_audio = torch.randn(1, 1, 16000, device=backbone.device)
    dummy_msg = torch.randint(0, 2, (1, 16), device=backbone.device)
    with torch.no_grad():
        out = backbone.forward(dummy_audio, dummy_msg)
    print(f"Forward pass OK. recon_wm shape: {out['recon_wm'].shape}")
