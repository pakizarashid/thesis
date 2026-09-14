"""
src/eval/layer_reweight.py

CARRIER-REWEIGHT (2026-09-14). Wraps the real msg_processor and scales each RVQ
layer's contribution to acoustic_wm (= sum of 7 per-layer msg_processor calls,
confirmed by reading backbone.py directly) before it's summed. Same call-order
assumption already relied on by LayerRecorder/LatentDeltaMsgProcessor elsewhere in
this project (7 calls, RVQ layers 2-8 in ascending order, verified via
carrier_probe.py's own diagnostic n_calls==7 check). No retraining: only changes
the inference-time magnitude of each layer's already-learned embedding on the
existing pretrained checkpoint. Motivated by CARRIER-PROBE's finding
(carrier-fragility-pivot-2026-09-12.md §7) that layer 2 is by far the most
architecture-sensitive layer (2.85x spread vs. 1.36x for layer 8) -- this tests
whether down-weighting it improves clone-ACC on low-bandwidth cloners without
hurting source ACC or the already-strong high-bandwidth architectures.
"""


class LayerReweightMsgProcessor:
    def __init__(self, base_msg_processor, layer_scales):
        assert len(layer_scales) == 7, f"need 7 scales (layers 2-8), got {len(layer_scales)}"
        self.base = base_msg_processor
        self.layer_scales = layer_scales
        self.call_idx = 0

    def reset(self):
        self.call_idx = 0

    def __call__(self, x, message):
        out = self.base(x, message)
        scale = self.layer_scales[self.call_idx % 7]
        self.call_idx += 1
        if scale != 1.0:
            out = out * scale
        return out
