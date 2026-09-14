"""
src/eval/layer_reweight.py

CARRIER-REWEIGHT (2026-09-14). Wraps the real msg_processor and scales each RVQ
layer's contribution to acoustic_wm (= sum of 7 per-layer msg_processor calls,
confirmed by reading backbone.py directly) before it's summed. Same call-order
assumption already relied on by LayerRecorder/LatentDeltaMsgProcessor elsewhere in
this project (7 calls, RVQ layers 2-8 in ascending order, verified via
carrier_probe.py's own diagnostic n_calls==7 check). No retraining: only changes
the inference-time magnitude of each layer's already-learned embedding on the
existing pretrained checkpoint.

2026-09-14 fix: the original version was a plain object, not an nn.Module.
backbone.model.msg_processor is a registered submodule, and torch's
nn.Module.__setattr__ refuses to assign a non-Module/None over a slot that is
already a submodule (TypeError: cannot assign ... as child module 'msg_processor'
(torch.nn.Module or None expected)). Fixed by subclassing nn.Module and moving the
scaling logic into forward() -- nn.Module.__call__ already routes to forward()
with the same (x, message) signature the rest of this project calls msg_processor
with, so no call-site changes anywhere else are needed.
"""
import torch.nn as nn


class LayerReweightMsgProcessor(nn.Module):
    def __init__(self, base_msg_processor, layer_scales):
        super().__init__()
        assert len(layer_scales) == 7, f"need 7 scales (layers 2-8), got {len(layer_scales)}"
        self.base = base_msg_processor
        self.layer_scales = layer_scales
        self.call_idx = 0

    def reset(self):
        self.call_idx = 0

    def forward(self, x, message):
        out = self.base(x, message)
        scale = self.layer_scales[self.call_idx % 7]
        self.call_idx += 1
        if scale != 1.0:
            out = out * scale
        return out
