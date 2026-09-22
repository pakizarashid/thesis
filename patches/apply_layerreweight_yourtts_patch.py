"""
apply_layerreweight_yourtts_patch.py -- adds --layer_scales to gen_samples_yourtts.py.
"""
import os, sys

_DEFAULT_TARGET = "src/eval/gen_samples_yourtts.py"
TARGET = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_TARGET
if not os.path.exists(TARGET) and os.path.exists(os.path.basename(TARGET)):
    TARGET = os.path.basename(TARGET)

OLD = (
    '    p.add_argument("--crop_seconds", type=float, default=20.0,\n'
    '                    help="Bumped up from the project\'s usual 3.0s default -- see the "\n'
    '                         "CROP_SECONDS note in this file\'s docstring. Only safe to lower back "\n'
    '                         "to 3.0 if --no_use_own_transcript is also passed.")\n'
    '    args = p.parse_args()\n'
    '\n'
    '    device = "cuda" if torch.cuda.is_available() else "cpu"\n'
    '    backbone = build_backbone(args.checkpoint, args.lora_r, args.lora_alpha, False, 32)\n'
    '    print("[gen_samples_yourtts] loading YourTTS surrogate...")\n'
    '    surrogate = load_yourtts_surrogate(device=device)\n'
)
NEW = (
    '    p.add_argument("--crop_seconds", type=float, default=20.0,\n'
    '                    help="Bumped up from the project\'s usual 3.0s default -- see the "\n'
    '                         "CROP_SECONDS note in this file\'s docstring. Only safe to lower back "\n'
    '                         "to 3.0 if --no_use_own_transcript is also passed.")\n'
    '    p.add_argument("--layer_scales", type=str, default="1,1,1,1,1,1,1",\n'
    '                    help="CARRIER-REWEIGHT (2026-09-14): 7 comma-separated floats, one per "\n'
    '                         "RVQ layer 2-8 in order, scaling that layer\'s msg_processor output "\n'
    '                         "before summing into acoustic_wm. No retraining. Default all-1.0 "\n'
    '                         "leaves behavior exactly unchanged.")\n'
    '    args = p.parse_args()\n'
    '\n'
    '    device = "cuda" if torch.cuda.is_available() else "cpu"\n'
    '    backbone = build_backbone(args.checkpoint, args.lora_r, args.lora_alpha, False, 32)\n'
    '    layer_scales = [float(s) for s in args.layer_scales.split(",")]\n'
    '    assert len(layer_scales) == 7, f"--layer_scales needs 7 floats, got {len(layer_scales)}"\n'
    '    if any(s != 1.0 for s in layer_scales):\n'
    '        from layer_reweight import LayerReweightMsgProcessor\n'
    '        backbone.model.msg_processor = LayerReweightMsgProcessor(backbone.model.msg_processor, layer_scales)\n'
    '        print(f"[layer_reweight] ACTIVE: scales={layer_scales} (layer2..layer8 in order)")\n'
    '    print("[gen_samples_yourtts] loading YourTTS surrogate...")\n'
    '    surrogate = load_yourtts_surrogate(device=device)\n'
)

def _apply(path, old, new):
    if not os.path.exists(path):
        raise SystemExit(f"ABORT: {path} not found (cwd={os.getcwd()}).")
    content = open(path, "r").read()
    if "--layer_scales" in content:
        print(f"[apply_layerreweight_yourtts_patch] {path} already patched -- skipping.")
        return
    count = content.count(old)
    if count != 1:
        raise SystemExit(f"ABORT: anchor occurs {count} times in {path} (expected 1). Nothing written.\n{old[:300]!r}")
    with open(path, "w") as f:
        f.write(content.replace(old, new, 1))
    print(f"[apply_layerreweight_yourtts_patch] Patched {path}.")

if __name__ == "__main__":
    _apply(TARGET, OLD, NEW)
