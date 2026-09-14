"""
apply_carrierprobe_importfix_patch.py

Removes carrier_probe.py's `from disruption_pgd import build_backbone,
random_message` -- disruption_pgd.py imports surrogate_vc.py at module scope,
which imports coqui-TTS, which needs a newer transformers than this notebook's
pinned transformers==4.41.2 (required for Amphion's llama_nar.py). Just
importing carrier_probe.py crashes with "cannot import name
'is_torch_greater_or_equal'" before any MaskGCT code runs. Fix: duplicate the
two torch-only functions actually needed (disruption_pgd's build_backbone
collapses to exactly this when include_ffn=False, the only way carrier_probe.py
ever calls it), same convention cloner_watermark_eval.py already uses for the
same reason.
"""
import os, sys

_DEFAULT_TARGET = "src/eval/carrier_probe.py"
TARGET = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_TARGET
if not os.path.exists(TARGET) and os.path.exists(os.path.basename(TARGET)):
    TARGET = os.path.basename(TARGET)

OLD = 'from disruption_pgd import build_backbone, random_message\n'
NEW = (
    '# 2026-09-14 fix: avoid disruption_pgd.py -- it imports surrogate_vc.py ->\n'
    '# coqui-TTS at module scope, which conflicts with transformers==4.41.2 (pinned\n'
    '# for Amphion/MaskGCT). Duplicate the two torch-only functions instead, same\n'
    '# convention cloner_watermark_eval.py already uses for the same reason.\n'
    'from backbone import VoiceMarkBackbone\n'
    'from adapters import apply_lora_adapters\n'
    '\n'
    '\n'
    'def build_backbone(checkpoint_path, r=8, alpha=16, include_ffn=False, capacity_lora_r=32):\n'
    '    backbone = VoiceMarkBackbone()\n'
    '    apply_lora_adapters(backbone, r=r, alpha=alpha)\n'
    '    if checkpoint_path is not None:\n'
    '        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)\n'
    '        backbone.model.load_state_dict(ckpt["lora_state_dict"], strict=False)\n'
    '        print(f"[build_backbone] loaded {checkpoint_path} (epoch {ckpt.get(\'epoch\')})")\n'
    '    else:\n'
    '        print("[build_backbone] No checkpoint given -- LoRA at zero-init (== pretrained VoiceMark).")\n'
    '    backbone.model.eval()\n'
    '    for p in backbone.model.parameters():\n'
    '        p.requires_grad_(False)\n'
    '    return backbone\n'
    '\n'
    '\n'
    'def random_message(nbits, batch_size, device, seed):\n'
    '    gen = torch.Generator(device=device).manual_seed(seed)\n'
    '    return torch.randint(0, 2, (batch_size, nbits), generator=gen, device=device)\n'
)

def _apply(path, old, new):
    if not os.path.exists(path):
        raise SystemExit(f"ABORT: {path} not found (cwd={os.getcwd()}).")
    content = open(path, "r").read()
    if "def build_backbone(" in content:
        print(f"[apply_carrierprobe_importfix_patch] {path} already patched -- skipping.")
        return
    count = content.count(old)
    if count != 1:
        raise SystemExit(f"ABORT: anchor occurs {count} times in {path} (expected 1). Nothing written.\n{old!r}")
    with open(path, "w") as f:
        f.write(content.replace(old, new, 1))
    print(f"[apply_carrierprobe_importfix_patch] Patched {path}.")

if __name__ == "__main__":
    _apply(TARGET, OLD, NEW)
