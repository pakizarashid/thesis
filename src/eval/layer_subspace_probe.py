"""
src/eval/layer_subspace_probe.py

ROUTE 1 of the post-Layer-1 plan: find, empirically, WHICH PART of the speaker
representation actually carries a watermark across the zero-shot cloning
boundary.

Motivation (from the 2026-09-09 XTTS result). Watermark ACC on an XTTS clone of
unprotected watermarked audio measured 0.5687 at n=100 -- statistically far
above chance (p ~ 2e-8) but practically weak (a single utterance is not
attributable). So a real channel exists from protected audio, through a
zero-shot cloner, to the detector. Nobody has yet asked WHERE that channel is.

VoiceMark spreads the message uniformly across RVQ layers 2-8 (st_model
computes `acoustic_wm = sum(msg_processor(x, message) for x in subset)`, one
msg_processor call per layer in the subset). A zero-shot cloner does not copy
the waveform -- it extracts a speaker embedding and resynthesizes -- so there
is no reason to expect every layer's contribution to survive that bottleneck
equally. If some layers survive markedly better than others, concentrating the
payload there is an immediate improvement for almost no cost, and it tells
Route 2's joint training WHERE to put its capacity.

METHOD. Reuses the call-counting msg_processor wrapper mechanism already built
and empirically validated for disruption_pgd_latent.py (its call-count
assumption check passed at 0.0000 relative error -- that mechanism is the
instrument this probe is built on). Here the wrapper routes the message into
exactly ONE msg_processor call (one RVQ layer) and passes every other layer
through UNWATERMARKED, concentrating the entire payload in a single layer.

TWO numbers per layer, and the second is meaningless without the first:
  - acc_source: watermark ACC on the watermarked audio itself, NOT cloned.
    This is that layer's CAPACITY -- can it carry a recoverable payload at all?
  - acc_clone:  watermark ACC on a real zero-shot clone of that audio.
    This is that layer's SURVIVAL through the cloning bottleneck.
A layer showing acc_clone at chance is uninformative if acc_source is also at
chance (the layer simply cannot carry the mark alone); it is a genuine negative
only if acc_source was high. Hence both, always, per layer.

  survival_ratio = (acc_clone - 0.5) / (acc_source - 0.5)
i.e. what fraction of the recoverable signal crosses the cloning boundary.
Reported only where acc_source is meaningfully above chance.

HONEST LIMITATION, state this in any writeup. VoiceMark's detector was trained
to read a mark embedded across ALL layers at once. Restricting embedding to one
layer is off-distribution for the detector, so a low acc_source may mean "the
detector was never trained to read from this layer alone" rather than "this
layer cannot carry a mark." This probe therefore measures where the PRETRAINED
system already places recoverable signal -- it is exploratory evidence for
where Route 2's training should focus, not a capacity bound on the layers
themselves.

DISCIPLINE (same as every other script here): run --diagnostic FIRST. It
verifies the call count matches the expected layer count, checks that the
pass-through approximation is sound, and confirms the all-layers control
reproduces the known baseline ACC -- so a broken assumption costs seconds, not
a multi-hour XTTS sweep.

Usage:
    # 1. ALWAYS first -- seconds, validates the mechanism before any sweep:
    python src/eval/layer_subspace_probe.py \\
        --checkpoint ./checkpoints/stage1_final_scaleup_recalibrated/recalibrated_final.pt \\
        --diagnostic

    # 2. Capacity only (no cloning -- fast, ~1 min): which layers can carry a mark?
    python src/eval/layer_subspace_probe.py \\
        --checkpoint ./checkpoints/stage1_final_scaleup_recalibrated/recalibrated_final.pt \\
        --skip_clone --n_utterances 25 \\
        --output results/results_layerprobe_capacity_n25.json

    # 3. Full probe with real zero-shot cloning (slow -- XTTS is autoregressive):
    python src/eval/layer_subspace_probe.py \\
        --checkpoint ./checkpoints/stage1_final_scaleup_recalibrated/recalibrated_final.pt \\
        --cloner xtts --n_utterances 25 \\
        --output results/results_layerprobe_xtts_n25.json
"""

import os
import sys
import json
import argparse
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.insert(0, os.path.dirname(__file__))

from librispeech import LibriSpeechSubset, collate_librispeech
from surrogate_vc import load_yourtts_surrogate
from disruption_pgd import build_backbone, random_message, detect_acc, compute_sim


class SingleLayerMsgProcessor:
    """
    Wraps the real msg_processor to concentrate the entire watermark payload in
    ONE RVQ layer.

    st_model computes, per its documented internals (see backbone.py):
        acoustic_wm = sum(msg_processor(x, message) for x in subset)
    calling msg_processor once per layer in the subset, in order. This wrapper
    counts those calls and:
      - on call index == target_call_idx: returns the REAL msg_processor output
        (the watermark is embedded in that layer), or
      - on every other call: returns x unchanged (that layer contributes its
        acoustic content but carries no message).
    With target_call_idx=None every call is passed to the real msg_processor,
    i.e. normal VoiceMark operation -- the control condition.

    Deliberately a plain Python object, NOT an nn.Module: it must not register
    as a submodule or appear in state_dict/parameters(). Same design as
    disruption_pgd_latent.py's LatentDeltaMsgProcessor, whose call-count
    assumption was empirically verified (0.0000 relative error).

    reset() MUST be called before each forward pass -- the counter is stateful
    and a stale count silently probes the wrong layer.
    """

    def __init__(self, real_mp, target_call_idx=None):
        self.real_mp = real_mp
        self.target_call_idx = target_call_idx
        self.call_count = 0
        self.calls_seen_last_pass = 0

    def reset(self):
        self.calls_seen_last_pass = self.call_count
        self.call_count = 0

    def __call__(self, *args, **kwargs):
        idx = self.call_count
        self.call_count += 1
        if self.target_call_idx is None or idx == self.target_call_idx:
            return self.real_mp(*args, **kwargs)
        # Pass the layer through unwatermarked. args[0] is the per-layer
        # acoustic tensor x; returning it keeps the audio reconstruction
        # intact while contributing no message to this layer.
        return args[0]

    def __getattr__(self, name):
        # Only reached when normal attribute lookup fails. Guard real_mp
        # explicitly or this recurses infinitely before __init__ sets it.
        if name == "real_mp":
            raise AttributeError(name)
        return getattr(self.real_mp, name)


def embed_single_layer(backbone, clean_audio, message, layer_idx):
    """
    Returns (recon_wm, n_msg_processor_calls) with the message embedded in
    exactly one RVQ layer (layer_idx), or in all layers when layer_idx is None.
    """
    st_model = backbone.model.st_model
    wrapper = SingleLayerMsgProcessor(backbone.model.msg_processor, target_call_idx=layer_idx)
    wrapper.reset()
    with torch.no_grad():
        _o, o_wm, _acoustic, _acoustic_wm = st_model(
            clean_audio, msg_processor=wrapper, message=message
        )
    return o_wm, wrapper.call_count


def clone_audio(cloner_kind, xtts, surrogate, audio, text, tmp_dir, tag):
    """One zero-shot clone, from whichever cloner was selected."""
    if cloner_kind == "xtts":
        from xtts_transfer_eval import xtts_clone
        return xtts_clone(xtts, audio, text, tmp_dir, tag)
    with torch.no_grad():
        return surrogate.clone_voice(audio, text=text)


def run_probe(backbone, surrogate, xtts, eval_loader, device, args, layer_ids):
    """
    For each layer id (None == all-layers control), measure capacity
    (acc on the watermarked audio) and, unless --skip_clone, survival
    (acc on a real clone of it).
    """
    per_layer = {}

    for layer_idx in layer_ids:
        label = "all" if layer_idx is None else str(layer_idx)
        print(f"\n{'=' * 60}\nLayer {label}"
              f"{' (control: normal VoiceMark, all layers)' if layer_idx is None else ''}"
              f"\n{'=' * 60}")

        accs_source, accs_clone, sims_clone = [], [], []

        for i, batch in enumerate(eval_loader):
            if i >= args.n_utterances:
                break
            clean_audio = batch["waveform"].to(device)
            message = random_message(16, clean_audio.shape[0], device, seed=123 + i)

            recon_wm, n_calls = embed_single_layer(backbone, clean_audio, message, layer_idx)
            acc_source = detect_acc(backbone, recon_wm, message)
            accs_source.append(acc_source)

            line = f"  [{i}] layer={label} acc_source={acc_source:.4f}"

            if not args.skip_clone:
                cloned = clone_audio(args.cloner, xtts, surrogate, recon_wm,
                                     args.surrogate_text, args.tmp_dir, f"L{label}_{i}")
                acc_clone = detect_acc(backbone, cloned, message)
                sim_clone = compute_sim(surrogate, clean_audio, cloned)
                accs_clone.append(acc_clone)
                sims_clone.append(sim_clone)
                line += f" acc_clone={acc_clone:.4f} sim_clone={sim_clone:.4f}"

            print(line, flush=True)

        entry = {
            "layer": label,
            "acc_source_mean": sum(accs_source) / len(accs_source),
            "acc_source_values": accs_source,
            "n_utterances": len(accs_source),
        }
        if accs_clone:
            entry["acc_clone_mean"] = sum(accs_clone) / len(accs_clone)
            entry["sim_clone_mean"] = sum(sims_clone) / len(sims_clone)
            entry["acc_clone_values"] = accs_clone
            entry["sim_clone_values"] = sims_clone
            # Fraction of this layer's recoverable signal that crosses the
            # cloning bottleneck. Undefined when the layer carries nothing.
            src_margin = entry["acc_source_mean"] - 0.5
            if src_margin > 0.02:
                entry["survival_ratio"] = (entry["acc_clone_mean"] - 0.5) / src_margin
            else:
                entry["survival_ratio"] = None
                entry["survival_ratio_note"] = (
                    "undefined -- acc_source is at/near chance, so this layer carries no "
                    "recoverable payload on its own and its clone number says nothing "
                    "about survival (see the script's HONEST LIMITATION note)"
                )

        per_layer[label] = entry

        if accs_clone:
            sr = entry["survival_ratio"]
            sr_str = "n/a (layer carries no payload)" if sr is None else f"{sr:.3f}"
            tail = (f"  acc_clone={entry['acc_clone_mean']:.4f}"
                    f"  sim_clone={entry['sim_clone_mean']:.4f}"
                    f"  survival_ratio={sr_str}")
        else:
            tail = "  (capacity only, --skip_clone)"
        print(f"\n  Layer {label} SUMMARY: acc_source={entry['acc_source_mean']:.4f}{tail}")

        # Incremental save after EVERY layer. A crash mid-sweep (or a Kaggle
        # session timeout) must never cost the layers already measured -- the
        # same lesson the AudioPure checkpoint crash taught this project.
        if args.output:
            with open(args.output, "w") as f:
                json.dump({
                    "label": f"layer_subspace_probe_{args.cloner if not args.skip_clone else 'capacity_only'}",
                    "checkpoint": args.checkpoint,
                    "cloner": None if args.skip_clone else args.cloner,
                    "n_utterances": args.n_utterances,
                    "layers_completed": list(per_layer.keys()),
                    "results": per_layer,
                }, f, indent=2)
            print(f"  [saved] {args.output} ({len(per_layer)} layer(s) so far)")

    return per_layer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--diagnostic", action="store_true",
                   help="Validate the mechanism (call count, pass-through validity, all-layers "
                        "control) on ONE utterance. Always run this before a sweep.")
    p.add_argument("--skip_clone", action="store_true",
                   help="Capacity only -- no cloning. Fast (~1 min); answers 'which layers can "
                        "carry a mark at all' before spending hours on the cloning sweep.")
    p.add_argument("--cloner", type=str, default="xtts", choices=["xtts", "yourtts"],
                   help="xtts = the held-out black-box evaluator the 0.5687 baseline was measured "
                        "on (use for comparability). yourtts = the differentiable surrogate, "
                        "faster, but it is the model PGD was optimized against.")
    p.add_argument("--layers", type=str, default="all,0,1,2,3,4,5,6",
                   help="Comma-separated call indices to probe; 'all' = normal VoiceMark control. "
                        "Indices are msg_processor CALL ORDER, which is RVQ layer order within "
                        "st_model's subset -- run --diagnostic to see how many calls actually occur.")
    p.add_argument("--n_utterances", type=int, default=25)
    p.add_argument("--surrogate_text", type=str, default="This is a test sentence for voice cloning.")
    p.add_argument("--tmp_dir", type=str, default="./tmp_layerprobe")
    p.add_argument("--data_root", type=str, default="./data/librispeech")
    p.add_argument("--n_speakers", type=int, default=10)
    p.add_argument("--utterances_per_speaker", type=int, default=10)
    p.add_argument("--n_eval_speakers", type=int, default=5)
    p.add_argument("--eval_utterances_per_speaker", type=int, default=5)
    p.add_argument("--crop_seconds", type=float, default=3.0)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--include_ffn", action="store_true")
    p.add_argument("--capacity_lora_r", type=int, default=32)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    eval_ds = LibriSpeechSubset(
        root=args.data_root, n_speakers=args.n_speakers,
        utterances_per_speaker=args.utterances_per_speaker,
        n_eval_speakers=args.n_eval_speakers,
        eval_utterances_per_speaker=args.eval_utterances_per_speaker,
        sample_rate=16000, crop_seconds=args.crop_seconds, split="eval",
    )
    eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False, collate_fn=collate_librispeech)

    backbone = build_backbone(args.checkpoint, args.lora_r, args.lora_alpha,
                              args.include_ffn, args.capacity_lora_r)
    backbone.model.eval()

    print("[main] Loading YourTTS surrogate (for SIM embedding, and cloning if --cloner yourtts)...")
    surrogate = load_yourtts_surrogate(device=device)

    xtts = None
    if not args.skip_clone and args.cloner == "xtts":
        from xtts_transfer_eval import load_xtts
        xtts = load_xtts(device)

    # ---------------- diagnostic ----------------
    if args.diagnostic:
        print("\n" + "=" * 60 + "\nDIAGNOSTIC MODE\n" + "=" * 60)
        batch = next(iter(eval_loader))
        clean_audio = batch["waveform"].to(device)
        message = random_message(16, clean_audio.shape[0], device, seed=123)

        # (1) how many msg_processor calls per forward == how many layers?
        _wm_all, n_calls = embed_single_layer(backbone, clean_audio, message, None)
        print(f"\n--- 1. Call count ---")
        print(f"  msg_processor called {n_calls} time(s) per forward pass.")
        print(f"  -> valid --layers indices are 0..{n_calls - 1}. VoiceMark embeds across RVQ "
              f"layers 2-8, so {n_calls} == 7 is the expected value; anything else means the "
              f"subset differs from what backbone.py documents -- STOP and inspect.")

        # (2) is the pass-through approximation sound? How much does
        #     msg_processor actually change a layer it processes?
        print(f"\n--- 2. Pass-through validity ---")
        st_model = backbone.model.st_model
        probe = SingleLayerMsgProcessor(backbone.model.msg_processor, target_call_idx=None)
        captured = {}
        real_mp = backbone.model.msg_processor

        class _Capture:
            def __init__(self, mp):
                self.mp = mp
                self.i = 0
            def __call__(self, *a, **k):
                out = self.mp(*a, **k)
                if self.i == 0:
                    captured["x"] = a[0].detach()
                    captured["out"] = out.detach()
                self.i += 1
                return out
            def __getattr__(self, n):
                if n == "mp":
                    raise AttributeError(n)
                return getattr(self.mp, n)

        with torch.no_grad():
            st_model(clean_audio, msg_processor=_Capture(real_mp), message=message)
        if "x" in captured:
            x, out = captured["x"], captured["out"]
            rel = (out - x).norm().item() / x.norm().clamp_min(1e-12).item()
            print(f"  ||msg_processor(x,m) - x|| / ||x|| = {rel:.4f} on the first layer.")
            print(f"  -> this is how much the watermark perturbs a layer it occupies. Small "
                  f"(<~0.2) means returning raw x for the OTHER layers is a faithful "
                  f"'unwatermarked' stand-in and this probe is measuring what it claims. "
                  f"Large means msg_processor transforms layers substantially beyond adding a "
                  f"message, and single-layer numbers should be read with care.")

        # (3) does the all-layers control reproduce the known baseline?
        acc_all = detect_acc(backbone, _wm_all, message)
        print(f"\n--- 3. All-layers control ---")
        print(f"  detection ACC (all layers, normal VoiceMark) = {acc_all:.4f}")
        print(f"  -> should be near this checkpoint's known ~0.99. If it is not, something "
              f"upstream is wrong and NO layer number below would be trustworthy.")

        # (4) one single-layer example
        wm_L0, _ = embed_single_layer(backbone, clean_audio, message, 0)
        acc_L0 = detect_acc(backbone, wm_L0, message)
        print(f"\n--- 4. Single-layer example (layer 0) ---")
        print(f"  detection ACC (message in layer 0 only) = {acc_L0:.4f}")
        print(f"  -> expected to be LOWER than the all-layers control: the detector was trained "
              f"to read a mark spread across every layer, so single-layer embedding is "
              f"off-distribution for it. Well above 0.5 = that layer carries real payload. "
              f"At 0.5 = it cannot carry the mark alone (which is itself a finding).")

        print(f"\n[main] Diagnostic complete. If the call count is as expected and the "
              f"all-layers control matches the known baseline, drop --diagnostic and run "
              f"--skip_clone first (fast capacity pass) before the full cloning sweep.")
        return

    # ---------------- sweep ----------------
    layer_ids = [None if tok.strip() == "all" else int(tok.strip())
                 for tok in args.layers.split(",") if tok.strip()]

    print(f"\n{'=' * 60}")
    print(f"LAYER SUBSPACE PROBE | layers={args.layers} | n_utterances={args.n_utterances} | "
          f"cloner={'(none, capacity only)' if args.skip_clone else args.cloner}")
    print(f"{'=' * 60}")

    per_layer = run_probe(backbone, surrogate, xtts, eval_loader, device, args, layer_ids)

    # ---------------- final table ----------------
    print(f"\n\n{'=' * 78}\nFINAL: which subspace carries the watermark across the cloning boundary?\n{'=' * 78}")
    if args.skip_clone:
        print(f"{'layer':<8}{'acc_source (capacity)':<24}")
        for lbl, e in per_layer.items():
            print(f"{lbl:<8}{e['acc_source_mean']:<24.4f}")
        print("\nCapacity pass only. Layers well above 0.5 are the candidates worth spending "
              "cloning time on -- re-run those without --skip_clone.")
    else:
        print(f"{'layer':<8}{'acc_source':<14}{'acc_clone':<14}{'sim_clone':<14}{'survival_ratio':<16}")
        for lbl, e in per_layer.items():
            sr = e.get("survival_ratio")
            sr_s = "n/a" if sr is None else f"{sr:.3f}"
            print(f"{lbl:<8}{e['acc_source_mean']:<14.4f}{e.get('acc_clone_mean', float('nan')):<14.4f}"
                  f"{e.get('sim_clone_mean', float('nan')):<14.4f}{sr_s:<16}")
        print("\nRead this as: acc_source = can the layer carry a mark at all; acc_clone = does "
              "that mark cross the cloning bottleneck; survival_ratio = what fraction of the "
              "recoverable signal survives. A layer with modest capacity but HIGH survival_ratio "
              "is the interesting one -- that is where Route 2's joint training should place "
              "payload. Compare acc_clone against the all-layers baseline (0.5687 at n=100, XTTS).")

    if args.output:
        print(f"\n[main] Results saved to {args.output}")


if __name__ == "__main__":
    main()
