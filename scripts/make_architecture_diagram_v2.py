"""
scripts/make_architecture_diagram_v2.py

A SECOND architecture figure, complementary to pipeline_diagram.png (which
shows system-level data flow -- clean audio -> codec -> embedder -> detector
-> AudioPure/YourTTS branches -> metrics). This one shows what pipeline_diagram
doesn't: the five VoiceMark losses, the SafeSpeech-derived disruption loss,
and the EXACT LoRA insertion points inside msg_processor/detector -- which
matters for the thesis write-up because "294,912 trainable parameters" means
nothing to a reader without knowing WHERE those parameters sit (attention
projections only, in the canonical setup) and where they DON'T (the
feedforward/MLP blocks, left entirely frozen -- the gap the capacity
experiment in train_stage2_capacity.py tests).

Run once to regenerate:
    python scripts/make_architecture_diagram_v2.py
Writes architecture_diagram_v2.png to the repo root, next to pipeline_diagram.png.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

# Same categorical palette as aggregate_results.py / the dataviz skill's
# reference palette, reused here for visual consistency across all figures
# this project generates.
BLUE = "#2a78d6"      # frozen model
AQUA = "#1baf7a"       # trainable (LoRA)
YELLOW = "#eda100"     # measurement / loss
VIOLET = "#4a3aa7"     # capacity-experiment addition (proposed, not canonical)
INK = "#0b0b0b"
MUTED = "#898781"
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"

FROZEN_FACE = "#eaf1fc"
TRAIN_FACE = "#e3f7ef"
LOSS_FACE = "#fdf2e0"
PROPOSED_FACE = "#efecf9"


def box(ax, xy, w, h, text, face, edge, fontsize=9, style="round,pad=0.35,rounding_size=6"):
    x, y = xy
    b = FancyBboxPatch((x, y), w, h, boxstyle=style, linewidth=1.4,
                        edgecolor=edge, facecolor=face, mutation_scale=1)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, color=INK, wrap=True)
    return (x, y, w, h)


def arrow(ax, start, end, color=MUTED, style="-|>", lw=1.3, connectionstyle="arc3,rad=0.0", ls="-"):
    a = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=14,
                         color=color, linewidth=lw, connectionstyle=connectionstyle, linestyle=ls)
    ax.add_patch(a)


def loss_label(ax, xy, text, fontsize=8):
    ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=fontsize,
             color="#7a5200", style="italic",
             bbox=dict(boxstyle="round,pad=0.25", fc=LOSS_FACE, ec=YELLOW, lw=1))


def main():
    fig, ax = plt.subplots(figsize=(15, 9))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")

    # --- Backbone spine ---
    box(ax, (2, 48), 16, 8, "Clean audio\n(waveform)", FROZEN_FACE, BLUE)
    box(ax, (22, 48), 18, 8, "st_model\n(SpeechTokenizer RVQ codec)\nFROZEN, all quantizers", FROZEN_FACE, BLUE)
    box(ax, (44, 48), 20, 10, "msg_processor (WMEmbedder)\ntransformer_decoder\nself_attn + cross_attn (multihead_attn)\n+ feedforward (linear1/linear2)", TRAIN_FACE, AQUA)
    box(ax, (68, 48), 16, 8, "recon_wm\n(watermarked audio)", FROZEN_FACE, BLUE)
    box(ax, (68, 32), 16, 8, "detector (WMDetector)\ntransformer, self_attn\n+ feedforward", TRAIN_FACE, AQUA)
    box(ax, (88, 32), 10, 8, "presence_logits\nchunk_logits", LOSS_FACE, YELLOW, fontsize=8)

    arrow(ax, (18, 52), (22, 52))
    arrow(ax, (40, 52), (44, 52))
    arrow(ax, (64, 52), (68, 52))
    arrow(ax, (76, 48), (76, 40))
    arrow(ax, (84, 36), (88, 36))

    # LoRA insertion badges (canonical, r=8/alpha=16, attention projections only)
    for cx, cy in [(48, 46.5), (54, 46.5)]:
        ax.add_patch(Circle((cx, cy), 1.1, facecolor=AQUA, edgecolor="white", linewidth=1, zorder=5))
    ax.text(51, 44.7, "LoRA (r=8, α=16) on self_attn + cross_attn\nin_proj/out_proj -- msg_processor",
            ha="center", va="center", fontsize=7.3, color="#0a5c37")

    for cx, cy in [(76, 30.5)]:
        ax.add_patch(Circle((cx, cy), 1.1, facecolor=AQUA, edgecolor="white", linewidth=1, zorder=5))
    ax.text(76, 28.7, "LoRA (r=8, α=16) on self_attn -- detector",
            ha="center", va="center", fontsize=7.3, color="#0a5c37")

    # Proposed capacity extension (train_stage2_capacity.py) -- feedforward LoRA,
    # msg_processor only, dashed violet to mark it as proposed/untested-at-scale.
    fx, fy, fw, fh = 44, 36, 20, 5
    b = FancyBboxPatch((fx, fy), fw, fh, boxstyle="round,pad=0.3,rounding_size=5",
                        linewidth=1.4, edgecolor=VIOLET, facecolor=PROPOSED_FACE, linestyle="--")
    ax.add_patch(b)
    ax.text(fx + fw / 2, fy + fh / 2,
            "PROPOSED (train_stage2_capacity.py):\nLoRA on msg_processor's feedforward too (r up to 32)\n-- previously entirely frozen; untested capacity increase",
            ha="center", va="center", fontsize=7.3, color=VIOLET)

    # --- VoiceMark's five losses ---
    loss_label(ax, (33, 58), "Lcos: cosine(acoustic, acoustic_wm)\n(pre/post-watermark RVQ latents)")
    arrow(ax, (33, 56.3), (33, 52.2), color=YELLOW, ls="--")

    loss_label(ax, (68, 58), "Lmel + Ladv\n(mel loss + hinge/feature-matching\nvs. frozen msstftd discriminator)")
    arrow(ax, (68, 56.3), (68, 56.1), color=YELLOW, ls="--")
    arrow(ax, (72, 57.3), (76, 48), color=YELLOW, ls="--", connectionstyle="arc3,rad=0.15")

    loss_label(ax, (94, 44), "Ldec: CE(chunk_logits, message)\nLvad: VAD-gated frame masking")

    # --- Disruption branch (SafeSpeech-derived) ---
    box(ax, (68, 14), 16, 8, "YourTTS surrogate\n(zero-shot VC) FROZEN", FROZEN_FACE, BLUE)
    box(ax, (88, 14), 10, 8, "cloned_output", FROZEN_FACE, BLUE, fontsize=8)
    arrow(ax, (76, 32), (76, 22), color=MUTED)
    arrow(ax, (84, 18), (88, 18))

    loss_label(ax, (68, 6),
               "sim mode (default): -cos(spk_emb(clean), spk_emb(cloned))\nmel mode: pivotal mel L1 (negated) + weight_l1*L1_to_noise\n+ weight_kl*KL_to_noise  [weight_kl ~1000x smaller than\nSafeSpeech's own tuning -- see gradient_diagnostic.py]")
    arrow(ax, (76, 14), (76, 9.5), color=YELLOW, ls="--")

    box(ax, (2, 14), 16, 8, "total = VoiceMark losses\n+ lambda(step) * disruption loss\n(lambda ramps 0 -> lambda_max\nover lambda_ramp_steps)", LOSS_FACE, YELLOW, fontsize=7.6)
    arrow(ax, (68, 10), (18, 18), color=YELLOW, ls="--", connectionstyle="arc3,rad=-0.2")
    arrow(ax, (44, 53), (18, 20), color=YELLOW, ls="--", connectionstyle="arc3,rad=0.2")

    # --- Legend ---
    legend_elems = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=FROZEN_FACE,
               markeredgecolor=BLUE, markersize=14, label="Frozen model / tensor"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=TRAIN_FACE,
               markeredgecolor=AQUA, markersize=14, label="Trainable (LoRA-wrapped submodule)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=LOSS_FACE,
               markeredgecolor=YELLOW, markersize=14, label="Loss term / measurement"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=PROPOSED_FACE,
               markeredgecolor=VIOLET, markersize=14, label="Proposed (untested) capacity extension"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=AQUA,
               markeredgecolor="white", markersize=10, label="LoRA insertion point"),
    ]
    ax.legend(handles=legend_elems, loc="lower left", bbox_to_anchor=(0.0, -0.02),
              frameon=False, fontsize=8.5, ncol=1)

    ax.set_title("Model-internals architecture: losses + exact LoRA insertion points\n"
                  "(companion figure to pipeline_diagram.png, which shows system-level data flow)",
                  fontsize=12, color=INK, pad=14)

    out_path = os.path.join(os.path.dirname(__file__), "..", "architecture_diagram_v2.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, facecolor=SURFACE)
    print(f"Wrote {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
