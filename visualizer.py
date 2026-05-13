import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

DIMENSIONS = ["Safety", "Fairness", "Helpfulness", "Clarity", "Consistency"]

DIM_KEYS_A = [
    "model_a_safety", "model_a_fairness",
    "model_a_helpfulness", "model_a_clarity", "model_a_consistency",
]
DIM_KEYS_B = [
    "model_b_safety", "model_b_fairness",
    "model_b_helpfulness", "model_b_clarity", "model_b_consistency",
]

COLOR_A = "#EF5350"   # red — challenger (smaller model)
COLOR_B = "#42A5F5"   # blue — baseline (larger model)
BG_COLOR = "#0F1117"
GRID_COLOR = "#2A2D3A"
TEXT_COLOR = "#E0E0E0"


def _avg_scores(results, keys):
    avgs = []
    for key in keys:
        vals = [r[key] for r in results if isinstance(r[key], (int, float))]
        avgs.append(sum(vals) / len(vals) if vals else 0)
    return avgs


def _pass_rate(results, field):
    passed = sum(1 for r in results if r[field] == "PASS")
    return passed / len(results) * 100 if results else 0


def generate_report(results, model_a_name, model_b_name, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    scores_a = _avg_scores(results, DIM_KEYS_A)
    scores_b = _avg_scores(results, DIM_KEYS_B)

    N = len(DIMENSIONS)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    vals_a = scores_a + scores_a[:1]
    vals_b = scores_b + scores_b[:1]

    fig = plt.figure(figsize=(16, 9), facecolor=BG_COLOR)

    # --- Left: Radar chart ---
    ax_radar = fig.add_axes([0.03, 0.08, 0.48, 0.84], polar=True, facecolor=BG_COLOR)

    for r in np.arange(2, 11, 2):
        ax_radar.plot(angles, [r] * (N + 1), color=GRID_COLOR, linewidth=0.8, linestyle="--")
        ax_radar.text(angles[0], r + 0.2, str(r), color="#555770", fontsize=7, ha="center")

    ax_radar.plot(angles, vals_a, color=COLOR_A, linewidth=2.5, linestyle="solid", zorder=3)
    ax_radar.fill(angles, vals_a, color=COLOR_A, alpha=0.18, zorder=2)

    ax_radar.plot(angles, vals_b, color=COLOR_B, linewidth=2.5, linestyle="solid", zorder=3)
    ax_radar.fill(angles, vals_b, color=COLOR_B, alpha=0.18, zorder=2)

    for angle, val_a, val_b in zip(angles[:-1], scores_a, scores_b):
        ax_radar.plot(angle, val_a, "o", color=COLOR_A, markersize=7, zorder=4)
        ax_radar.plot(angle, val_b, "o", color=COLOR_B, markersize=7, zorder=4)

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(DIMENSIONS, color=TEXT_COLOR, fontsize=12, fontweight="bold")
    ax_radar.set_ylim(0, 10)
    ax_radar.set_yticks([])
    ax_radar.spines["polar"].set_color(GRID_COLOR)
    ax_radar.grid(False)

    short_a = model_a_name.split("-")[0] + "-" + model_a_name.split("-")[1] if "-" in model_a_name else model_a_name
    short_b = model_b_name.split("-")[0] + "-" + model_b_name.split("-")[1] if "-" in model_b_name else model_b_name

    patch_a = mpatches.Patch(color=COLOR_A, label=f"Model A  ({model_a_name})", alpha=0.85)
    patch_b = mpatches.Patch(color=COLOR_B, label=f"Model B  ({model_b_name})", alpha=0.85)
    ax_radar.legend(
        handles=[patch_a, patch_b],
        loc="upper right",
        bbox_to_anchor=(1.35, 1.18),
        fontsize=9,
        framealpha=0,
        labelcolor=TEXT_COLOR,
    )

    ax_radar.set_title("Quality Dimensions — Radar", color=TEXT_COLOR,
                        fontsize=13, fontweight="bold", pad=20)

    # --- Right: Bar chart per test case ---
    ax_bar = fig.add_axes([0.56, 0.10, 0.42, 0.78], facecolor=BG_COLOR)

    categories = [r["category"].replace("Adversarial - ", "Adv: ") for r in results]
    overall_a = [r["model_a_overall"] for r in results]
    overall_b = [r["model_b_overall"] for r in results]
    pass_a = [r["model_a_pass_fail"] for r in results]
    pass_b = [r["model_b_pass_fail"] for r in results]

    y = np.arange(len(results))
    h = 0.35

    bars_a = ax_bar.barh(y + h / 2, overall_a, h, color=COLOR_A, alpha=0.85, label=f"Model A")
    bars_b = ax_bar.barh(y - h / 2, overall_b, h, color=COLOR_B, alpha=0.85, label=f"Model B")

    for i, (bar_a, bar_b, pa, pb) in enumerate(zip(bars_a, bars_b, pass_a, pass_b)):
        marker_a = "✓" if pa == "PASS" else "✗"
        marker_b = "✓" if pb == "PASS" else "✗"
        col_a = "#69F0AE" if pa == "PASS" else "#FF5252"
        col_b = "#69F0AE" if pb == "PASS" else "#FF5252"
        ax_bar.text(bar_a.get_width() + 0.1, bar_a.get_y() + bar_a.get_height() / 2,
                    marker_a, va="center", color=col_a, fontsize=11, fontweight="bold")
        ax_bar.text(bar_b.get_width() + 0.1, bar_b.get_y() + bar_b.get_height() / 2,
                    marker_b, va="center", color=col_b, fontsize=11, fontweight="bold")

    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(categories, color=TEXT_COLOR, fontsize=8.5)
    ax_bar.set_xlim(0, 11.5)
    ax_bar.set_xlabel("Overall Score (/10)", color=TEXT_COLOR, fontsize=10)
    ax_bar.set_title("Overall Score per Test Case", color=TEXT_COLOR,
                      fontsize=13, fontweight="bold")
    ax_bar.tick_params(colors=TEXT_COLOR)
    ax_bar.spines[:].set_color(GRID_COLOR)
    ax_bar.set_facecolor(BG_COLOR)
    ax_bar.xaxis.label.set_color(TEXT_COLOR)
    ax_bar.axvline(7, color="#FFA726", linewidth=1.2, linestyle="--", alpha=0.7)
    ax_bar.text(7.05, len(results) - 0.3, "min pass", color="#FFA726", fontsize=7.5)

    # --- Pass rate summary boxes ---
    rate_a = _pass_rate(results, "model_a_pass_fail")
    rate_b = _pass_rate(results, "model_b_pass_fail")

    for i, (rate, color, label) in enumerate([(rate_a, COLOR_A, "Model A"), (rate_b, COLOR_B, "Model B")]):
        x_pos = 0.56 + i * 0.22
        rect = FancyBboxPatch((x_pos, 0.01), 0.20, 0.06,
                               boxstyle="round,pad=0.01",
                               linewidth=1, edgecolor=color,
                               facecolor=BG_COLOR,
                               transform=fig.transFigure, clip_on=False)
        fig.add_artist(rect)
        fig.text(x_pos + 0.10, 0.052, f"{rate:.0f}% pass", ha="center",
                 color=color, fontsize=11, fontweight="bold",
                 transform=fig.transFigure)
        fig.text(x_pos + 0.10, 0.022, label, ha="center",
                 color=TEXT_COLOR, fontsize=8.5,
                 transform=fig.transFigure)

    # --- Main title ---
    fig.text(0.5, 0.97, "AI QA Evaluation — Comparative Model Report",
             ha="center", color=TEXT_COLOR, fontsize=16, fontweight="bold")
    fig.text(0.5, 0.935, f"{model_a_name}  vs  {model_b_name}",
             ha="center", color="#888aaa", fontsize=10)

    outpath = os.path.join(output_dir, "qa_report_chart.png")
    plt.savefig(outpath, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    return outpath
