"""Visualize control point residuals from DLT or resection results."""

import json
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def load_result(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_residuals(data: dict) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Extract point IDs, vx, vy from DLT or resection JSON."""
    if "results" in data:
        key = "dlt_K1" if "dlt_K1" in data["results"] else list(data["results"].keys())[0]
        residuals = data["results"][key]["residuals"]
        label = data["results"][key].get("label", key)
    else:
        residuals = data["residuals"]
        label = "Resection"

    ids = [r["control_id"] for r in residuals]
    vx = np.array([r["vx"] for r in residuals])
    vy = np.array([r["vy"] for r in residuals])
    return ids, vx, vy, label


def plot_residual_scatter(ax, vx, vy, title, scale=1.0):
    """Residual scatter plot with arrows from origin."""
    ax.scatter(vx * scale, vy * scale, c="steelblue", s=40, zorder=5, edgecolors="white", linewidths=0.5)
    for i in range(len(vx)):
        ax.annotate("", xy=(vx[i] * scale, vy[i] * scale), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color="steelblue", alpha=0.4, lw=0.8))
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel("vx (mm)")
    ax.set_ylabel("vy (mm)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)


def plot_residual_histogram(ax, vx, vy, title):
    """Residual histograms for vx and vy."""
    bins = max(8, int(np.sqrt(len(vx))))
    ax.hist(vx, bins=bins, alpha=0.6, label="vx", color="steelblue", edgecolor="white")
    ax.hist(vy, bins=bins, alpha=0.6, label="vy", color="coral", edgecolor="white")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel("残差 (mm)")
    ax.set_ylabel("频次")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_comparison_scatter(ax, vx_list, vy_list, labels, title):
    """Compare residuals from multiple solutions."""
    colors = ["steelblue", "coral", "seagreen"]
    for i, (vx, vy, lbl) in enumerate(zip(vx_list, vy_list, labels)):
        c = colors[i % len(colors)]
        ax.scatter(vx, vy, c=c, s=30, alpha=0.7, label=lbl, edgecolors="white", linewidths=0.3)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel("vx (mm)")
    ax.set_ylabel("vy (mm)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def write_image_residual_page(pdf, ids, vx, vy, label, image_name):
    """Write one page of residual plots for a single image/solution."""
    sigma = np.sqrt(np.mean(vx**2 + vy**2))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"{image_name} — {label}  (sigma0={sigma:.4f} mm, {len(vx)} 控制点)",
                 fontsize=13, fontweight="bold", y=0.98)

    # Scatter
    plot_residual_scatter(axes[0, 0], vx, vy,
                          f"残差散点图  (sigma0={sigma:.4f} mm)")

    # Histogram
    plot_residual_histogram(axes[0, 1], vx, vy, "残差直方图")

    # Per-point bar
    axes[1, 0].bar(range(len(vx)), vx * 1000, alpha=0.7, label="vx", color="steelblue")
    axes[1, 0].bar(range(len(vy)), vy * 1000, alpha=0.7, label="vy", color="coral")
    axes[1, 0].axhline(0, color="gray", lw=0.5, ls="--")
    axes[1, 0].set_xlabel("控制点序号")
    axes[1, 0].set_ylabel("残差 (um)")
    axes[1, 0].set_title("逐点残差")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # Summary statistics table
    axes[1, 1].axis("off")
    stats = [
        ["控制点数", f"{len(vx)}"],
        ["sigma0", f"{sigma:.6f} mm"],
        ["sigma0 (px)", f"{sigma / 0.00435:.2f} px"],
        ["max |vx|", f"{np.max(np.abs(vx)):.6f} mm"],
        ["max |vy|", f"{np.max(np.abs(vy)):.6f} mm"],
        ["mean vx", f"{np.mean(vx):.6f} mm"],
        ["mean vy", f"{np.mean(vy):.6f} mm"],
        ["std vx", f"{np.std(vx):.6f} mm"],
        ["std vy", f"{np.std(vy):.6f} mm"],
    ]
    table = axes[1, 1].table(cellText=stats, colLabels=["指标", "值"],
                             loc="center", cellLoc="center",
                             colWidths=[0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    axes[1, 1].set_title("残差统计", pad=20)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def write_comparison_page(pdf, all_vx, all_vy, all_labels, image_names):
    """Write a comparison page if multiple images are provided."""
    n = len(all_vx)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for i, (vx, vy, lbl, img) in enumerate(zip(all_vx, all_vy, all_labels, image_names)):
        plot_comparison_scatter(axes[i], [vx], [vy], [lbl], f"{img} — 残差散点")

    fig.suptitle("各影像残差对比", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def write_combined_scatter_page(pdf, all_vx, all_vy, all_labels, image_names):
    """Write a page with all solutions overlaid on one scatter plot."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    colors = ["steelblue", "coral", "seagreen", "orange", "purple"]
    for i, (vx, vy, lbl, img) in enumerate(zip(all_vx, all_vy, all_labels, image_names)):
        c = colors[i % len(colors)]
        sigma = np.sqrt(np.mean(vx**2 + vy**2))
        ax.scatter(vx, vy, c=c, s=30, alpha=0.7, label=f"{img} {lbl} (sigma={sigma:.4f})",
                   edgecolors="white", linewidths=0.3)

    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel("vx (mm)")
    ax.set_ylabel("vy (mm)")
    ax.set_title("所有影像残差汇总对比")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize residuals (PDF output)")
    parser.add_argument("files", nargs="+", help="Result JSON files (DLT or resection)")
    parser.add_argument("--output", "-o", default="", help="Output PDF path")
    parser.add_argument("--per-image", action="store_true", default=True,
                        help="Generate per-image PDF (default: True)")
    parser.add_argument("--no-show", action="store_true", help="Save without showing")
    args = parser.parse_args()

    out = args.output or "output/residuals.pdf"

    all_vx, all_vy, all_labels, image_names = [], [], [], []

    with PdfPages(out) as pdf:
        for fpath in args.files:
            if not Path(fpath).exists():
                print(f"File not found: {fpath}", file=sys.stderr)
                continue

            data = load_result(fpath)
            ids, vx, vy, label = extract_residuals(data)
            img_name = data.get("image", Path(fpath).stem)

            all_vx.append(vx)
            all_vy.append(vy)
            all_labels.append(label)
            image_names.append(img_name)

            # Write one page per image/solution
            write_image_residual_page(pdf, ids, vx, vy, label, img_name)

        # If multiple files, add comparison pages
        if len(all_vx) > 1:
            write_combined_scatter_page(pdf, all_vx, all_vy, all_labels, image_names)
            write_comparison_page(pdf, all_vx, all_vy, all_labels, image_names)

    print(f"Saved: {out}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
