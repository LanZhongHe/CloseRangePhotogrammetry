"""Compare DLT and resection results side by side."""

import json
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_comparison(dlt_path: str, resection_path: str) -> None:
    dlt = load_json(dlt_path)
    res = load_json(resection_path)

    img = dlt.get("image", Path(dlt_path).stem)
    n_ctrl = dlt.get("num_control_points", "?")

    print(f"\n{'=' * 70}")
    print(f"  {img}  ({n_ctrl} control points)")
    print(f"{'=' * 70}")

    # Header
    hdr = f"{'指标':<28s} {'DLT(无畸变)':>14s} {'DLT+K1':>14s} {'后方交会':>14s}"
    print(hdr)
    print("-" * 70)

    # sigma0 mm
    nodist = dlt["results"]["dlt_no_distortion"]
    k1 = dlt["results"]["dlt_K1"]
    res_sigma = res["sigma0_mm"]
    print(f"{'sigma0 (mm)':<28s} {nodist['sigma0_mm']:>14.6f} {k1['sigma0_mm']:>14.6f} {res_sigma:>14.6f}")

    # sigma0 px
    px = res["intrinsics"]["pixel_size"]
    print(f"{'sigma0 (px)':<28s} {nodist['sigma0_mm']/px:>14.2f} {k1['sigma0_mm']/px:>14.2f} {res['sigma0_px']:>14.2f}")

    # f
    print(f"{'f (mm)':<28s} {nodist['intrinsics']['f']:>14.4f} {k1['intrinsics']['f']:>14.4f} {res['intrinsics']['f']:>14.4f}")

    # x0, y0
    print(f"{'x0 (mm)':<28s} {nodist['intrinsics']['x0']:>14.4f} {k1['intrinsics']['x0']:>14.4f} {res['intrinsics']['x0']:>14.4f}")
    print(f"{'y0 (mm)':<28s} {nodist['intrinsics']['y0']:>14.4f} {k1['intrinsics']['y0']:>14.4f} {res['intrinsics']['y0']:>14.4f}")

    # Distortion
    print(f"{'K1':<28s} {'—':>14s} {k1['distortion']['K1']:>14.2e} {res['distortion']['K1']:>14.2e}")
    print(f"{'K2':<28s} {'—':>14s} {k1['distortion']['K2']:>14.2e} {res['distortion']['K2']:>14.2e}")
    print(f"{'P1':<28s} {'—':>14s} {k1['distortion']['P1']:>14.2e} {res['distortion']['P1']:>14.2e}")
    print(f"{'P2':<28s} {'—':>14s} {k1['distortion']['P2']:>14.2e} {res['distortion']['P2']:>14.2e}")

    # Exterior orientation
    ext_n = nodist["exterior"]
    ext_r = res["exterior"]
    print(f"{'Xs (mm)':<28s} {ext_n['Xs']:>14.2f} {'—':>14s} {ext_r['Xs']:>14.2f}")
    print(f"{'Ys (mm)':<28s} {ext_n['Ys']:>14.2f} {'—':>14s} {ext_r['Ys']:>14.2f}")
    print(f"{'Zs (mm)':<28s} {ext_n['Zs']:>14.2f} {'—':>14s} {ext_r['Zs']:>14.2f}")
    print(f"{'omega (deg)':<28s} {ext_n['omega_deg']:>14.4f} {'—':>14s} {ext_r['omega_deg']:>14.4f}")
    print(f"{'phi (deg)':<28s} {ext_n['phi_deg']:>14.4f} {'—':>14s} {ext_r['phi_deg']:>14.4f}")
    print(f"{'kappa (deg)':<28s} {ext_n['kappa_deg']:>14.4f} {'—':>14s} {ext_r['kappa_deg']:>14.4f}")

    # Iterations
    print(f"{'迭代次数':<28s} {nodist['num_iterations']:>14d} {k1['num_iterations']:>14d} {res['num_iterations']:>14d}")

    # Analysis
    print(f"\n--- 分析 ---")
    improvement = (nodist["sigma0_mm"] - k1["sigma0_mm"]) / nodist["sigma0_mm"] * 100
    print(f"  DLT畸变校正精度提升: {improvement:.1f}%")
    dlt_vs_res = k1["sigma0_mm"] / res["sigma0_mm"]
    print(f"  DLT+K1 / 后方交会 sigma0 比值: {dlt_vs_res:.1f}x")
    print(f"  焦距一致性: DLT={k1['intrinsics']['f']:.2f} mm, 后方交会={res['intrinsics']['f']:.2f} mm, 差={abs(k1['intrinsics']['f'] - res['intrinsics']['f']):.4f} mm")


def build_comparison_table(dlt_path: str, resection_path: str) -> tuple[str, list[list[str]]]:
    """Build table data for PDF output."""
    dlt = load_json(dlt_path)
    res = load_json(resection_path)

    img = dlt.get("image", Path(dlt_path).stem)
    n_ctrl = dlt.get("num_control_points", "?")

    nodist = dlt["results"]["dlt_no_distortion"]
    k1 = dlt["results"]["dlt_K1"]
    px = res["intrinsics"]["pixel_size"]

    improvement = (nodist["sigma0_mm"] - k1["sigma0_mm"]) / nodist["sigma0_mm"] * 100
    dlt_vs_res = k1["sigma0_mm"] / res["sigma0_mm"]

    title = f"{img} ({n_ctrl} 控制点)"
    rows = [
        ["sigma0 (mm)", f"{nodist['sigma0_mm']:.6f}", f"{k1['sigma0_mm']:.6f}", f"{res['sigma0_mm']:.6f}"],
        ["sigma0 (px)", f"{nodist['sigma0_mm']/px:.2f}", f"{k1['sigma0_mm']/px:.2f}", f"{res['sigma0_px']:.2f}"],
        ["f (mm)", f"{nodist['intrinsics']['f']:.4f}", f"{k1['intrinsics']['f']:.4f}", f"{res['intrinsics']['f']:.4f}"],
        ["x0 (mm)", f"{nodist['intrinsics']['x0']:.4f}", f"{k1['intrinsics']['x0']:.4f}", f"{res['intrinsics']['x0']:.4f}"],
        ["y0 (mm)", f"{nodist['intrinsics']['y0']:.4f}", f"{k1['intrinsics']['y0']:.4f}", f"{res['intrinsics']['y0']:.4f}"],
        ["K1", "—", f"{k1['distortion']['K1']:.2e}", f"{res['distortion']['K1']:.2e}"],
        ["K2", "—", f"{k1['distortion']['K2']:.2e}", f"{res['distortion']['K2']:.2e}"],
        ["P1", "—", f"{k1['distortion']['P1']:.2e}", f"{res['distortion']['P1']:.2e}"],
        ["P2", "—", f"{k1['distortion']['P2']:.2e}", f"{res['distortion']['P2']:.2e}"],
        ["Xs (mm)", f"{nodist['exterior']['Xs']:.2f}", "—", f"{res['exterior']['Xs']:.2f}"],
        ["Ys (mm)", f"{nodist['exterior']['Ys']:.2f}", "—", f"{res['exterior']['Ys']:.2f}"],
        ["Zs (mm)", f"{nodist['exterior']['Zs']:.2f}", "—", f"{res['exterior']['Zs']:.2f}"],
        ["omega (deg)", f"{nodist['exterior']['omega_deg']:.4f}", "—", f"{res['exterior']['omega_deg']:.4f}"],
        ["phi (deg)", f"{nodist['exterior']['phi_deg']:.4f}", "—", f"{res['exterior']['phi_deg']:.4f}"],
        ["kappa (deg)", f"{nodist['exterior']['kappa_deg']:.4f}", "—", f"{res['exterior']['kappa_deg']:.4f}"],
        ["迭代次数", f"{nodist['num_iterations']}", f"{k1['num_iterations']}", f"{res['num_iterations']}"],
        ["", "", "", ""],
        ["DLT畸变校正提升", f"{improvement:.1f}%", "", ""],
        ["DLT+K1/后方交会 sigma0", f"{dlt_vs_res:.2f}x", "", ""],
        ["焦距差 (mm)", f"{abs(k1['intrinsics']['f'] - res['intrinsics']['f']):.4f}", "", ""],
    ]
    return title, rows


def write_pdf_comparison(dlt_paths: list[str], res_paths: list[str], out_path: str):
    """Generate PDF with comparison tables."""
    with PdfPages(out_path) as pdf:
        for dlt_path, res_path in zip(dlt_paths, res_paths):
            if not Path(dlt_path).exists():
                print(f"File not found: {dlt_path}", file=sys.stderr)
                continue
            if not Path(res_path).exists():
                print(f"File not found: {res_path}", file=sys.stderr)
                continue

            title, rows = build_comparison_table(dlt_path, res_path)

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.axis("off")
            ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

            col_labels = ["指标", "DLT (无畸变)", "DLT + K1", "后方交会"]
            table = ax.table(cellText=rows, colLabels=col_labels,
                             loc="center", cellLoc="center",
                             colWidths=[0.28, 0.22, 0.22, 0.22])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.3)

            # Style header row
            for j in range(4):
                cell = table[0, j]
                cell.set_facecolor("#4472C4")
                cell.set_text_props(color="white", fontweight="bold")

            # Alternate row colors
            for i in range(1, len(rows) + 1):
                for j in range(4):
                    cell = table[i, j]
                    if rows[i-1][0] in ["DLT畸变校正提升", "DLT+K1/后方交会 sigma0", "焦距差 (mm)"]:
                        cell.set_facecolor("#E2EFDA")
                    elif i % 2 == 0:
                        cell.set_facecolor("#D9E2F3")

            fig.tight_layout()
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare DLT and resection results")
    parser.add_argument("--dlt", nargs=2, default=["output/dlt_DSC_0035.json", "output/dlt_DSC_0037.json"],
                        help="DLT result JSON files")
    parser.add_argument("--resection", nargs=2, default=["output/resection_DSC_0035.json", "output/resection_DSC_0037.json"],
                        help="Resection result JSON files")
    parser.add_argument("--output", "-o", default="output/comparison.pdf",
                        help="Output PDF path")
    parser.add_argument("--no-show", action="store_true", help="Save without showing")
    args = parser.parse_args()

    # Console output
    for dlt_path, res_path in zip(args.dlt, args.resection):
        if not Path(dlt_path).exists():
            print(f"File not found: {dlt_path}", file=sys.stderr)
            continue
        if not Path(res_path).exists():
            print(f"File not found: {res_path}", file=sys.stderr)
            continue
        print_comparison(dlt_path, res_path)

    print()

    # PDF output
    write_pdf_comparison(args.dlt, args.resection, args.output)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
