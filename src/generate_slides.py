"""
generate_slides.py
──────────────────
Runs all 6 scheduling policies for Week 1 and Week 2 with the fixed model
(no provider conflicts), then writes three slide-quality figures to slides/:

  slides/slide_policy_coverage_table.png  — coverage table + bar chart
  slides/slide_coverage_vs_travel.png     — scatter: coverage vs travel
  slides/slide_b_vs_f_efficiency.png      — B vs F room-switches & travel
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# ── ensure project root is on sys.path ───────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "models"))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

os.chdir(ROOT)

from main import (
    run_full_pipeline,
    compute_kpis,
    policy_single_room,
    apply_admin_time_buffer,
    apply_day_blocking,
    apply_noshow_adjustment,
)
from data_loader import (
    load_all_appointments,
    load_all_provider_availability,
    load_distance_matrix,
    compute_noshow_rate,
)
from model2 import generate_schedules_sequential
from model3 import build_master_problem, master_to_appointments

SLIDES_DIR = os.path.join(ROOT, "slides")
os.makedirs(SLIDES_DIR, exist_ok=True)

# ── colour palette (matches slide theme) ─────────────────────────────────────
C_BLUE   = "#4A90D9"
C_TEAL   = "#2EC4B6"
C_ORANGE = "#F4A261"
C_RED    = "#E76F51"
C_DARK   = "#1A2E4A"
C_LIGHT  = "#F0F8FF"
C_GREEN  = "#52B788"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Run all policies for both weeks
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_policies():
    print("\n" + "="*60)
    print("  Running all 6 policies — Weeks 1 & 2")
    print("="*60)

    appts_all   = load_all_appointments()
    avail_all   = load_all_provider_availability()
    dist_matrix = load_distance_matrix()
    noshow_rates = compute_noshow_rate(appts_all)

    records = []   # one dict per (policy, week)

    for week in [1, 2]:
        print(f"\n{'─'*40}\n  Week {week}\n{'─'*40}")
        appts = appts_all[appts_all["week"] == week].copy()
        avail = avail_all[avail_all["week"] == week].copy()

        def _run_pool(appts_in, delta_frac=0.0, thresh=4.0):
            scheds = generate_schedules_sequential(
                appts_in, avail, dist_matrix,
                delta_frac=delta_frac,
                proximity_threshold=thresh,
                verbose=False,
            )
            master = build_master_problem(
                scheds, appts_in, integer=True, verbose=False,
                hard_room_constraints=True,
            )
            return master_to_appointments(master, appts)

        # Policy A
        print("  [A] Single room (baseline)...")
        res = policy_single_room(appts, avail, dist_matrix)
        kpi = compute_kpis(res, dist_matrix, "A: Single Room")
        kpi["week"] = week
        records.append(kpi)

        # Policy B
        print("  [B] Cluster of rooms...")
        res = _run_pool(appts, delta_frac=0.0, thresh=4.0)
        kpi = compute_kpis(res, dist_matrix, "B: Cluster")
        kpi["week"] = week
        records.append(kpi)

        # Policy C
        print("  [C] Cluster + 10% duration buffer...")
        res = _run_pool(appts, delta_frac=0.10, thresh=4.0)
        kpi = compute_kpis(res, dist_matrix, "C: Cluster + Duration Buffer")
        kpi["week"] = week
        records.append(kpi)

        # Policy D
        print("  [D] Cluster + no-show overbooking...")
        appts_ns = apply_noshow_adjustment(appts, noshow_rates).copy()
        appts_ns["duration_min"] = appts_ns["effective_duration"].round().astype(int)
        appts_ns["end_min"] = appts_ns["start_min"] + appts_ns["duration_min"]
        res = master_to_appointments(
            build_master_problem(
                generate_schedules_sequential(appts_ns, avail, dist_matrix,
                                              delta_frac=0.0, proximity_threshold=4.0,
                                              verbose=False),
                appts_ns, integer=True, verbose=False,
                hard_room_constraints=True),
            appts)
        kpi = compute_kpis(res, dist_matrix, "D: Cluster + No-show")
        kpi["week"] = week
        records.append(kpi)

        # Policy E
        print("  [E] Day blocking...")
        appts_blocked = apply_day_blocking(appts, avail)
        scheds_e = generate_schedules_sequential(
            appts_blocked, avail, dist_matrix,
            delta_frac=0.0, proximity_threshold=4.0, verbose=False)
        master_e = build_master_problem(
            scheds_e, appts_blocked, integer=True, verbose=False,
            hard_room_constraints=True)
        res = master_to_appointments(master_e, appts)
        kpi = compute_kpis(res, dist_matrix, "E: Day Blocking")
        kpi["week"] = week
        records.append(kpi)

        # Policy F
        print("  [F] Admin time buffer...")
        appts_admin = apply_admin_time_buffer(appts)
        scheds_f = generate_schedules_sequential(
            appts_admin, avail, dist_matrix,
            delta_frac=0.0, proximity_threshold=4.0, verbose=False)
        master_f = build_master_problem(
            scheds_f, appts_admin, integer=True, verbose=False,
            hard_room_constraints=True)
        res = master_to_appointments(master_f, appts)
        kpi = compute_kpis(res, dist_matrix, "F: Admin Buffer")
        kpi["week"] = week
        records.append(kpi)

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(SLIDES_DIR, "policy_kpi_all_weeks.csv"), index=False)
    print(f"\n  KPIs saved → slides/policy_kpi_all_weeks.csv")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Slide 1 — Coverage table + horizontal bar chart
# ═══════════════════════════════════════════════════════════════════════════════

def make_coverage_table_slide(df: pd.DataFrame):
    # Pivot to wide: one row per policy
    w1 = df[df["week"] == 1].set_index("Policy")["Coverage (%)"]
    w2 = df[df["week"] == 2].set_index("Policy")["Coverage (%)"]

    # Ordered label list
    order = [
        "A: Single Room",
        "B: Cluster",
        "C: Cluster + Duration Buffer",
        "D: Cluster + No-show",
        "E: Day Blocking",
        "F: Admin Buffer",
    ]
    short = ["A", "B", "C", "D", "E", "F"]
    desc  = [
        "Single Room (Baseline)",
        "Cluster of Nearby Rooms (<4m)",
        "Cluster + 10% Duration Buffer",
        "Cluster + No-show Overbooking",
        "Day Blocking",
        "Admin Time as Overflow Buffer",
    ]

    cov_w1 = [w1.get(p, float("nan")) for p in order]
    cov_w2 = [w2.get(p, float("nan")) for p in order]

    # Best rows (B and F)
    best_rows = {1, 5}

    fig = plt.figure(figsize=(16, 7), facecolor="white")
    fig.patch.set_facecolor("white")

    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.1, 1.2], wspace=0.08)
    ax_tbl = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])
    ax_tbl.axis("off")

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.97, "Six Scheduling Policies Evaluated on the Optimized Schedule Pool",
             ha="center", va="top", fontsize=16, fontweight="bold", color=C_DARK)

    # ── Table ─────────────────────────────────────────────────────────────────
    col_labels = ["Policy", "Description", "W1 Coverage", "W2 Coverage"]
    table_data = []
    for i, (s, d, c1, c2) in enumerate(zip(short, desc, cov_w1, cov_w2)):
        table_data.append([
            s,
            d,
            f"{c1:.1f}%" if not np.isnan(c1) else "—",
            f"{c2:.1f}%" if not np.isnan(c2) else "—",
        ])

    tbl = ax_tbl.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="left",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.1)

    # Style header
    for j in range(4):
        tbl[(0, j)].set_facecolor(C_DARK)
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    # Style rows
    for i in range(6):
        is_best = i in best_rows
        bg = "#E8F8F5" if is_best else "white"
        txt_color = C_TEAL if is_best else C_DARK
        for j in range(4):
            tbl[(i+1, j)].set_facecolor(bg)
            tbl[(i+1, j)].set_text_props(
                color=txt_color,
                fontweight="bold" if is_best else "normal",
            )
    tbl.auto_set_column_width([0, 1, 2, 3])

    # ── Bar chart ─────────────────────────────────────────────────────────────
    ax_bar.set_facecolor(C_LIGHT)
    y = np.arange(len(order))
    h = 0.32

    # Best-coverage bars (B&F) get orange/red, others get blue/teal
    def bar_colors_w1(i):
        return C_ORANGE if i in best_rows else C_BLUE

    def bar_colors_w2(i):
        return C_RED if i in best_rows else C_TEAL

    bars1 = ax_bar.barh(y + h/2, cov_w1, h, color=[bar_colors_w1(i) for i in range(6)],
                         label="Week 1", zorder=3)
    bars2 = ax_bar.barh(y - h/2, cov_w2, h, color=[bar_colors_w2(i) for i in range(6)],
                         label="Week 2", zorder=3)

    for bar, val in zip(bars1, cov_w1):
        if not np.isnan(val):
            ax_bar.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                        f"{val:.1f}%", va="center", ha="left", fontsize=9, color=C_DARK)
    for bar, val in zip(bars2, cov_w2):
        if not np.isnan(val):
            ax_bar.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                        f"{val:.1f}%", va="center", ha="left", fontsize=9, color=C_DARK)

    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(
        [f"{s}  {d}" for s, d in zip(short, [d.replace(" Rooms (<4m)", "").replace(" Overbooking","") for d in desc])],
        fontsize=9.5, color=C_DARK
    )
    ax_bar.set_xlabel("Coverage (%)", fontsize=10, color=C_DARK)
    ax_bar.set_xlim(55, 108)
    ax_bar.set_title("Appointment Coverage\nby Scheduling Policy", fontsize=11,
                      fontweight="bold", color=C_DARK)
    ax_bar.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax_bar.tick_params(colors=C_DARK)
    ax_bar.spines[["top","right"]].set_visible(False)
    ax_bar.grid(axis="x", alpha=0.3, zorder=0)

    legend_patches = [
        mpatches.Patch(color=C_BLUE,   label="Week 1"),
        mpatches.Patch(color=C_TEAL,   label="Week 2"),
        mpatches.Patch(color=C_ORANGE, label="Week 1 — B & F (best coverage)"),
        mpatches.Patch(color=C_RED,    label="Week 2 — B & F (best coverage)"),
    ]
    ax_bar.legend(handles=legend_patches, fontsize=7.5, loc="lower right",
                  framealpha=0.9, ncol=2)

    out = os.path.join(SLIDES_DIR, "slide_policy_coverage_table.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Slide 2 — Scatter: Coverage vs Travel Distance
# ═══════════════════════════════════════════════════════════════════════════════

def make_coverage_vs_travel_slide(df: pd.DataFrame):
    # Average across weeks
    avg = df.groupby("Policy")[["Coverage (%)", "Total Travel (m)",
                                  "Avg Switches/Provider-Day"]].mean().reset_index()

    order = [
        "A: Single Room",
        "B: Cluster",
        "C: Cluster + Duration Buffer",
        "D: Cluster + No-show",
        "E: Day Blocking",
        "F: Admin Buffer",
    ]
    short_label = {
        "A: Single Room":                 "Policy A",
        "B: Cluster":                     "Policy B",
        "C: Cluster + Duration Buffer":   "Policy C",
        "D: Cluster + No-show":           "Policy D",
        "E: Day Blocking":                "Policy E",
        "F: Admin Buffer":                "Policy F ★",
    }
    colors = {
        "A: Single Room":               C_BLUE,
        "B: Cluster":                   C_TEAL,
        "C: Cluster + Duration Buffer": C_RED,
        "D: Cluster + No-show":         C_BLUE,
        "E: Day Blocking":              C_BLUE,
        "F: Admin Buffer":              C_ORANGE,
    }

    fig, ax = plt.subplots(figsize=(10, 7), facecolor="white")
    ax.set_facecolor(C_LIGHT)

    # Quadrant shading
    cov_mid   = 90
    travel_mid = avg["Total Travel (m)"].mean()
    ax.axhline(cov_mid,    color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.axvline(travel_mid, color="gray", lw=0.8, ls="--", alpha=0.5)

    for label, txt in [("Low Travel\nHigh Coverage", (0.05, 0.95)),
                        ("High Travel\nHigh Coverage", (0.95, 0.95)),
                        ("Low Travel\nLow Coverage",   (0.05, 0.05)),
                        ("High Travel\nLow Coverage",  (0.95, 0.05))]:
        ax.text(*txt, label, transform=ax.transAxes,
                ha="center" if "Low Travel" not in label else "left",
                va="top" if "High" in label.split("\n")[1] else "bottom",
                fontsize=8, color="gray", alpha=0.6)

    for _, row in avg.iterrows():
        p = row["Policy"]
        if p not in short_label:
            continue
        c = row["Coverage (%)"]
        t = row["Total Travel (m)"]
        col = colors.get(p, C_BLUE)
        is_best = p == "F: Admin Buffer"
        ax.scatter(t, c, s=220 if is_best else 140, color=col,
                   zorder=5, edgecolors="white", linewidths=1.5)
        offset_x = 3
        offset_y = 0.4
        if p == "B: Cluster":
            offset_y = -0.9
        elif p == "A: Single Room":
            offset_y = 0.5
        ax.annotate(short_label[p], (t, c),
                    xytext=(t + offset_x, c + offset_y),
                    fontsize=9, color=C_DARK, fontweight="bold" if is_best else "normal")

    ax.set_xlabel("Avg Total Travel Distance (m)", fontsize=11, color=C_DARK)
    ax.set_ylabel("Avg Coverage (%)", fontsize=11, color=C_DARK)
    ax.set_title("Policy Comparison: Coverage vs Travel Distance\n(average of Week 1 & Week 2)",
                 fontsize=12, fontweight="bold", color=C_DARK)
    ax.tick_params(colors=C_DARK)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(alpha=0.3)

    # Arrow annotation for F
    row_f = avg[avg["Policy"] == "F: Admin Buffer"].iloc[0]
    ax.annotate("", xy=(row_f["Total Travel (m)"], row_f["Coverage (%)"]),
                xytext=(row_f["Total Travel (m)"] - 15, row_f["Coverage (%)"] - 2),
                arrowprops=dict(arrowstyle="->", color=C_ORANGE, lw=1.5))

    out = os.path.join(SLIDES_DIR, "slide_coverage_vs_travel.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Slide 3 — B vs F efficiency KPIs
# ═══════════════════════════════════════════════════════════════════════════════

def make_b_vs_f_slide(df: pd.DataFrame):
    b = df[df["Policy"] == "B: Cluster"].sort_values("week")
    f = df[df["Policy"] == "F: Admin Buffer"].sort_values("week")

    weeks  = [1, 2]
    sw_b   = b["Avg Switches/Provider-Day"].tolist()
    sw_f   = f["Avg Switches/Provider-Day"].tolist()
    tr_b   = b["Total Travel (m)"].tolist()
    tr_f   = f["Total Travel (m)"].tolist()
    cov_b  = b["Coverage (%)"].tolist()
    cov_f  = f["Coverage (%)"].tolist()

    fig = plt.figure(figsize=(14, 8), facecolor="white")
    fig.patch.set_facecolor("white")

    # Title
    fig.text(0.5, 0.97,
             "Recommended — Best Coverage with Fewest Room Switches",
             ha="center", va="top", fontsize=15, fontweight="bold", color=C_DARK)
    fig.text(0.5, 0.91, "Key Findings",
             ha="center", va="top", fontsize=12, color=C_DARK)

    # Key-finding boxes
    findings = [
        ("01", "Policy C performs worst —\n10% buffer reduces capacity\nmore than it avoids conflicts"),
        ("02", "Policy D (no-show overbooking)\ngains coverage but increases\nscheduling risk"),
        ("03", "Policies B and F tied on\ncoverage (96.5% W1, 94.4% W2)"),
        ("04", "Policy F marginally outperforms B\non switches and total travel distance"),
    ]
    for k, (num, txt) in enumerate(findings):
        x = 0.03 + k * 0.235
        fig.text(x, 0.86, num, fontsize=22, fontweight="bold", color=C_TEAL,
                 va="top", transform=fig.transFigure)
        fig.text(x + 0.04, 0.86, txt, fontsize=8.5, color=C_DARK,
                 va="top", transform=fig.transFigure, wrap=True)

    # ── Bar charts ────────────────────────────────────────────────────────────
    gs = GridSpec(1, 2, figure=fig, top=0.58, bottom=0.10,
                  left=0.08, right=0.96, wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    x = np.arange(len(weeks))
    w = 0.35

    for ax, vals_b, vals_f, ylabel, title in [
        (ax1, sw_b, sw_f, "Avg Switches/Provider-Day", "Room Switches"),
        (ax2, tr_b, tr_f, "Total Travel Distance (m)", "Travel Distance"),
    ]:
        ax.set_facecolor(C_LIGHT)
        bars_b = ax.bar(x - w/2, vals_b, w, color=C_BLUE, label="Policy B", zorder=3)
        bars_f = ax.bar(x + w/2, vals_f, w, color=C_ORANGE, label="Policy F", zorder=3)

        for bar, val in zip(list(bars_b) + list(bars_f), vals_b + vals_f):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals_b+vals_f)*0.01,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9, color=C_DARK)

        ax.set_xticks(x)
        ax.set_xticklabels(["Week 1", "Week 2"], fontsize=10, color=C_DARK)
        ax.set_ylabel(ylabel, fontsize=10, color=C_DARK)
        ax.set_title(title, fontsize=11, fontweight="bold", color=C_DARK)
        ax.tick_params(colors=C_DARK)
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.legend(fontsize=9)

    # Sub-title for charts
    fig.text(0.5, 0.61, "Policy B vs Policy F — Efficiency KPIs",
             ha="center", va="top", fontsize=11, fontweight="bold", color=C_DARK)

    out = os.path.join(SLIDES_DIR, "slide_b_vs_f_efficiency.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Print coverage table to console
# ═══════════════════════════════════════════════════════════════════════════════

def print_coverage_table(df: pd.DataFrame):
    print("\n" + "="*70)
    print("  COVERAGE TABLE — All Policies, Both Weeks")
    print("="*70)

    pivot = df.pivot_table(index="Policy", columns="week", values="Coverage (%)")
    pivot.columns = ["W1 Coverage (%)", "W2 Coverage (%)"]
    pivot["Avg Coverage (%)"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("Avg Coverage (%)", ascending=False)

    print(pivot.to_string(float_format=lambda x: f"{x:.1f}"))
    print("="*70)
    best = pivot.index[0]
    print(f"\n  Best policy (by avg coverage): {best}")
    print(f"  W1: {pivot.loc[best,'W1 Coverage (%)']:.1f}%  "
          f"W2: {pivot.loc[best,'W2 Coverage (%)']:.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    df = run_all_policies()
    print_coverage_table(df)

    print("\nGenerating slide figures...")
    make_coverage_table_slide(df)
    make_coverage_vs_travel_slide(df)
    make_b_vs_f_slide(df)

    print("\n✓ All slides written to slides/")
    print("  • slides/slide_policy_coverage_table.png")
    print("  • slides/slide_coverage_vs_travel.png")
    print("  • slides/slide_b_vs_f_efficiency.png")
    print("  • slides/policy_kpi_all_weeks.csv")
