"""
Final Results Generator
═══════════════════════════════════════════════════════════════════════════
Runs all 6 scheduling policies for both weeks with the room-conflict fix,
identifies the best policy, runs the full advanced pipeline (CG → LR → TS)
for that policy, and writes everything to final_results/.

Output layout
─────────────
final_results/
  bounds/
    bounds_summary_week1.csv / week2.csv
    lr_history_week1.csv / week2.csv
    ts_history_week1.csv / week2.csv
  schedules/
    best_policy_schedule_week1.csv / week2.csv
    day_schedules/
      week1_<day>.csv, week2_<day>.csv
  gantt_charts/
    provider_view/  — one PNG per day per week
    room_view/      — one PNG per day per week
    overview/       — one overview PNG per week
  kpis/
    kpi_analysis.ipynb
    policy_comparison_all.png
  policy_comparison_kpis.csv
"""

import sys, os

# ── Path setup (run from project root) ───────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "models"))

import pandas as pd
import numpy as np
import json
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

from data_loader import (
    load_all_appointments, load_all_provider_availability,
    load_distance_matrix, ROOMS,
)
from column_generation import run_column_generation
from lagrangian import run_lagrangian_relaxation
from tabu_search import run_tabu_search
from model3 import master_to_appointments
from visualization import (
    plot_gantt_provider, plot_gantt_room,
    plot_gantt_week_overview, plot_kpi_comparison,
)
from main import (
    policy_single_room, apply_day_blocking,
    apply_admin_time_buffer, apply_noshow_adjustment, compute_kpis,
)

# ── Output directories ────────────────────────────────────────────────────────
OUT = os.path.join(ROOT, "final_results")
DIRS = [
    f"{OUT}/bounds",
    f"{OUT}/schedules/day_schedules",
    f"{OUT}/gantt_charts/provider_view",
    f"{OUT}/gantt_charts/room_view",
    f"{OUT}/gantt_charts/overview",
    f"{OUT}/kpis",
]
for d in DIRS:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Run all 6 policies for both weeks to compute updated KPIs
# ─────────────────────────────────────────────────────────────────────────────

def run_all_policies(week: int) -> tuple[dict, list[dict]]:
    """
    Returns (policy_results dict, kpi list).
    Each policy result is a DataFrame with assigned_room etc.
    """
    print(f"\n{'█' * 62}")
    print(f"  WEEK {week} — Running all 6 policies")
    print(f"{'█' * 62}")

    appts_all   = load_all_appointments()
    avail_all   = load_all_provider_availability()
    dist_matrix = load_distance_matrix()
    appts = appts_all[appts_all["week"] == week].copy()
    avail = avail_all[avail_all["week"] == week].copy()

    from data_loader import compute_noshow_rate
    noshow_rates = compute_noshow_rate(appts_all)

    results = {}
    kpis    = []

    # ── Policy A ──────────────────────────────────────────────────────────────
    print("\n[A] Single Room (baseline)...")
    res_a = policy_single_room(appts, avail, dist_matrix)
    res_a["policy"] = "A_single_room"
    results["A_single_room"] = res_a
    kpi = compute_kpis(res_a, dist_matrix, "A: Single Room")
    kpis.append(kpi)
    print(f"     Coverage: {kpi['Coverage (%)']}%  |  Switches: {kpi['Avg Switches/PD']}  |  Travel: {kpi['Total Travel (m)']}m")

    # ── Policy B ──────────────────────────────────────────────────────────────
    print("\n[B] Cluster (CG)...")
    cg_b = run_column_generation(appts, avail, dist_matrix,
                                 delta_frac=0.0, proximity_threshold=4.0)
    res_b = master_to_appointments(cg_b["master_result"], appts)
    res_b["policy"] = "B_cluster"
    results["B_cluster"] = res_b
    results["B_cluster_cg"] = cg_b
    kpi = compute_kpis(res_b, dist_matrix, "B: Cluster")
    kpis.append(kpi)
    print(f"     Coverage: {kpi['Coverage (%)']}%  |  Switches: {kpi['Avg Switches/PD']}  |  Travel: {kpi['Total Travel (m)']}m")

    # ── Policy C ──────────────────────────────────────────────────────────────
    print("\n[C] Robust Buffer (10% buffer)...")
    cg_c = run_column_generation(appts, avail, dist_matrix,
                                 delta_frac=0.10, proximity_threshold=4.0)
    res_c = master_to_appointments(cg_c["master_result"], appts)
    res_c["policy"] = "C_robust_buffer"
    results["C_robust_buffer"] = res_c
    kpi = compute_kpis(res_c, dist_matrix, "C: Robust Buffer")
    kpis.append(kpi)
    print(f"     Coverage: {kpi['Coverage (%)']}%  |  Switches: {kpi['Avg Switches/PD']}  |  Travel: {kpi['Total Travel (m)']}m")

    # ── Policy D ──────────────────────────────────────────────────────────────
    print("\n[D] No-show Robust...")
    appts_ns = apply_noshow_adjustment(appts, noshow_rates)
    appts_ns["duration_min"] = appts_ns["effective_duration"].round().astype(int)
    appts_ns["end_min"]      = appts_ns["start_min"] + appts_ns["duration_min"]
    cg_d = run_column_generation(appts_ns, avail, dist_matrix,
                                 delta_frac=0.10, proximity_threshold=4.0)
    res_d = master_to_appointments(cg_d["master_result"], appts)
    res_d["policy"] = "D_robust_noshow"
    results["D_robust_noshow"] = res_d
    kpi = compute_kpis(res_d, dist_matrix, "D: No-show Robust")
    kpis.append(kpi)
    print(f"     Coverage: {kpi['Coverage (%)']}%  |  Switches: {kpi['Avg Switches/PD']}  |  Travel: {kpi['Total Travel (m)']}m")

    # ── Policy E ──────────────────────────────────────────────────────────────
    print("\n[E] Day Blocking...")
    appts_blocked = apply_day_blocking(appts, avail)
    cg_e = run_column_generation(appts_blocked, avail, dist_matrix,
                                 delta_frac=0.0, proximity_threshold=4.0)
    res_e = master_to_appointments(cg_e["master_result"], appts)
    res_e["policy"] = "E_day_blocking"
    results["E_day_blocking"] = res_e
    kpi = compute_kpis(res_e, dist_matrix, "E: Day Blocking")
    kpis.append(kpi)
    print(f"     Coverage: {kpi['Coverage (%)']}%  |  Switches: {kpi['Avg Switches/PD']}  |  Travel: {kpi['Total Travel (m)']}m")

    # ── Policy F ──────────────────────────────────────────────────────────────
    print("\n[F] Admin Buffer (CG)...")
    appts_admin = apply_admin_time_buffer(appts)
    cg_f = run_column_generation(appts_admin, avail, dist_matrix,
                                 delta_frac=0.0, proximity_threshold=4.0)
    res_f = master_to_appointments(cg_f["master_result"], appts)
    res_f["policy"] = "F_admin_buffer"
    results["F_admin_buffer"] = res_f
    results["F_admin_buffer_cg"] = cg_f
    kpi = compute_kpis(res_f, dist_matrix, "F: Admin Buffer")
    kpis.append(kpi)
    print(f"     Coverage: {kpi['Coverage (%)']}%  |  Switches: {kpi['Avg Switches/PD']}  |  Travel: {kpi['Total Travel (m)']}m")

    return results, kpis


def select_best_policy(kpis_w1: list[dict], kpis_w2: list[dict]) -> str:
    """
    Pick the best policy:
      1. Highest coverage (both weeks combined)
      2. Fewest average switches/PD
      3. Least total travel
      4. Fewest rooms used
    Returns the Policy label of the winner.
    """
    w1 = {k["Policy"]: k for k in kpis_w1}
    w2 = {k["Policy"]: k for k in kpis_w2}
    all_policies = list(w1.keys())

    scores = []
    for p in all_policies:
        k1, k2 = w1[p], w2[p]
        avg_cov     = (k1["Coverage (%)"]        + k2["Coverage (%)"])        / 2
        avg_sw      = (k1["Avg Switches/PD"]      + k2["Avg Switches/PD"])      / 2
        avg_travel  = (k1["Total Travel (m)"]     + k2["Total Travel (m)"])     / 2
        avg_rooms   = (k1["Rooms Used"]           + k2["Rooms Used"])           / 2
        scores.append((p, avg_cov, avg_sw, avg_travel, avg_rooms))

    # Sort: coverage desc, switches asc, travel asc, rooms asc
    scores.sort(key=lambda x: (-x[1], x[2], x[3], x[4]))
    winner = scores[0][0]

    print("\n" + "═" * 62)
    print("  POLICY RANKING (coverage ↓ → switches ↑ → travel ↑ → rooms ↑)")
    print("═" * 62)
    headers = f"  {'Policy':<25} {'Cov%':>6}  {'Sw/PD':>6}  {'Travel':>8}  {'Rooms':>5}"
    print(headers)
    print("  " + "─" * 58)
    for rank, (p, cov, sw, trav, rooms) in enumerate(scores, 1):
        star = " ← BEST" if rank == 1 else ""
        print(f"  {p:<25} {cov:>6.1f}  {sw:>6.2f}  {trav:>8.1f}  {rooms:>5.1f}{star}")
    print("═" * 62)

    return winner


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Advanced pipeline for best policy
# ─────────────────────────────────────────────────────────────────────────────

def run_advanced_for_policy(
    policy_key: str,
    week: int,
    appts_all: pd.DataFrame,
    avail_all: pd.DataFrame,
    dist_matrix: pd.DataFrame,
) -> dict:
    """
    Runs CG + LR + TS for the given policy / week.
    Returns dict with z_lp, z_ilp, z_lr, z_ts, gap_pct,
    master_result, ts_result, lr_result.
    """
    from data_loader import compute_noshow_rate

    appts = appts_all[appts_all["week"] == week].copy()
    avail = avail_all[avail_all["week"] == week].copy()

    # Apply policy-specific data transformations
    if policy_key == "F_admin_buffer":
        appts_sched = apply_admin_time_buffer(appts)
        delta_frac  = 0.0
    elif policy_key == "B_cluster":
        appts_sched = appts.copy()
        delta_frac  = 0.0
    elif policy_key == "C_robust_buffer":
        appts_sched = appts.copy()
        delta_frac  = 0.10
    elif policy_key == "D_robust_noshow":
        noshow_rates = compute_noshow_rate(appts_all)
        appts_sched  = apply_noshow_adjustment(appts, noshow_rates)
        appts_sched["duration_min"] = appts_sched["effective_duration"].round().astype(int)
        appts_sched["end_min"]      = appts_sched["start_min"] + appts_sched["duration_min"]
        delta_frac = 0.10
    elif policy_key == "E_day_blocking":
        appts_sched = apply_day_blocking(appts, avail)
        delta_frac  = 0.0
    else:
        appts_sched = appts.copy()
        delta_frac  = 0.0

    # ── Column Generation ─────────────────────────────────────────────────────
    cg = run_column_generation(
        appts_sched, avail, dist_matrix,
        delta_frac=delta_frac,
        proximity_threshold=4.0,
        max_iterations=3,
    )
    z_lp  = cg["lp_bound"]
    z_ilp = cg["ilp_bound"]

    # ── Lagrangian Relaxation ─────────────────────────────────────────────────
    lr = run_lagrangian_relaxation(
        cg["schedules"], appts_sched,
        z_ub=z_ilp,
        max_iterations=50,
        verbose=False,
    )
    z_lr = lr["best_lower_bound"]

    # ── Tabu Search ───────────────────────────────────────────────────────────
    selected = cg["master_result"].get("selected_schedules", [])
    if selected:
        ts = run_tabu_search(
            selected, appts_sched, avail, dist_matrix,
            proximity_threshold=4.0,
            k_max=20,
            max_iterations=150,
            max_no_improve=50,
            verbose=False,
        )
        z_ts = ts["best_cost"]
    else:
        ts   = None
        z_ts = z_ilp

    gap_pct = (z_ts - z_lr) / abs(z_ts) * 100 if z_ts not in (0, float("inf")) else None

    print(f"\n  Week {week} bounds:")
    print(f"    z_LR  (Lagrangian LB) = {z_lr:.4f}")
    print(f"    z_LP  (CG LP bound)   = {z_lp:.4f}")
    print(f"    z_ILP (CG ILP)        = {z_ilp:.4f}")
    print(f"    z_TS  (Tabu Search)   = {z_ts:.4f}")
    if gap_pct is not None:
        print(f"    Duality Gap           = {gap_pct:.2f}%")

    return {
        "z_lp": z_lp, "z_ilp": z_ilp, "z_lr": z_lr, "z_ts": z_ts,
        "gap_pct": gap_pct,
        "cg_result": cg,
        "lr_result": lr,
        "ts_result": ts,
        "appts_original": appts,
        "appts_sched": appts_sched,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Save schedules
# ─────────────────────────────────────────────────────────────────────────────

def save_schedules(result_df: pd.DataFrame, week: int, policy_key: str):
    """Save full-week schedule and per-day CSVs."""
    fname = f"{OUT}/schedules/best_policy_schedule_week{week}.csv"
    result_df.to_csv(fname, index=False)
    print(f"  Saved: {fname}")

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    for day in days:
        day_df = result_df[result_df["day_of_week"] == day]
        if day_df.empty:
            continue
        fname = f"{OUT}/schedules/day_schedules/week{week}_{day}.csv"
        day_df.to_csv(fname, index=False)
        print(f"  Saved: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Save Gantt charts
# ─────────────────────────────────────────────────────────────────────────────

def save_gantts(result_df: pd.DataFrame, week: int, policy_key: str):
    """Generate and save all Gantt charts for one week."""
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    short = policy_key.replace("_", "-")

    for day in days:
        day_df = result_df[result_df["day_of_week"] == day]
        if day_df.empty:
            continue
        # Provider view
        plot_gantt_provider(
            result_df, week=week, policy=short, day_filter=day,
            save_path=f"{OUT}/gantt_charts/provider_view/week{week}_{day}_provider.png",
        )
        # Room view
        plot_gantt_room(
            result_df, week=week, policy=short, day_filter=day,
            save_path=f"{OUT}/gantt_charts/room_view/week{week}_{day}_room.png",
        )

    # Overview
    plot_gantt_week_overview(
        result_df, week=week, policy=short,
        save_path=f"{OUT}/gantt_charts/overview/week{week}_overview.png",
    )


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Save bounds
# ─────────────────────────────────────────────────────────────────────────────

def save_bounds(adv: dict, week: int):
    """Save bounds CSV and history CSVs."""
    bounds_df = pd.DataFrame([{
        "z_LR (Lagrangian LB)": adv["z_lr"],
        "z_LP (CG LP bound)":   adv["z_lp"],
        "z_ILP (CG ILP)":       adv["z_ilp"],
        "z_TS (Tabu Search)":   adv["z_ts"],
        "Duality Gap (%)":      adv["gap_pct"],
    }])
    bounds_df.to_csv(f"{OUT}/bounds/bounds_summary_week{week}.csv", index=False)

    lr_hist = adv["lr_result"].get("history")
    if lr_hist is not None:
        lr_hist.to_csv(f"{OUT}/bounds/lr_history_week{week}.csv", index=False)

    ts = adv["ts_result"]
    if ts is not None:
        ts_hist = ts.get("history")
        if ts_hist is not None:
            ts_hist.to_csv(f"{OUT}/bounds/ts_history_week{week}.csv", index=False)

    print(f"  Saved bounds for week {week}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Build KPI notebook
# ─────────────────────────────────────────────────────────────────────────────

def build_kpi_notebook(
    kpis_w1: list[dict],
    kpis_w2: list[dict],
    best_policy: str,
    best_key: str,
    adv_w1: dict,
    adv_w2: dict,
):
    """Create a self-contained Jupyter notebook with all KPI figures."""
    nb = new_notebook()
    cells = []

    # ── Title ──────────────────────────────────────────────────────────────────
    cells.append(new_markdown_cell(f"""# Examination Room Scheduling — Final KPI Analysis

**Best Policy Selected:** {best_policy}

This notebook presents all Key Performance Indicators for all 6 scheduling policies
across both weeks, with deep-dive analysis for the selected best policy.

**Bounds computed (CG → LR → TS pipeline):**
| Bound | Week 1 | Week 2 |
|-------|--------|--------|
| z_LR (Lagrangian LB) | {adv_w1['z_lr']:.2f} | {adv_w2['z_lr']:.2f} |
| z_LP (CG LP bound)   | {adv_w1['z_lp']:.2f} | {adv_w2['z_lp']:.2f} |
| z_ILP (CG ILP)       | {adv_w1['z_ilp']:.2f} | {adv_w2['z_ilp']:.2f} |
| z_TS (Tabu Search)   | {adv_w1['z_ts']:.2f} | {adv_w2['z_ts']:.2f} |
| Duality Gap (%)      | {adv_w1['gap_pct']:.2f}% | {adv_w2['gap_pct']:.2f}% |
"""))

    # ── Setup cell ──────────────────────────────────────────────────────────────
    cells.append(new_code_cell("""\
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.getcwd(), '..', '..', 'models'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm

# ── Load pre-computed results ─────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath('__file__')) if '__file__' in dir() else os.getcwd()
FINAL = os.path.join(HERE, '..', '..', 'final_results')

kpis_w1 = pd.read_csv(f'{FINAL}/policy_comparison_kpis_week1.csv')
kpis_w2 = pd.read_csv(f'{FINAL}/policy_comparison_kpis_week2.csv')
sched_w1 = pd.read_csv(f'{FINAL}/schedules/best_policy_schedule_week1.csv')
sched_w2 = pd.read_csv(f'{FINAL}/schedules/best_policy_schedule_week2.csv')
bounds_w1 = pd.read_csv(f'{FINAL}/bounds/bounds_summary_week1.csv')
bounds_w2 = pd.read_csv(f'{FINAL}/bounds/bounds_summary_week2.csv')

print("Data loaded successfully.")
print(f"  Week 1: {len(sched_w1)} appointments")
print(f"  Week 2: {len(sched_w2)} appointments")
"""))

    # ── Section 1: Policy comparison ──────────────────────────────────────────
    cells.append(new_markdown_cell("## 1. Policy Comparison — All 6 Policies, Both Weeks"))

    cells.append(new_code_cell("""\
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Policy Comparison — Key Performance Indicators', fontsize=15, fontweight='bold')

metrics = [
    ('Coverage (%)',          'Coverage (%)',            True),
    ('Avg Switches/PD',       'Avg Room Switches / Provider-Day', False),
    ('Total Travel (m)',      'Total Travel Distance (m)',        False),
    ('Rooms Used',            'Distinct Rooms Used',              False),
]
weeks  = [kpis_w1, kpis_w2]
titles = ['Week 1', 'Week 2']
colors = plt.cm.Set2(np.linspace(0, 1, len(kpis_w1)))

for col_i, (metric, label, higher_better) in enumerate(metrics):
    for row_i, (df, wtitle) in enumerate(zip(weeks, titles)):
        ax = axes[row_i][col_i]
        vals = df[metric]
        bars = ax.bar(df['Policy'], vals, color=colors, edgecolor='black', linewidth=0.7)
        ax.set_title(f'{label}\\n{wtitle}', fontsize=9)
        ax.set_xticklabels(df['Policy'], rotation=30, ha='right', fontsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01*(vals.max() or 1),
                    f'{v:.1f}', ha='center', va='bottom', fontsize=8)
        if higher_better:
            ax.set_ylim(0, 115)

plt.tight_layout()
plt.savefig(f'{FINAL}/kpis/policy_comparison_all.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved policy_comparison_all.png")
"""))

    # ── Section 2: KPI table ───────────────────────────────────────────────────
    cells.append(new_markdown_cell("## 2. Full KPI Tables"))

    cells.append(new_code_cell("""\
print("=== WEEK 1 ===")
display(kpis_w1.set_index('Policy'))
print("\\n=== WEEK 2 ===")
display(kpis_w2.set_index('Policy'))
"""))

    # ── Section 3: Best policy deep-dive ──────────────────────────────────────
    cells.append(new_markdown_cell(f"## 3. Best Policy Deep-Dive: {best_policy}"))

    cells.append(new_code_cell("""\
# Combine both weeks for deep-dive
sched_all = pd.concat([sched_w1, sched_w2], ignore_index=True)
scheduled = sched_all[sched_all['assigned_room'].notna()].copy()

# ── 3a: Coverage by day and week ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
colors_days = plt.cm.Pastel1(np.linspace(0, 1, 5))

for ax, (df, label) in zip(axes, [(sched_w1, 'Week 1'), (sched_w2, 'Week 2')]):
    day_cov = []
    for d in days_order:
        ddf = df[df['day_of_week'] == d]
        if len(ddf) == 0:
            day_cov.append(0)
        else:
            day_cov.append(ddf['assigned_room'].notna().mean() * 100)
    bars = ax.bar(days_order, day_cov, color=colors_days, edgecolor='black', linewidth=0.7)
    ax.set_title(f'Coverage by Day — {label}', fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_ylabel('Coverage (%)')
    for bar, v in zip(bars, day_cov):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{v:.0f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f'{FINAL}/kpis/coverage_by_day.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

    cells.append(new_code_cell("""\
# ── 3b: Switches and travel per provider-day ─────────────────────────────────
if 'num_switches' in sched_all.columns and 'total_travel' in sched_all.columns:
    pd_stats_all = (sched_all
        .groupby(['provider', 'day_of_week', 'week'])
        .agg(num_switches=('num_switches', 'first'),
             total_travel=('total_travel', 'first'))
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Switches distribution
    ax = axes[0]
    w1_sw = pd_stats_all[pd_stats_all['week'] == 1]['num_switches']
    w2_sw = pd_stats_all[pd_stats_all['week'] == 2]['num_switches']
    ax.hist([w1_sw, w2_sw], bins=range(0, int(max(w1_sw.max(), w2_sw.max())) + 2),
            label=['Week 1', 'Week 2'], color=['steelblue', 'coral'],
            edgecolor='black', alpha=0.8, rwidth=0.8)
    ax.set_title('Room Switches per Provider-Day Distribution', fontsize=11)
    ax.set_xlabel('Number of Room Switches')
    ax.set_ylabel('Count of Provider-Days')
    ax.legend()
    ax.set_xticks(range(0, int(max(w1_sw.max(), w2_sw.max())) + 2))

    # Travel distribution
    ax = axes[1]
    w1_tr = pd_stats_all[pd_stats_all['week'] == 1]['total_travel']
    w2_tr = pd_stats_all[pd_stats_all['week'] == 2]['total_travel']
    max_tr = max(w1_tr.max(), w2_tr.max())
    bins = np.linspace(0, max_tr + 1, 20)
    ax.hist([w1_tr, w2_tr], bins=bins, label=['Week 1', 'Week 2'],
            color=['steelblue', 'coral'], edgecolor='black', alpha=0.8)
    ax.set_title('Total Travel per Provider-Day Distribution', fontsize=11)
    ax.set_xlabel('Travel Distance (m)')
    ax.set_ylabel('Count of Provider-Days')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{FINAL}/kpis/switches_travel_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
"""))

    cells.append(new_code_cell("""\
# ── 3c: Room utilisation heatmap (rooms × days) ───────────────────────────────
for week_num, sched in [(1, sched_w1), (2, sched_w2)]:
    sched_ok = sched[sched['assigned_room'].notna()].copy()
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

    # minutes used per (room, day)
    room_day_min = (sched_ok.groupby(['assigned_room', 'day_of_week'])
                    ['duration_min'].sum()
                    .unstack(fill_value=0)
                    .reindex(columns=[d for d in days_order if d in sched_ok['day_of_week'].unique()],
                             fill_value=0))

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(room_day_min.values, aspect='auto', cmap='YlOrRd')
    plt.colorbar(im, ax=ax, label='Occupied Minutes')
    ax.set_xticks(range(len(room_day_min.columns)))
    ax.set_xticklabels(room_day_min.columns, fontsize=9)
    ax.set_yticks(range(len(room_day_min.index)))
    ax.set_yticklabels(room_day_min.index, fontsize=9)
    ax.set_title(f'Room Utilisation Heatmap — Week {week_num}', fontsize=12)
    for i in range(len(room_day_min.index)):
        for j in range(len(room_day_min.columns)):
            v = room_day_min.values[i, j]
            ax.text(j, i, f'{v:.0f}', ha='center', va='center',
                    fontsize=8, color='black' if v < room_day_min.values.max()*0.6 else 'white')
    plt.tight_layout()
    plt.savefig(f'{FINAL}/kpis/room_utilisation_heatmap_week{week_num}.png', dpi=150, bbox_inches='tight')
    plt.show()
"""))

    cells.append(new_code_cell("""\
# ── 3d: Provider compactness by provider ─────────────────────────────────────
for week_num, sched in [(1, sched_w1), (2, sched_w2)]:
    sched_ok = sched[sched['assigned_room'].notna()].copy()
    rooms_per_pd = (sched_ok.groupby(['provider', 'day_of_week'])
                   ['assigned_room'].nunique().reset_index())
    rooms_per_pd.columns = ['provider', 'day', 'rooms_used']
    provider_compact = (rooms_per_pd.groupby('provider')
                        .apply(lambda g: (g['rooms_used'] == 1).mean() * 100)
                        .sort_values(ascending=False))

    fig, ax = plt.subplots(figsize=(16, 5))
    colors_compact = ['#2ecc71' if v == 100 else '#e74c3c' if v < 50 else '#f39c12'
                      for v in provider_compact.values]
    ax.bar(provider_compact.index, provider_compact.values,
           color=colors_compact, edgecolor='black', linewidth=0.6)
    ax.axhline(100, color='green', linestyle='--', linewidth=1, alpha=0.5, label='100% compact')
    ax.set_title(f'Provider Compactness (% of days in single room) — Week {week_num}', fontsize=11)
    ax.set_xlabel('Provider')
    ax.set_ylabel('Compactness (%)')
    ax.set_xticklabels(provider_compact.index, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 115)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{FINAL}/kpis/provider_compactness_week{week_num}.png', dpi=150, bbox_inches='tight')
    plt.show()
"""))

    cells.append(new_code_cell("""\
# ── 3e: Peak concurrent rooms by time-of-day ─────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(25, 10), sharey=True)
fig.suptitle('Concurrent Rooms in Use by Time of Day', fontsize=13, fontweight='bold')

days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
for row_i, (sched, wlabel) in enumerate([(sched_w1, 'Week 1'), (sched_w2, 'Week 2')]):
    sched_ok = sched[sched['assigned_room'].notna()].copy()
    for col_i, day in enumerate(days_order):
        ax = axes[row_i][col_i]
        day_df = sched_ok[sched_ok['day_of_week'] == day]
        if day_df.empty:
            ax.set_title(f'{day}\\n{wlabel}', fontsize=8)
            continue
        time_range = range(int(day_df['start_min'].min()), int(day_df['end_min'].max()) + 1)
        concurrent = []
        times = []
        for t in time_range:
            n = ((day_df['start_min'] <= t) & (day_df['end_min'] > t)).sum()
            concurrent.append(n)
            times.append(t)
        ax.fill_between(times, concurrent, alpha=0.6, color='steelblue')
        ax.plot(times, concurrent, color='navy', linewidth=0.8)
        ax.axvline(720, color='red', linestyle='--', linewidth=0.8, alpha=0.5)  # lunch
        ax.set_title(f'{day}\\n{wlabel}', fontsize=8)
        ax.set_xlabel('Time (min)', fontsize=7)
        if col_i == 0:
            ax.set_ylabel('Concurrent Rooms', fontsize=8)
        x_ticks = range(480, 1021, 120)
        ax.set_xticks(list(x_ticks))
        ax.set_xticklabels([f'{t//60:02d}:{t%60:02d}' for t in x_ticks], fontsize=6, rotation=45)

plt.tight_layout()
plt.savefig(f'{FINAL}/kpis/concurrent_rooms_by_time.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

    # ── Section 4: Bounds ─────────────────────────────────────────────────────
    cells.append(new_markdown_cell("## 4. Optimality Bounds (CG → LR → TS Pipeline)"))

    cells.append(new_code_cell("""\
# Bounds comparison
bounds_combined = pd.concat([
    bounds_w1.assign(Week='Week 1'),
    bounds_w2.assign(Week='Week 2'),
])
print("Bounds Summary:")
display(bounds_combined.set_index('Week'))

# Plot bounds hierarchy
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Optimality Bounds — CG → LR → TS Pipeline', fontsize=13, fontweight='bold')

bound_cols = ['z_LR (Lagrangian LB)', 'z_LP (CG LP bound)', 'z_ILP (CG ILP)', 'z_TS (Tabu Search)']
labels     = ['z_LR\\n(Lagrangian LB)', 'z_LP\\n(CG LP bound)', 'z_ILP\\n(CG ILP)', 'z_TS\\n(Tabu Search)']
colors_bnd = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

for ax, (bdf, wlabel) in zip(axes, [(bounds_w1, 'Week 1'), (bounds_w2, 'Week 2')]):
    vals = [bdf[c].values[0] for c in bound_cols]
    bars = ax.bar(labels, vals, color=colors_bnd, edgecolor='black', linewidth=0.8, width=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                f'{v:.1f}', ha='center', va='bottom', fontsize=9)
    gap = bdf['Duality Gap (%)'].values[0]
    ax.set_title(f'{wlabel}   |   Duality Gap = {gap:.2f}%', fontsize=11)
    ax.set_ylabel('Objective Value')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

plt.tight_layout()
plt.savefig(f'{FINAL}/kpis/bounds_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

    # ── Section 5: Tabu search convergence ───────────────────────────────────
    cells.append(new_markdown_cell("## 5. Tabu Search Convergence"))

    cells.append(new_code_cell("""\
for week_num in [1, 2]:
    try:
        ts_hist = pd.read_csv(f'{FINAL}/bounds/ts_history_week{week_num}.csv')
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(ts_hist.index, ts_hist.iloc[:, 0] if 'cost' not in ts_hist.columns
                else ts_hist['cost'], label='Current cost', alpha=0.7, linewidth=1)
        if 'best_cost' in ts_hist.columns:
            ax.plot(ts_hist.index, ts_hist['best_cost'], label='Best cost', linewidth=1.5, color='red')
        ax.set_title(f'Tabu Search Convergence — Week {week_num}', fontsize=11)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective Value')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'{FINAL}/kpis/ts_convergence_week{week_num}.png', dpi=150, bbox_inches='tight')
        plt.show()
    except FileNotFoundError:
        print(f"TS history for week {week_num} not found.")
"""))

    # ── Section 6: Gantt chart embeds ─────────────────────────────────────────
    cells.append(new_markdown_cell("## 6. Schedule Gantt Charts\n\nGantt charts saved in `final_results/gantt_charts/`. Preview overviews below."))

    cells.append(new_code_cell("""\
from IPython.display import Image, display as ipy_display
for week_num in [1, 2]:
    fpath = f'{FINAL}/gantt_charts/overview/week{week_num}_overview.png'
    if os.path.exists(fpath):
        print(f"\\nWeek {week_num} Overview:")
        ipy_display(Image(filename=fpath, width=1200))
"""))

    # ── Section 7: Additional KPIs ────────────────────────────────────────────
    cells.append(new_markdown_cell("## 7. Additional KPIs"))

    cells.append(new_code_cell("""\
# No-show appointments (if tracked)
for week_num, sched in [(1, sched_w1), (2, sched_w2)]:
    print(f"\\n=== Week {week_num} ===")
    total = len(sched)
    scheduled_count = sched['assigned_room'].notna().sum()
    unscheduled = total - scheduled_count

    # Appointments per day
    day_counts = sched.groupby('day_of_week').size().reindex(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'], fill_value=0)
    print(f"  Total appointments      : {total}")
    print(f"  Scheduled               : {scheduled_count} ({scheduled_count/total*100:.1f}%)")
    print(f"  Unscheduled             : {unscheduled} ({unscheduled/total*100:.1f}%)")
    print(f"  Unique providers        : {sched['provider'].nunique()}")
    print(f"  Unique rooms used       : {sched['assigned_room'].nunique()}")
    print(f"  Appointments by day:")
    for day, cnt in day_counts.items():
        print(f"    {day:<12}: {cnt}")
"""))

    cells.append(new_code_cell("""\
# Avg appointments per provider per day
for week_num, sched in [(1, sched_w1), (2, sched_w2)]:
    sched_ok = sched[sched['assigned_room'].notna()]
    appts_per_pd = sched_ok.groupby(['provider', 'day_of_week']).size()
    print(f"\\nWeek {week_num} — Appointments per Provider-Day:")
    print(f"  Mean : {appts_per_pd.mean():.1f}")
    print(f"  Max  : {appts_per_pd.max()}")
    print(f"  Min  : {appts_per_pd.min()}")

    fig, ax = plt.subplots(figsize=(12, 4))
    appts_per_pd.hist(ax=ax, bins=range(1, appts_per_pd.max() + 2),
                      color='steelblue', edgecolor='black', rwidth=0.8)
    ax.set_title(f'Appointments per Provider-Day — Week {week_num}', fontsize=11)
    ax.set_xlabel('Number of Appointments')
    ax.set_ylabel('Count of Provider-Days')
    plt.tight_layout()
    plt.savefig(f'{FINAL}/kpis/appts_per_providerday_week{week_num}.png',
                dpi=150, bbox_inches='tight')
    plt.show()
"""))

    nb.cells = cells
    nb_path = f"{OUT}/kpis/kpi_analysis.ipynb"
    with open(nb_path, "w") as f:
        nbformat.write(nb, f)
    print(f"  Saved notebook: {nb_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "█" * 62)
    print("  FINAL RESULTS GENERATOR")
    print("█" * 62)

    # ── Phase 1: Compare all policies ─────────────────────────────────────────
    print("\n" + "─" * 62)
    print("  PHASE 1: Running all 6 policies for both weeks")
    print("─" * 62)
    results_w1, kpis_w1 = run_all_policies(week=1)
    results_w2, kpis_w2 = run_all_policies(week=2)

    # Save comparison KPI tables
    pd.DataFrame(kpis_w1).to_csv(f"{OUT}/policy_comparison_kpis_week1.csv", index=False)
    pd.DataFrame(kpis_w2).to_csv(f"{OUT}/policy_comparison_kpis_week2.csv", index=False)

    # ── Phase 2: Identify best policy ─────────────────────────────────────────
    best_policy_label = select_best_policy(kpis_w1, kpis_w2)
    # Map label back to key
    label_to_key = {
        "A: Single Room":    "A_single_room",
        "B: Cluster":        "B_cluster",
        "C: Robust Buffer":  "C_robust_buffer",
        "D: No-show Robust": "D_robust_noshow",
        "E: Day Blocking":   "E_day_blocking",
        "F: Admin Buffer":   "F_admin_buffer",
    }
    best_key = label_to_key.get(best_policy_label, "F_admin_buffer")

    print(f"\n  Best policy: {best_policy_label}  (key: {best_key})")

    # ── Phase 3: Full advanced pipeline for best policy ───────────────────────
    print("\n" + "─" * 62)
    print(f"  PHASE 3: Advanced pipeline for {best_policy_label}")
    print("─" * 62)

    appts_all   = load_all_appointments()
    avail_all   = load_all_provider_availability()
    dist_matrix = load_distance_matrix()

    adv_w1 = run_advanced_for_policy(best_key, 1, appts_all, avail_all, dist_matrix)
    adv_w2 = run_advanced_for_policy(best_key, 2, appts_all, avail_all, dist_matrix)

    # Build result DataFrames from the CG ILP result (already conflict-free via
    # resolve_room_conflicts).  We do NOT use the raw TS assignment because TS
    # only penalises conflicts rather than eliminating them, so its best_df can
    # still contain room double-bookings.
    def build_result_df(adv, appts_all, best_key, week):
        appts = appts_all[appts_all["week"] == week].copy()
        res = master_to_appointments(adv["cg_result"]["master_result"], appts)
        res["policy"] = best_key
        return res

    res_w1 = build_result_df(adv_w1, appts_all, best_key, week=1)
    res_w2 = build_result_df(adv_w2, appts_all, best_key, week=2)

    # ── Phase 4: Save schedules ────────────────────────────────────────────────
    print("\n  Saving schedules...")
    save_schedules(res_w1, week=1, policy_key=best_key)
    save_schedules(res_w2, week=2, policy_key=best_key)

    # ── Phase 5: Save Gantt charts ────────────────────────────────────────────
    print("\n  Generating Gantt charts...")
    save_gantts(res_w1, week=1, policy_key=best_key)
    save_gantts(res_w2, week=2, policy_key=best_key)

    # ── Phase 6: Save bounds ──────────────────────────────────────────────────
    print("\n  Saving bounds...")
    save_bounds(adv_w1, week=1)
    save_bounds(adv_w2, week=2)

    # ── Phase 7: Build Jupyter notebook ──────────────────────────────────────
    print("\n  Building KPI notebook...")
    build_kpi_notebook(kpis_w1, kpis_w2, best_policy_label, best_key, adv_w1, adv_w2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "█" * 62)
    print("  COMPLETE — final_results/ is ready")
    print("█" * 62)
    print(f"\n  Best policy : {best_policy_label}")
    print(f"\n  Contents of final_results/:")
    for d, _, files in os.walk(OUT):
        level = d.replace(OUT, '').count(os.sep)
        indent = '  ' * (level + 1)
        print(f'{indent}{os.path.basename(d)}/')
        subindent = '  ' * (level + 2)
        for f in sorted(files):
            print(f'{subindent}{f}')


if __name__ == "__main__":
    main()
