"""
generate_policy_f_gantts.py
────────────────────────────
Runs Policy F (admin time as overflow buffer) for Weeks 1 & 2 with the
fixed model (no provider conflicts), then saves:

  slides/F_gantt_charts/gantt_provider_<day>_W<week>.png
  slides/F_gantt_charts/gantt_room_<day>_W<week>.png

One chart per day per view per week (10 provider + 10 room = 20 files).
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "models"))
os.chdir(ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import pandas as pd

from data_loader import (
    load_all_appointments,
    load_all_provider_availability,
    load_distance_matrix,
    minutes_to_str,
    ROOMS,
)
from main import apply_admin_time_buffer
from model2 import generate_schedules_sequential
from model3 import build_master_problem, master_to_appointments

OUT_DIR = os.path.join(ROOT, "slides", "F_gantt_charts")
os.makedirs(OUT_DIR, exist_ok=True)

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
ADMIN_SHADING = [(540, 570), (690, 720), (720, 780), (990, 1020)]  # weekday blocks


# ── colour helpers ────────────────────────────────────────────────────────────

def _room_colors(rooms):
    cmap = cm.get_cmap("tab20", max(len(rooms), 1))
    return {r: cmap(i) for i, r in enumerate(rooms)}


def _provider_colors(providers):
    cmap = cm.get_cmap("tab20b", max(len(providers), 1))
    return {p: cmap(i) for i, p in enumerate(providers)}


# ── Gantt: provider view ──────────────────────────────────────────────────────

def plot_provider(df: pd.DataFrame, day: str, week: int):
    day_df = df[(df["day_of_week"] == day) & df["assigned_room"].notna()].copy()
    if day_df.empty:
        return

    providers  = sorted(day_df["provider"].unique())
    rooms_used = sorted(day_df["assigned_room"].dropna().unique())
    color_map  = _room_colors(rooms_used)

    fig, ax = plt.subplots(figsize=(16, max(6, len(providers) * 0.55)))
    ax.set_facecolor("#F7FBFF")
    ax.set_title(f"Provider Schedule — {day}, Week {week}  [Policy F: Admin Buffer]",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Provider", fontsize=10)

    for i, prov in enumerate(providers):
        for _, row in day_df[day_df["provider"] == prov].iterrows():
            room  = row["assigned_room"]
            color = color_map.get(room, "grey")
            ax.barh(i, row["duration_min"], left=row["start_min"],
                    color=color, edgecolor="white", linewidth=0.6, height=0.72)
            if row["duration_min"] >= 12:
                ax.text(row["start_min"] + 1.5, i,
                        str(room).replace("ER", "R"),
                        va="center", fontsize=6.5, color="white", fontweight="bold")

    for s, e in ADMIN_SHADING:
        ax.axvspan(s, e, alpha=0.07, color="red")

    ax.set_yticks(range(len(providers)))
    ax.set_yticklabels(providers, fontsize=8)

    x_ticks = list(range(480, 1021, 30))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([minutes_to_str(t) for t in x_ticks], rotation=45, fontsize=7)
    ax.set_xlim(480, 1020)

    patches = [mpatches.Patch(color=color_map[r], label=r) for r in rooms_used]
    ax.legend(handles=patches, loc="upper right", fontsize=7,
              ncol=4, title="Room", framealpha=0.8)

    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25, linestyle="--")

    plt.tight_layout()
    fname = os.path.join(OUT_DIR, f"gantt_provider_{day}_W{week}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ── Gantt: room view ──────────────────────────────────────────────────────────

def plot_room(df: pd.DataFrame, day: str, week: int):
    day_df = df[(df["day_of_week"] == day) & df["assigned_room"].notna()].copy()
    if day_df.empty:
        return

    providers   = sorted(day_df["provider"].unique())
    color_map   = _provider_colors(providers)
    rooms_in_use = sorted(
        day_df["assigned_room"].dropna().unique(),
        key=lambda r: int(r.replace("ER", "")) if r.replace("ER", "").isdigit() else 99,
    )

    fig, ax = plt.subplots(figsize=(16, max(6, len(rooms_in_use) * 0.65)))
    ax.set_facecolor("#F7FBFF")
    ax.set_title(f"Room Utilisation — {day}, Week {week}  [Policy F: Admin Buffer]",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Examination Room", fontsize=10)

    for i, room in enumerate(rooms_in_use):
        for _, row in day_df[day_df["assigned_room"] == room].iterrows():
            color = color_map.get(row["provider"], "grey")
            ax.barh(i, row["duration_min"], left=row["start_min"],
                    color=color, edgecolor="white", linewidth=0.6, height=0.72)
            if row["duration_min"] >= 12:
                ax.text(row["start_min"] + 1.5, i,
                        row["provider"][-3:],
                        va="center", fontsize=6.5, color="white", fontweight="bold")

    for s, e in ADMIN_SHADING:
        ax.axvspan(s, e, alpha=0.07, color="red")

    ax.set_yticks(range(len(rooms_in_use)))
    ax.set_yticklabels(rooms_in_use, fontsize=9)

    x_ticks = list(range(480, 1021, 30))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([minutes_to_str(t) for t in x_ticks], rotation=45, fontsize=7)
    ax.set_xlim(480, 1020)

    patches = [mpatches.Patch(color=color_map[p], label=p) for p in providers]
    ax.legend(handles=patches, loc="upper right", fontsize=6.5,
              ncol=3, title="Provider", framealpha=0.8)

    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25, linestyle="--")

    plt.tight_layout()
    fname = os.path.join(OUT_DIR, f"gantt_room_{day}_W{week}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    appts_all   = load_all_appointments()
    avail_all   = load_all_provider_availability()
    dist_matrix = load_distance_matrix()

    for week in [1, 2]:
        print(f"\n{'='*50}\n  Policy F — Week {week}\n{'='*50}")

        appts = appts_all[appts_all["week"] == week].copy()
        avail = avail_all[avail_all["week"] == week].copy()

        appts_admin = apply_admin_time_buffer(appts)

        print("  Running Model 2 (sequential, conflict-free)...")
        schedules = generate_schedules_sequential(
            appts_admin, avail, dist_matrix,
            delta_frac=0.0, proximity_threshold=4.0, verbose=False,
        )

        print("  Running Model 3 (hard room constraints)...")
        master = build_master_problem(
            schedules, appts_admin,
            integer=True, verbose=False,
            hard_room_constraints=True,
        )
        print(f"  Master status: {master['status']}  cost: {master['total_cost']:.2f}")

        result_df = master_to_appointments(master, appts)

        # Clip display durations at admin-block boundaries (cosmetic only)
        clipped = appts_admin.set_index("appt_id")[["end_min", "duration_min"]]
        result_df["end_min"]      = result_df["appt_id"].map(clipped["end_min"]).fillna(result_df["end_min"])
        result_df["duration_min"] = result_df["appt_id"].map(clipped["duration_min"]).fillna(result_df["duration_min"])

        scheduled = result_df["assigned_room"].notna().sum()
        total     = len(result_df)
        print(f"  Coverage: {scheduled}/{total} = {scheduled/total*100:.1f}%")

        print("\n  Generating Gantt charts...")
        days_in_data = sorted(
            result_df["day_of_week"].dropna().unique(),
            key=lambda d: DAY_ORDER.index(d) if d in DAY_ORDER else 99,
        )
        for day in days_in_data:
            plot_provider(result_df, day, week)
            plot_room(result_df, day, week)

    print(f"\n✓ All Gantt charts saved to slides/F_gantt_charts/")


if __name__ == "__main__":
    main()
