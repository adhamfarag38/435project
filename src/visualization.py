"""
Visualization module: Gantt charts and KPI comparisons.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import numpy as np
from data_loader import ROOMS, minutes_to_str


# ─── Colour palette ──────────────────────────────────────────────────────────

def _room_color_map(rooms):
    cmap = cm.get_cmap("tab20", len(rooms))
    return {r: cmap(i) for i, r in enumerate(rooms)}


def _provider_color_map(providers):
    cmap = cm.get_cmap("tab20b", len(providers))
    return {p: cmap(i) for i, p in enumerate(providers)}


# ─── Gantt: per provider ─────────────────────────────────────────────────────

def plot_gantt_provider(
    result_df: pd.DataFrame,
    week: int = 1,
    policy: str = "",
    day_filter: str | None = None,
    save_path: str | None = None,
):
    """
    Gantt chart with providers on Y-axis and time on X-axis.
    Each bar represents one appointment, coloured by assigned room.
    """
    df = result_df.dropna(subset=["assigned_room"]).copy()
    if day_filter:
        df = df[df["day_of_week"] == day_filter]
    if df.empty:
        print("No data to plot for Gantt (provider view).")
        return

    days = sorted(df["day_of_week"].unique(),
                  key=lambda d: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"].index(d)
                  if d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"] else 99)

    rooms_used = sorted(df["assigned_room"].dropna().unique())
    color_map = _room_color_map(rooms_used)

    for day in days:
        day_df = df[df["day_of_week"] == day]
        providers = sorted(day_df["provider"].unique())

        fig, ax = plt.subplots(figsize=(16, max(6, len(providers) * 0.5)))
        ax.set_title(f"Provider Schedule — {day}, Week {week}  [{policy}]", fontsize=13)
        ax.set_xlabel("Time")
        ax.set_ylabel("Provider")

        y_ticks = []
        y_labels = []

        for i, prov in enumerate(providers):
            p_df = day_df[day_df["provider"] == prov]
            y_ticks.append(i)
            y_labels.append(prov)
            for _, row in p_df.iterrows():
                room = row["assigned_room"]
                color = color_map.get(room, "grey")
                ax.barh(i, row["duration_min"], left=row["start_min"],
                        color=color, edgecolor="black", linewidth=0.5, height=0.7)
                if row["duration_min"] >= 15:
                    ax.text(row["start_min"] + 1, i,
                            str(room).replace("ER", "R"), va="center",
                            fontsize=6, color="white", fontweight="bold")

        # Admin / lunch shading
        for blk_start, blk_end in [(540, 570), (690, 720), (720, 780), (990, 1020)]:
            ax.axvspan(blk_start, blk_end, alpha=0.08, color="red", label="_nolegend_")

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=8)

        # X-axis: every 30 min
        x_ticks = list(range(480, 1021, 30))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([minutes_to_str(t) for t in x_ticks], rotation=45, fontsize=7)
        ax.set_xlim(480, 1020)

        # Legend
        patches = [mpatches.Patch(color=color_map[r], label=r) for r in rooms_used]
        ax.legend(handles=patches, loc="upper right", fontsize=7,
                  ncol=4, title="Room", framealpha=0.7)

        plt.tight_layout()
        fname = save_path or f"gantt_provider_{day}_W{week}_{policy}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fname}")


# ─── Gantt: per room ─────────────────────────────────────────────────────────

def plot_gantt_room(
    result_df: pd.DataFrame,
    week: int = 1,
    policy: str = "",
    day_filter: str | None = None,
    save_path: str | None = None,
):
    """
    Gantt chart with examination rooms on Y-axis.
    Each bar represents one appointment, coloured by provider.
    """
    df = result_df.dropna(subset=["assigned_room"]).copy()
    if day_filter:
        df = df[df["day_of_week"] == day_filter]
    if df.empty:
        print("No data to plot for Gantt (room view).")
        return

    days = sorted(df["day_of_week"].unique(),
                  key=lambda d: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"].index(d)
                  if d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"] else 99)

    providers = sorted(df["provider"].unique())
    color_map = _provider_color_map(providers)

    for day in days:
        day_df = df[df["day_of_week"] == day]
        rooms_in_use = sorted(
            day_df["assigned_room"].dropna().unique(),
            key=lambda r: int(r.replace("ER", "")) if r.replace("ER", "").isdigit() else 99
        )

        fig, ax = plt.subplots(figsize=(16, max(6, len(rooms_in_use) * 0.6)))
        ax.set_title(f"Room Utilisation — {day}, Week {week}  [{policy}]", fontsize=13)
        ax.set_xlabel("Time")
        ax.set_ylabel("Examination Room")

        y_ticks = []
        y_labels = []

        for i, room in enumerate(rooms_in_use):
            r_df = day_df[day_df["assigned_room"] == room]
            y_ticks.append(i)
            y_labels.append(room)
            for _, row in r_df.iterrows():
                color = color_map.get(row["provider"], "grey")
                ax.barh(i, row["duration_min"], left=row["start_min"],
                        color=color, edgecolor="black", linewidth=0.5, height=0.7)
                if row["duration_min"] >= 15:
                    ax.text(row["start_min"] + 1, i,
                            row["provider"][-3:], va="center",
                            fontsize=6, color="white", fontweight="bold")

        # Admin shading
        for blk_start, blk_end in [(540, 570), (690, 720), (720, 780), (990, 1020)]:
            ax.axvspan(blk_start, blk_end, alpha=0.08, color="red")

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=9)

        x_ticks = list(range(480, 1021, 30))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([minutes_to_str(t) for t in x_ticks], rotation=45, fontsize=7)
        ax.set_xlim(480, 1020)

        patches = [mpatches.Patch(color=color_map[p], label=p) for p in providers
                   if p in day_df["provider"].values]
        ax.legend(handles=patches, loc="upper right", fontsize=7,
                  ncol=4, title="Provider", framealpha=0.7)

        plt.tight_layout()
        fname = save_path or f"gantt_room_{day}_W{week}_{policy}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fname}")


# ─── Full-week overview Gantt ────────────────────────────────────────────────

def plot_gantt_week_overview(
    result_df: pd.DataFrame,
    week: int = 1,
    policy: str = "",
    save_path: str | None = None,
):
    """
    Single figure with 5 side-by-side panels (Mon–Fri).
    Providers on the Y-axis, time on the X-axis, bars coloured by room.
    Panels share the same Y-axis (provider list) and X-axis (time range).
    Designed to fit on one presentation slide.
    """
    DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    ADMIN_BLOCKS = [(540, 570), (690, 720), (720, 780), (990, 1020)]

    df = result_df.dropna(subset=["assigned_room"]).copy()

    # Consistent provider order (by total appointments descending)
    provider_order = (
        df.groupby("provider").size()
        .sort_values(ascending=False)
        .index.tolist()
    )
    prov_y = {p: i for i, p in enumerate(provider_order)}

    rooms_used = sorted(df["assigned_room"].dropna().unique())
    color_map  = _room_color_map(rooms_used)

    days_present = [d for d in DAY_ORDER if d in df["day_of_week"].values]
    n_days = len(days_present)

    fig, axes = plt.subplots(
        1, n_days,
        figsize=(5 * n_days, max(8, len(provider_order) * 0.45)),
        sharey=True,
    )
    if n_days == 1:
        axes = [axes]

    fig.suptitle(
        f"Policy F — Full Week Schedule  |  Week {week}",
        fontsize=14, fontweight="bold", y=1.01,
    )

    for ax, day in zip(axes, days_present):
        day_df = df[df["day_of_week"] == day]

        # Admin block shading
        for blk_start, blk_end in ADMIN_BLOCKS:
            ax.axvspan(blk_start, blk_end, alpha=0.10, color="red", zorder=0)

        # Appointment bars
        for _, row in day_df.iterrows():
            y     = prov_y.get(row["provider"], 0)
            color = color_map.get(row["assigned_room"], "grey")
            ax.barh(y, row["duration_min"], left=row["start_min"],
                    color=color, edgecolor="white", linewidth=0.4, height=0.7)
            if row["duration_min"] >= 20:
                ax.text(
                    row["start_min"] + 1, y,
                    str(row["assigned_room"]).replace("ER", "R"),
                    va="center", fontsize=5.5, color="white", fontweight="bold",
                )

        # Providers scheduled this day (highlight active rows)
        active = set(day_df["provider"])
        for p in provider_order:
            if p not in active:
                ax.barh(prov_y[p], 540, left=480,
                        color="#f0f0f0", edgecolor="none", height=0.7, zorder=0)

        ax.set_title(f"{day}\n({len(day_df)} appts, "
                     f"{day_df['provider'].nunique()} providers)",
                     fontsize=9, pad=4)
        ax.set_xlim(480, 1020)
        x_ticks = list(range(480, 1021, 60))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([minutes_to_str(t) for t in x_ticks],
                           rotation=45, fontsize=7)
        ax.tick_params(axis="y", which="both", length=0)
        ax.grid(axis="x", linestyle="--", alpha=0.3, zorder=1)

    # Y-axis labels on leftmost panel only
    axes[0].set_yticks(list(prov_y.values()))
    axes[0].set_yticklabels(list(prov_y.keys()), fontsize=7)
    axes[0].set_ylabel("Provider", fontsize=9)

    # Shared room legend (bottom centre)
    patches = [mpatches.Patch(color=color_map[r], label=r) for r in rooms_used]
    fig.legend(
        handles=patches, loc="lower center",
        ncol=min(len(rooms_used), 8),
        fontsize=7, title="Room", title_fontsize=8,
        bbox_to_anchor=(0.5, -0.06), framealpha=0.9,
    )

    plt.tight_layout()
    fname = save_path or f"gantt_week_overview_W{week}_{policy}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ─── KPI comparison bar chart ─────────────────────────────────────────────────

def plot_kpi_comparison(kpis: list[dict], save_path: str | None = None):
    """
    Bar charts comparing KPIs across policies.
    """
    df = pd.DataFrame(kpis)
    metrics = [
        ("Coverage (%)", "Coverage (%)"),
        ("Avg Switches/Provider-Day", "Avg Room Switches per Provider-Day"),
        ("Total Travel (m)", "Total Provider Travel (m)"),
        ("Room Utilisation (%)", "Room Utilisation (%)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Policy Comparison — Key Performance Indicators", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    colors = plt.cm.Set2(np.linspace(0, 1, len(df)))

    for ax, (col, title) in zip(axes, metrics):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        bars = ax.bar(df["Policy"], df[col], color=colors, edgecolor="black", linewidth=0.8)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(col)
        ax.set_xticks(range(len(df["Policy"])))
        ax.set_xticklabels(df["Policy"], rotation=20, ha="right", fontsize=9)
        for bar, val in zip(bars, df[col]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * (df[col].max() or 1),
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fname = save_path or "kpi_comparison.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ─── KPI table print ──────────────────────────────────────────────────────────

def print_kpi_table(kpis: list[dict]):
    df = pd.DataFrame(kpis)
    print("\n" + "=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)


if __name__ == "__main__":
    # Quick smoke-test with dummy data
    from data_loader import load_all_appointments, load_all_provider_availability
    appts = load_all_appointments()
    avail = load_all_provider_availability()

    # Fake assignment: give every appointment ER1 so we can test the chart
    test_df = appts[appts["week"] == 1].copy()
    test_df["assigned_room"] = "ER1"
    test_df["num_switches"] = 0
    test_df["total_travel"] = 0.0

    plot_gantt_provider(test_df, week=1, policy="test", day_filter="Monday")
    print("Visualization test complete.")
