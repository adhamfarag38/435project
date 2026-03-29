"""
Generate Policy F schedules and Gantt charts for both weeks.

Policy F: Admin time used as overflow buffer (proximity cluster ≤ 4m).
Runs Model 2 (sequential schedule generation) → Model 3 (master selection),
then produces:
  - gantt_room_<day>_W<week>_F_admin_buffer.png   (rooms on Y-axis)
  - gantt_provider_<day>_W<week>_F_admin_buffer.png (providers on Y-axis)
  - policy_f_schedule_week<week>.csv               (full assignment table)
"""

import pandas as pd
from data_loader import load_all_appointments, load_all_provider_availability, load_distance_matrix
from model2 import generate_schedules_sequential
from model3 import build_master_problem, master_to_appointments
from visualization import plot_gantt_provider, plot_gantt_room, plot_kpi_comparison
from main import apply_admin_time_buffer, compute_kpis

PROXIMITY_THRESHOLD = 4.0   # metres — rooms within this distance form the cluster
DELTA_FRAC          = 0.0   # no duration buffer (admin blocks absorb overruns instead)


def run_policy_f(week: int, verbose: bool = False):
    print(f"\n{'=' * 60}")
    print(f"  Policy F — Admin Buffer Schedule  |  Week {week}")
    print(f"{'=' * 60}")

    # ── Load data ──────────────────────────────────────────────────────────────
    appts_all   = load_all_appointments()
    avail_all   = load_all_provider_availability()
    dist_matrix = load_distance_matrix()

    appts = appts_all[appts_all["week"] == week].copy()
    avail = avail_all[avail_all["week"] == week].copy()
    print(f"  Appointments: {len(appts)} | "
          f"Providers: {appts['provider'].nunique()} | "
          f"Days: {appts['day_of_week'].nunique()}")

    # ── Apply admin-time buffer (clip end_min at admin block boundaries) ───────
    print("\n  Applying admin-time buffer...")
    appts_admin = apply_admin_time_buffer(appts)

    # ── Model 2: generate one candidate schedule per (provider, day) ───────────
    print("\n  Running Model 2 (schedule generation)...")
    schedules = generate_schedules_sequential(
        appts_admin, avail, dist_matrix,
        delta_frac=DELTA_FRAC,
        proximity_threshold=PROXIMITY_THRESHOLD,
        verbose=verbose,
    )
    n_feasible = sum(1 for s in schedules if s.get("feasible", False))
    print(f"  {len(schedules)} schedules generated, {n_feasible} feasible.")

    # ── Model 3: select optimal combination ───────────────────────────────────
    print("\n  Running Model 3 (master schedule selection)...")
    master = build_master_problem(schedules, appts_admin, integer=True, verbose=False)
    print(f"  Master status : {master['status']}")
    print(f"  Master cost   : {master['total_cost']:.2f}")

    # Merge back against ORIGINAL appointments (report real appointment coverage).
    # Use clipped durations for display so bars don't visually extend into admin blocks.
    result_df = master_to_appointments(master, appts)
    clipped_lookup = appts_admin.set_index("appt_id")[["end_min", "duration_min"]]
    result_df["end_min"]      = result_df["appt_id"].map(clipped_lookup["end_min"]).fillna(result_df["end_min"])
    result_df["duration_min"] = result_df["appt_id"].map(clipped_lookup["duration_min"]).fillna(result_df["duration_min"])
    result_df["policy"] = "F_admin_buffer"

    # ── KPIs ──────────────────────────────────────────────────────────────────
    kpi = compute_kpis(result_df, dist_matrix, "F: Admin Buffer")
    print(f"\n  Coverage              : {kpi['Coverage (%)']}%  "
          f"({kpi['Appointments Scheduled']}/{kpi['Total Appointments']})")
    print(f"  Avg switches/prov-day : {kpi['Avg Switches/Provider-Day']}")
    print(f"  Total travel          : {kpi['Total Travel (m)']} m")
    print(f"  Room utilisation      : {kpi['Room Utilisation (%)']}%")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = f"policy_f_schedule_week{week}.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"\n  Schedule saved → {csv_path}")

    # ── Gantt: room view (one chart per day) ──────────────────────────────────
    print("\n  Generating room-view Gantt charts...")
    plot_gantt_room(result_df, week=week, policy="F_admin_buffer")

    # ── Gantt: provider view (one chart per day) ──────────────────────────────
    print("\n  Generating provider-view Gantt charts...")
    plot_gantt_provider(result_df, week=week, policy="F_admin_buffer")

    return result_df, kpi


def main():
    all_kpis = []
    for week in [1, 2]:
        _, kpi = run_policy_f(week=week, verbose=False)
        kpi["Week"] = week
        all_kpis.append(kpi)

    print("\n\nSummary across both weeks:")
    print(pd.DataFrame(all_kpis)[
        ["Week", "Coverage (%)", "Appointments Scheduled",
         "Avg Switches/Provider-Day", "Total Travel (m)", "Room Utilisation (%)"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
