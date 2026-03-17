"""
Main driver: Examination Room Scheduling Optimization Engine
────────────────────────────────────────────────────────────────────────────────
Runs the three-model decomposition framework and compares scheduling policies.

Policies compared:
  A. Single room per provider per day (baseline)
  B. Cluster of rooms (within proximity threshold) [column generation]
  C. Day blocking (skip providers unavailable that day)
  D. Admin time buffer (stricter overlap detection)
  E. Overbooking with no-show adjustment
  F. Robust scheduling with duration uncertainty buffer
"""

import pandas as pd
import numpy as np
from data_loader import (
    load_all_appointments, load_all_provider_availability,
    load_distance_matrix, get_provider_cluster, ROOMS
)
from model1 import run_model1_all_days
from model2 import generate_all_schedules, generate_schedules_sequential, schedules_to_dataframe
from model3 import build_master_problem, master_to_appointments
from visualization import (
    plot_gantt_provider, plot_gantt_room,
    plot_kpi_comparison, print_kpi_table
)


# ─── Policy configurations ───────────────────────────────────────────────────

POLICIES = {
    "A_single_room": {
        "proximity_threshold": 0.0,   # force single room only
        "delta_frac": 0.0,
        "description": "Single room per provider-day (baseline)",
    },
    "B_cluster": {
        "proximity_threshold": 4.0,   # rooms within 4m
        "delta_frac": 0.0,
        "description": "Cluster of rooms (proximity ≤ 4m)",
    },
    "C_admin_buffer": {
        "proximity_threshold": 4.0,
        "delta_frac": 0.10,           # 10% duration buffer
        "description": "Cluster + 10% duration uncertainty buffer",
    },
    "D_robust_noshow": {
        "proximity_threshold": 4.0,
        "delta_frac": 0.10,
        "noshow_adjust": True,
        "description": "Robust: cluster + buffer + no-show adjustment",
    },
}


# ─── Policy A: Single-room baseline ─────────────────────────────────────────

def policy_single_room(appointments, provider_avail, dist_matrix):
    """
    Assign each provider to their pre-assigned single room for the entire day.
    No optimisation; mimics current clinical practice.
    """
    rows = []
    for (provider, day, week), grp in appointments.groupby(["provider", "day_of_week", "week"]):
        # Get assigned room from availability data
        avail_row = provider_avail[
            (provider_avail["provider"] == provider) &
            (provider_avail["day"] == day) &
            (provider_avail["week"] == week)
        ]
        home_room = None
        if not avail_row.empty:
            home_room = avail_row.iloc[0]["room_am"] or avail_row.iloc[0]["room_pm"]

        for _, appt in grp.iterrows():
            rows.append({
                "appt_id": appt["appt_id"],
                "assigned_room": home_room,
                "policy": "A_single_room",
                "num_switches": 0,
                "total_travel": 0.0,
            })

    if not rows:
        return appointments.copy()

    res_df = pd.DataFrame(rows)
    return appointments.merge(
        res_df[["appt_id", "assigned_room", "policy", "num_switches", "total_travel"]],
        on="appt_id", how="left"
    )


# ─── No-show overbooking adjustment ─────────────────────────────────────────

def apply_noshow_adjustment(appointments, noshow_rates):
    """
    Returns effective appointment duration adjusted for no-show probability:
      d_hat_a = (1 - rho_a) * d_a
    Used for overbooking: shorter effective duration allows more appointments.
    """
    appts = appointments.copy()
    appts["rho"] = appts["provider"].map(noshow_rates).fillna(appts["no_show"].mean())
    appts["effective_duration"] = (1 - appts["rho"]) * appts["duration_min"]
    return appts


# ─── KPI computation ─────────────────────────────────────────────────────────

def compute_kpis(result_df, dist_matrix, policy_name):
    """
    Compute Key Performance Indicators for a scheduling solution.

    KPIs:
      - Coverage rate: % appointments assigned a room
      - Avg switches per provider-day
      - Total travel distance (m)
      - Room utilisation: % of clinic hours rooms are occupied
      - Avg appointments per room per day
    """
    total_appts = len(result_df)
    assigned = result_df["assigned_room"].notna().sum()
    coverage = assigned / total_appts * 100 if total_appts > 0 else 0

    if "num_switches" in result_df.columns:
        avg_switches = result_df.groupby(["provider", "day_of_week", "week"])["num_switches"].first().mean()
        total_travel = result_df.groupby(["provider", "day_of_week", "week"])["total_travel"].first().sum()
    else:
        avg_switches = 0
        total_travel = 0

    # Room utilisation (minutes occupied / total clinic minutes available)
    # Clinic hours: 540-1020 = 480 min/day (Mon-Thu), 480 min/day (Fri with different admin)
    clinic_min_per_day = 480
    days_in_data = result_df["date"].nunique()
    total_available_room_min = len(ROOMS) * days_in_data * clinic_min_per_day
    occupied_min = result_df[result_df["assigned_room"].notna()]["duration_min"].sum()
    utilisation = occupied_min / total_available_room_min * 100 if total_available_room_min > 0 else 0

    return {
        "Policy": policy_name,
        "Coverage (%)": round(coverage, 1),
        "Avg Switches/Provider-Day": round(float(avg_switches) if avg_switches == avg_switches else 0, 2),
        "Total Travel (m)": round(float(total_travel), 1),
        "Room Utilisation (%)": round(utilisation, 1),
        "Appointments Scheduled": int(assigned),
        "Total Appointments": int(total_appts),
    }


# ─── Main orchestration ──────────────────────────────────────────────────────

def run_full_pipeline(
    week: int = 1,
    policies_to_run: list[str] | None = None,
    verbose_model2: bool = False,
    save_results: bool = True
):
    """
    Full pipeline:
      1. Load data
      2. Run Model 1 (feasibility check)
      3. Run Model 2 (schedule generation) under different policies
      4. Run Model 3 (master selection)
      5. Compute KPIs and visualise
    """
    print("=" * 60)
    print(f"Examination Room Scheduling — Week {week}")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[1] Loading data...")
    appts_all = load_all_appointments()
    avail_all  = load_all_provider_availability()
    dist_matrix = load_distance_matrix()

    appts = appts_all[appts_all["week"] == week].copy()
    avail = avail_all[avail_all["week"] == week].copy()
    print(f"    Appointments: {len(appts)}, Providers: {appts['provider'].nunique()}, "
          f"Days: {appts['day_of_week'].nunique()}")

    # ── Model 1: Feasibility check ────────────────────────────────────────────
    print("\n[2] Running Model 1 (Feasibility Packing)...")
    appts_m1 = run_model1_all_days(appts, delta_frac=0.0, verbose=False)
    print(f"    Feasibility check complete. Columns: {list(appts_m1.columns)}")

    # ── Compute no-show rates ─────────────────────────────────────────────────
    from data_loader import compute_noshow_rate
    noshow_rates = compute_noshow_rate(appts_all)

    all_kpis = []
    policy_results = {}

    if policies_to_run is None:
        policies_to_run = list(POLICIES.keys())

    # ── Policy A: Baseline single-room ────────────────────────────────────────
    if "A_single_room" in policies_to_run:
        print("\n[3A] Policy A — Single room (baseline)...")
        res_a = policy_single_room(appts, avail, dist_matrix)
        res_a["policy"] = "A_single_room"
        policy_results["A_single_room"] = res_a
        kpi_a = compute_kpis(res_a, dist_matrix, "A: Single Room")
        all_kpis.append(kpi_a)
        print(f"     Coverage: {kpi_a['Coverage (%)']}%")

    # ── Policy B: Cluster-based optimisation (Model 2 → Model 3) ─────────────
    if "B_cluster" in policies_to_run:
        print("\n[3B] Policy B — Cluster rooms (Model 2 sequential + Model 3)...")
        cfg = POLICIES["B_cluster"]
        schedules_b = generate_schedules_sequential(
            appts, avail, dist_matrix,
            delta_frac=cfg["delta_frac"],
            proximity_threshold=cfg["proximity_threshold"],
            verbose=verbose_model2
        )
        master_b = build_master_problem(schedules_b, appts, integer=True, verbose=False)
        res_b = master_to_appointments(master_b, appts)
        res_b["policy"] = "B_cluster"
        policy_results["B_cluster"] = res_b
        kpi_b = compute_kpis(res_b, dist_matrix, "B: Cluster")
        all_kpis.append(kpi_b)
        print(f"     Master cost: {master_b['total_cost']:.2f}, "
              f"Coverage: {kpi_b['Coverage (%)']}%")

    # ── Policy C: Robust (cluster + duration buffer) ──────────────────────────
    if "C_admin_buffer" in policies_to_run:
        print("\n[3C] Policy C — Robust scheduling (10% buffer)...")
        cfg = POLICIES["C_admin_buffer"]
        schedules_c = generate_schedules_sequential(
            appts, avail, dist_matrix,
            delta_frac=cfg["delta_frac"],
            proximity_threshold=cfg["proximity_threshold"],
            verbose=verbose_model2
        )
        master_c = build_master_problem(schedules_c, appts, integer=True, verbose=False)
        res_c = master_to_appointments(master_c, appts)
        res_c["policy"] = "C_admin_buffer"
        policy_results["C_admin_buffer"] = res_c
        kpi_c = compute_kpis(res_c, dist_matrix, "C: Robust Buffer")
        all_kpis.append(kpi_c)
        print(f"     Master cost: {master_c['total_cost']:.2f}, "
              f"Coverage: {kpi_c['Coverage (%)']}%")

    # ── Policy D: No-show adjusted (overbooking) ──────────────────────────────
    if "D_robust_noshow" in policies_to_run:
        print("\n[3D] Policy D — No-show adjustment (overbooking)...")
        appts_ns = apply_noshow_adjustment(appts, noshow_rates)
        # Temporarily replace duration with effective duration for overlap computation
        appts_ns_sched = appts_ns.copy()
        appts_ns_sched["duration_min"] = appts_ns_sched["effective_duration"].round().astype(int)
        appts_ns_sched["end_min"] = appts_ns_sched["start_min"] + appts_ns_sched["duration_min"]

        cfg = POLICIES["D_robust_noshow"]
        schedules_d = generate_schedules_sequential(
            appts_ns_sched, avail, dist_matrix,
            delta_frac=cfg["delta_frac"],
            proximity_threshold=cfg["proximity_threshold"],
            verbose=verbose_model2
        )
        master_d = build_master_problem(schedules_d, appts_ns_sched, integer=True, verbose=False)
        res_d = master_to_appointments(master_d, appts)
        res_d["policy"] = "D_robust_noshow"
        policy_results["D_robust_noshow"] = res_d
        kpi_d = compute_kpis(res_d, dist_matrix, "D: No-show Robust")
        all_kpis.append(kpi_d)
        print(f"     Master cost: {master_d['total_cost']:.2f}, "
              f"Coverage: {kpi_d['Coverage (%)']}%")

    # ── KPI summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("KPI SUMMARY")
    print("=" * 60)
    kpi_df = pd.DataFrame(all_kpis)
    print(kpi_df.to_string(index=False))

    if save_results:
        kpi_df.to_csv(f"results_kpi_week{week}.csv", index=False)
        for policy_name, res_df in policy_results.items():
            res_df.to_csv(f"results_{policy_name}_week{week}.csv", index=False)
        print(f"\nResults saved to results_*_week{week}.csv")

    # ── Visualisations ────────────────────────────────────────────────────────
    if policy_results:
        best_policy = max(policy_results.items(),
                          key=lambda kv: kv[1]["assigned_room"].notna().mean())
        best_name, best_df = best_policy

        print(f"\nGenerating Gantt charts for best policy: {best_name}")
        plot_gantt_provider(best_df, week=week, policy=best_name)
        plot_gantt_room(best_df, week=week, policy=best_name)
        plot_kpi_comparison(all_kpis)

    return policy_results, kpi_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Exam Room Scheduling Optimizer")
    parser.add_argument("--week", type=int, default=1, choices=[1, 2],
                        help="Week to solve (1 or 2)")
    parser.add_argument("--policy", nargs="+",
                        choices=list(POLICIES.keys()),
                        default=list(POLICIES.keys()),
                        help="Policies to run")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose solver output")
    args = parser.parse_args()

    run_full_pipeline(
        week=args.week,
        policies_to_run=args.policy,
        verbose_model2=args.verbose,
        save_results=True
    )
