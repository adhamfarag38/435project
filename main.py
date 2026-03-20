"""
Main driver: Examination Room Scheduling Optimization Engine
────────────────────────────────────────────────────────────────────────────────
Runs the three-model decomposition framework and compares scheduling policies.

Policies compared (matching course validation framework):
  A. Single room per provider per day (baseline)         — policy (a)
  B. Cluster of rooms (within proximity threshold)        — policy (b)
  C. Robust duration buffer (10% uncertainty)             — policy (f)
  D. Overbooking with no-show adjustment                  — policy (e)
  E. Day blocking (skip unavailable provider-days)        — policy (c)
  F. Admin time buffer (use admin blocks for overruns)    — policy (d)
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
        "description": "Single room per provider-day (baseline)",    # policy (a)
    },
    "B_cluster": {
        "proximity_threshold": 4.0,   # rooms within 4m
        "delta_frac": 0.0,
        "description": "Cluster of rooms (proximity ≤ 4m)",          # policy (b)
    },
    "C_robust_buffer": {
        "proximity_threshold": 4.0,
        "delta_frac": 0.10,           # 10% duration uncertainty buffer
        "description": "Cluster + 10% duration uncertainty buffer",  # policy (f)
    },
    "D_robust_noshow": {
        "proximity_threshold": 4.0,
        "delta_frac": 0.10,
        "noshow_adjust": True,
        "description": "Robust: cluster + buffer + no-show adjustment",  # policy (e)
    },
    "E_day_blocking": {
        "proximity_threshold": 4.0,
        "delta_frac": 0.0,
        "description": "Block unavailable provider-days (respect day-off schedule)",  # policy (c)
    },
    "F_admin_buffer": {
        "proximity_threshold": 4.0,
        "delta_frac": 0.0,
        "description": "Admin time used as buffer for appointment overruns",  # policy (d)
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


# ─── Policy E: Day-blocking filter ───────────────────────────────────────────

def apply_day_blocking(appointments: pd.DataFrame,
                        provider_avail: pd.DataFrame) -> pd.DataFrame:
    """
    Policy (c): Block certain days for certain healthcare providers.

    Removes appointments for providers who are marked unavailable (no room
    assignment) on that day in the provider availability schedule.
    Blocked appointments will appear as unscheduled (null room) in KPIs,
    correctly reducing coverage for days when providers are not working.

    Returns filtered appointments (subset of input).
    """
    avail_lookup = {
        (row["provider"], row["day"], row["week"]): row["available"]
        for _, row in provider_avail.iterrows()
    }
    mask = appointments.apply(
        lambda r: avail_lookup.get((r["provider"], r["day_of_week"], r["week"]), True),
        axis=1
    )
    blocked = (~mask).sum()
    print(f"     [Day blocking] Removed {blocked} appointments "
          f"({blocked / len(appointments) * 100:.1f}%) for unavailable providers.")
    return appointments[mask].copy()


# ─── Policy F: Admin time buffer ──────────────────────────────────────────────

def apply_admin_time_buffer(appointments: pd.DataFrame) -> pd.DataFrame:
    """
    Policy (d): Use certain admin time for extenuating circumstances.

    Admin blocks (morning huddle, noon prep, lunch, afternoon close) are
    treated as flexible buffer time that absorbs appointment overruns.
    Concretely: if an appointment's end_min extends past the START of the
    next admin block, we clip it to that admin block start.

    Effect: two appointments that technically overlap by a few minutes
    (one running into admin time) are no longer flagged as conflicting.
    This allows tighter scheduling and increases room utilisation.

    Returns a copy of appointments with adjusted end_min / duration_min.
    """
    from data_loader import ADMIN_BLOCKS

    appts = appointments.copy()
    clipped = 0
    for idx, row in appts.iterrows():
        key = "friday" if row["day_of_week"] == "Friday" else "weekday"
        for _, (blk_start, blk_end) in ADMIN_BLOCKS[key].items():
            # Appointment straddles admin block start → clip end at block boundary
            if row["start_min"] < blk_start < row["end_min"]:
                appts.at[idx, "end_min"]      = blk_start
                appts.at[idx, "duration_min"] = blk_start - int(row["start_min"])
                clipped += 1
                break   # Only apply the first applicable block
    print(f"     [Admin buffer] Clipped {clipped} appointments "
          f"({clipped / len(appts) * 100:.1f}%) at admin block boundaries.")
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

    # ── Policy C: Robust duration buffer (policy f) ───────────────────────────
    if "C_robust_buffer" in policies_to_run:
        print("\n[3C] Policy C — Robust scheduling (10% duration buffer, policy f)...")
        cfg = POLICIES["C_robust_buffer"]
        schedules_c = generate_schedules_sequential(
            appts, avail, dist_matrix,
            delta_frac=cfg["delta_frac"],
            proximity_threshold=cfg["proximity_threshold"],
            verbose=verbose_model2
        )
        master_c = build_master_problem(schedules_c, appts, integer=True, verbose=False)
        res_c = master_to_appointments(master_c, appts)
        res_c["policy"] = "C_robust_buffer"
        policy_results["C_robust_buffer"] = res_c
        kpi_c = compute_kpis(res_c, dist_matrix, "C: Robust Buffer (f)")
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

    # ── Policy E: Day blocking (policy c) ────────────────────────────────────
    if "E_day_blocking" in policies_to_run:
        print("\n[3E] Policy E — Day blocking (policy c): skip unavailable provider-days...")
        appts_blocked = apply_day_blocking(appts, avail)
        cfg = POLICIES["E_day_blocking"]
        schedules_e = generate_schedules_sequential(
            appts_blocked, avail, dist_matrix,
            delta_frac=cfg["delta_frac"],
            proximity_threshold=cfg["proximity_threshold"],
            verbose=verbose_model2
        )
        master_e = build_master_problem(schedules_e, appts_blocked, integer=True, verbose=False)
        # Merge back against ORIGINAL appts so blocked appointments appear as unscheduled
        res_e = master_to_appointments(master_e, appts)
        res_e["policy"] = "E_day_blocking"
        policy_results["E_day_blocking"] = res_e
        kpi_e = compute_kpis(res_e, dist_matrix, "E: Day Blocking (c)")
        all_kpis.append(kpi_e)
        print(f"     Master cost: {master_e['total_cost']:.2f}, "
              f"Coverage: {kpi_e['Coverage (%)']}%")

    # ── Policy F: Admin time buffer (policy d) ────────────────────────────────
    if "F_admin_buffer" in policies_to_run:
        print("\n[3F] Policy F — Admin time buffer (policy d): absorb overruns into admin blocks...")
        appts_admin = apply_admin_time_buffer(appts)
        cfg = POLICIES["F_admin_buffer"]
        schedules_f = generate_schedules_sequential(
            appts_admin, avail, dist_matrix,
            delta_frac=cfg["delta_frac"],
            proximity_threshold=cfg["proximity_threshold"],
            verbose=verbose_model2
        )
        master_f = build_master_problem(schedules_f, appts_admin, integer=True, verbose=False)
        # Merge back against ORIGINAL appts to report actual appointment coverage
        res_f = master_to_appointments(master_f, appts)
        res_f["policy"] = "F_admin_buffer"
        policy_results["F_admin_buffer"] = res_f
        kpi_f = compute_kpis(res_f, dist_matrix, "F: Admin Buffer (d)")
        all_kpis.append(kpi_f)
        print(f"     Master cost: {master_f['total_cost']:.2f}, "
              f"Coverage: {kpi_f['Coverage (%)']}%")

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


# ─── Advanced Pipeline: CG → LR → TS ────────────────────────────────────────

def run_advanced_pipeline(
    week: int = 1,
    delta_frac: float = 0.0,
    proximity_threshold: float = 4.0,
    cg_max_iter: int = 30,
    lr_max_iter: int = 150,
    ts_max_iter: int = 300,
    ts_k_max: int = 20,
    verbose: bool = False,
    save_results: bool = True,
):
    """
    Advanced optimisation pipeline combining three course methods:

      1. COLUMN GENERATION (Dantzig-Wolfe decomposition)
         ─ Iteratively adds schedules with negative reduced cost to the master pool.
         ─ LP relaxation → dual prices → pricing subproblem (Model 2) → add column.
         ─ Produces: LP lower bound (z_LP), ILP upper bound (z_ILP).

      2. LAGRANGIAN RELAXATION  (Subgradient method)
         ─ Relaxes appointment-coverage constraints with multipliers u_a ≥ 0.
         ─ Updates u via subgradient ascent with Polyak step size.
         ─ Produces: Lagrangian lower bound z_LR ≤ z*.

      3. TABU SEARCH  (Post-optimisation metaheuristic)
         ─ Starts from the CG ILP solution and explores single-swap neighbourhood.
         ─ Maintains a FIFO tabu list to prevent cycling; aspiration criterion
           overrides tabu when a global improvement is found.
         ─ Produces: improved upper bound z_TS ≤ z_ILP.

      BOUNDS REPORTED
      ───────────────
        z_LR  ≤  z_LP  ≤  z*  ≤  z_TS  ≤  z_ILP

        Duality gap = (z_TS − z_LR) / z_TS × 100 %

      B&B / B&C NOTE
      ──────────────
      The ILP solves in steps 1 and the LR subproblems all use CBC, which
      internally applies Branch-and-Bound (B&B) with Gomory cutting planes
      (Branch-and-Cut).  The LP relaxation bound computed in step 1 IS the
      B&B root-node LP relaxation bound — it lower-bounds every node in the
      B&B tree.  The gap between this LP bound and the ILP solution measures
      how much B&C tightening was required.
    """
    from column_generation import run_column_generation
    from lagrangian        import run_lagrangian_relaxation
    from tabu_search       import run_tabu_search
    from model3            import master_to_appointments

    print("\n" + "█" * 62)
    print(f"  ADVANCED PIPELINE — Week {week}")
    print("█" * 62)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\n[0] Loading data...")
    appts_all  = load_all_appointments()
    avail_all  = load_all_provider_availability()
    dist_matrix = load_distance_matrix()

    appts = appts_all[appts_all["week"] == week].copy()
    avail = avail_all[avail_all["week"] == week].copy()
    print(f"    Appointments: {len(appts)}, "
          f"Providers: {appts['provider'].nunique()}, "
          f"Days: {appts['day_of_week'].nunique()}")

    # ── Step 1: Column Generation ──────────────────────────────────────────────
    cg = run_column_generation(
        appts, avail, dist_matrix,
        delta_frac=delta_frac,
        proximity_threshold=proximity_threshold,
        max_iterations=cg_max_iter,
        verbose=verbose,
    )
    z_lp  = cg["lp_bound"]
    z_ilp = cg["ilp_bound"]

    # ── Step 2: Lagrangian Relaxation ──────────────────────────────────────────
    lr = run_lagrangian_relaxation(
        cg["schedules"], appts,
        z_ub=z_ilp,
        max_iterations=lr_max_iter,
        verbose=verbose,
    )
    z_lr = lr["best_lower_bound"]

    # ── Step 3: Tabu Search ────────────────────────────────────────────────────
    selected_scheds = cg["master_result"].get("selected_schedules", [])
    if not selected_scheds:
        print("[Advanced] No feasible ILP solution to improve — skipping TS.")
        ts = None
        z_ts = z_ilp
    else:
        ts = run_tabu_search(
            selected_scheds, appts, avail, dist_matrix,
            proximity_threshold=proximity_threshold,
            k_max=ts_k_max,
            max_iterations=ts_max_iter,
            verbose=verbose,
        )
        z_ts = ts["best_cost"]

    # ── Bound Summary ─────────────────────────────────────────────────────────
    gap_pct = (z_ts - z_lr) / abs(z_ts) * 100 if z_ts not in (0, float("inf")) else None

    print("\n" + "█" * 62)
    print("  FINAL BOUNDS SUMMARY")
    print("█" * 62)
    print(f"\n  Method                        Bound")
    print(f"  {'─'*50}")
    if z_lr is not None:
        print(f"  Lagrangian lower bound z_LR : {z_lr:>12.4f}  (≤ optimal)")
    if z_lp is not None:
        print(f"  LP relaxation bound    z_LP : {z_lp:>12.4f}  (≤ optimal, B&B root)")
    print(f"  CG ILP solution        z_ILP: {z_ilp:>12.4f}  (feasible solution)")
    print(f"  Tabu Search solution   z_TS : {z_ts:>12.4f}  (improved feasible)")
    if gap_pct is not None:
        print(f"\n  Duality gap (z_TS − z_LR)/z_TS : {gap_pct:.2f}%")
        if gap_pct < 5:
            print("  → Gap < 5%: solution is near-optimal.")
        elif gap_pct < 15:
            print("  → Gap < 15%: solution is good quality.")
        else:
            print("  → Gap > 15%: more CG/LR iterations may help.")
    print(f"\n  B&B note: the ILP solves used CBC (Branch-and-Cut internally).")
    print(f"  LP bound {z_lp:.4f} is the B&B root-node relaxation bound.")
    print(f"  B&C tightening: {abs(z_ilp - z_lp):.4f} above LP bound.")
    print("█" * 62 + "\n")

    # ── Save Results ───────────────────────────────────────────────────────────
    if save_results:
        if ts is not None:
            ts["best_df"].to_csv(f"results_advanced_week{week}.csv", index=False)
        lr["history"].to_csv(f"lr_history_week{week}.csv", index=False)
        if ts is not None:
            ts["history"].to_csv(f"ts_history_week{week}.csv", index=False)

        bounds_df = pd.DataFrame([{
            "z_LR (Lagrangian LB)":  z_lr,
            "z_LP (CG LP bound)":    z_lp,
            "z_ILP (CG ILP)":        z_ilp,
            "z_TS (Tabu Search)":    z_ts,
            "Duality Gap (%)":       gap_pct,
        }])
        bounds_df.to_csv(f"bounds_summary_week{week}.csv", index=False)
        print(f"Results saved: results_advanced_week{week}.csv, "
              f"lr_history_week{week}.csv, ts_history_week{week}.csv, "
              f"bounds_summary_week{week}.csv")

    return {
        "z_lp":         z_lp,
        "z_ilp":        z_ilp,
        "z_lr":         z_lr,
        "z_ts":         z_ts,
        "gap_pct":      gap_pct,
        "cg_result":    cg,
        "lr_result":    lr,
        "ts_result":    ts,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Exam Room Scheduling Optimizer")
    parser.add_argument("--week",     type=int,  default=1, choices=[1, 2])
    parser.add_argument("--mode",     type=str,  default="advanced",
                        choices=["basic", "advanced"],
                        help="basic = policy comparison; advanced = CG+LR+TS")
    parser.add_argument("--policy",   nargs="+",
                        choices=list(POLICIES.keys()) + ["C_admin_buffer"],  # legacy alias
                        default=list(POLICIES.keys()))
    parser.add_argument("--verbose",  action="store_true")
    args = parser.parse_args()

    if args.mode == "advanced":
        run_advanced_pipeline(
            week=args.week,
            verbose=args.verbose,
            save_results=True,
        )
    else:
        run_full_pipeline(
            week=args.week,
            policies_to_run=args.policy,
            verbose_model2=args.verbose,
            save_results=True,
        )
