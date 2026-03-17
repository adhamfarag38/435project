"""
Model 3: Master Schedule Selection
────────────────────────────────────────────────────────────────────────────────
Selects the best combination of provider-day schedules for the entire clinic.

This is a Set Partitioning / Covering problem:
  - Each appointment must be covered exactly once  (appointment coverage)
  - No room conflict across providers at the same time  (room capacity)
  - Exactly one schedule per (provider, day) pair  (one-schedule-per-day)

In a full column-generation framework, Model 2 is the pricing sub-problem and
this is the Restricted Master Problem (RMP).  We implement both the LP relaxation
(for dual prices) and the ILP (for integer solution).
"""

import pulp
import pandas as pd
import numpy as np
from typing import Optional
from data_loader import load_all_appointments, load_all_provider_availability, load_distance_matrix
from model2 import generate_all_schedules


def build_master_problem(
    schedules: list[dict],
    appointments: pd.DataFrame,
    integer: bool = True,
    verbose: bool = False
) -> dict:
    """
    Build and solve the Master Schedule Selection model.

    Parameters
    ----------
    schedules  : list of schedule dicts from Model 2 (each has 'alpha', 'beta', 'cost', etc.)
    appointments : full appointment DataFrame (to know which appt_ids must be covered)
    integer    : if True solve ILP; if False solve LP relaxation
    verbose    : solver verbosity

    Returns
    -------
    dict with:
        'status', 'feasible', 'selected_schedules', 'total_cost',
        'dual_appointment', 'dual_room'  (duals only for LP relaxation)
    """
    # Filter to feasible schedules only
    feasible_schedules = [s for s in schedules if s.get("feasible", False)]
    if not feasible_schedules:
        return {"status": "NoFeasibleSchedules", "feasible": False,
                "selected_schedules": [], "total_cost": float("inf"),
                "dual_appointment": {}, "dual_room": {}}

    # Index schedules
    S = list(range(len(feasible_schedules)))

    # All appointment IDs that must be covered
    all_appt_ids = set(appointments["appt_id"].tolist())
    # Only cover appointments that appear in at least one schedule
    covered_appts = set()
    for sch in feasible_schedules:
        covered_appts |= set(sch["alpha"].keys())
    A_cover = all_appt_ids & covered_appts

    # Provider-day groups
    pd_groups = {}
    for s, sch in enumerate(feasible_schedules):
        key = (sch["provider"], sch["day"], sch["week"])
        pd_groups.setdefault(key, []).append(s)

    # ── Build PuLP model ─────────────────────────────────────────────────────
    mdl = pulp.LpProblem("Model3_MasterSchedule", pulp.LpMinimize)

    cat = "Binary" if integer else "Continuous"
    lb, ub = 0.0, 1.0
    lam = {s: pulp.LpVariable(f"lam_{s}", lowBound=lb, upBound=ub, cat=cat)
           for s in S}

    # ── Constraints ──────────────────────────────────────────────────────────
    # (Objective is set after room slack variables are created below)

    # 1. Appointment coverage: each appt covered exactly once
    appt_constrs = {}
    for a in A_cover:
        covering = [s for s in S if a in feasible_schedules[s]["alpha"]]
        if covering:
            c = mdl.addConstraint(
                pulp.lpSum(lam[s] for s in covering) == 1,
                name=f"appt_{a}"
            )
            appt_constrs[a] = c

    # 2. Room capacity: at most one provider per (room, time-slot, day)
    # Soft constraint with penalty slack to ensure feasibility.
    # In a full column-generation loop, new schedules would eliminate conflicts;
    # here slacks catch residual cross-provider conflicts.
    all_rt = set()
    for sch in feasible_schedules:
        all_rt |= set(sch["beta"].keys())

    ROOM_PENALTY = 1e4   # large penalty to discourage room conflicts
    room_slack = {}
    room_constrs = {}
    for (r, t) in all_rt:
        using = [s for s in S if (r, t) in feasible_schedules[s]["beta"]]
        if len(using) > 1:
            sl = pulp.LpVariable(f"slack_r_{r}_{t}", lowBound=0)
            room_slack[(r, t)] = sl
            c = mdl.addConstraint(
                pulp.lpSum(lam[s] for s in using) - sl <= 1,
                name=f"room_{r}_{t}"
            )
            room_constrs[(r, t)] = c

    # Add slack penalty to objective
    mdl += pulp.lpSum(feasible_schedules[s]["cost"] * lam[s] for s in S) + \
           pulp.lpSum(ROOM_PENALTY * sl for sl in room_slack.values())

    # 3. One schedule per (provider, day, week)
    for (prov, day, week), s_list in pd_groups.items():
        if len(s_list) > 0:
            tag = f"{prov}_{day}_W{week}".replace(" ", "_")
            mdl += pulp.lpSum(lam[s] for s in s_list) == 1, f"oneschedule_{tag}"

    # ── Solve ─────────────────────────────────────────────────────────────────
    solver = pulp.PULP_CBC_CMD(msg=1 if verbose else 0, timeLimit=120)
    mdl.solve(solver)

    status = pulp.LpStatus[mdl.status]
    feasible_sol = mdl.status == 1

    selected = []
    total_cost = float("inf")
    dual_appt = {}
    dual_room = {}

    if feasible_sol:
        total_cost = pulp.value(mdl.objective)
        selected = [s for s in S if pulp.value(lam[s]) is not None
                    and pulp.value(lam[s]) > 0.5]

        if not integer:
            # Extract dual variables (shadow prices) for column generation
            for a, c in appt_constrs.items():
                dual_appt[a] = c.pi if hasattr(c, "pi") else 0.0
            for rt, c in room_constrs.items():
                dual_room[rt] = c.pi if hasattr(c, "pi") else 0.0

    return {
        "status": status,
        "feasible": feasible_sol,
        "selected_schedules": [feasible_schedules[s] for s in selected],
        "selected_indices": selected,
        "total_cost": total_cost,
        "dual_appointment": dual_appt,
        "dual_room": dual_room,
        "num_schedules_considered": len(feasible_schedules),
    }


def master_to_appointments(
    master_result: dict,
    appointments: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge selected schedule assignments back into the appointments DataFrame.
    Returns appointment DataFrame with 'assigned_room', 'schedule_cost' columns.
    """
    rows = []
    for sch in master_result.get("selected_schedules", []):
        for appt_id, room in sch["assignment"].items():
            rows.append({
                "appt_id": appt_id,
                "assigned_room": room,
                "provider": sch["provider"],
                "day_model": sch["day"],
                "schedule_cost": sch["cost"],
                "num_switches": sch["num_switches"],
                "total_travel": sch["total_travel"],
            })

    result_cols = ["appt_id", "assigned_room", "schedule_cost", "num_switches", "total_travel"]
    if not rows:
        out = appointments.copy()
        for col in result_cols[1:]:
            out[col] = None
        return out

    res_df = pd.DataFrame(rows).drop_duplicates("appt_id")
    return appointments.merge(res_df[result_cols], on="appt_id", how="left")


def compute_reduced_cost(
    schedule: dict,
    dual_appt: dict[int, float],
    dual_room: dict[tuple, float]
) -> float:
    """
    Compute reduced cost of a schedule for column generation pricing.
    A negative reduced cost means the column should enter the master problem.
    """
    rc = schedule["cost"]
    rc -= sum(dual_appt.get(a, 0.0) for a in schedule["alpha"])
    rc -= sum(dual_room.get(rt, 0.0) for rt in schedule["beta"])
    return rc


if __name__ == "__main__":
    from data_loader import load_all_appointments, load_all_provider_availability

    print("Loading data...")
    appts = load_all_appointments()
    avail = load_all_provider_availability()
    dist  = load_distance_matrix()

    # Limit to Week 1 for quick test
    appts_w1 = appts[appts["week"] == 1]

    print("Generating provider-day schedules (Model 2)...")
    schedules = generate_all_schedules(appts_w1, avail, dist, verbose=False)

    print(f"\nGenerated {len(schedules)} schedules, "
          f"{sum(s['feasible'] for s in schedules)} feasible.")

    print("\nSolving Master Problem (Model 3)...")
    result = build_master_problem(schedules, appts_w1, integer=True, verbose=False)
    print(f"Status: {result['status']}, Total cost: {result['total_cost']:.2f}")
    print(f"Selected {len(result['selected_schedules'])} schedules.")
