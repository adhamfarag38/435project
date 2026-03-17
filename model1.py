"""
Model 1: Feasibility Packing Model
────────────────────────────────────────────────────────────────────────────────
Assigns appointments to examination rooms while:
  - Respecting time conflicts (no two overlapping appts in same room)
  - Applying robust duration buffers (Delta_a)
  - Minimizing the number of distinct rooms used (improved objective)

Fix vs original formulation:
  Original objective  min Σ_a Σ_r x_ar  is trivially constant = |A| due to the
  assignment equality constraint.  We instead minimise Σ_r y_r, the number of
  rooms actually opened, which yields compact, meaningful schedules.
"""

import pulp
import pandas as pd
from data_loader import (
    load_all_appointments, load_distance_matrix, compute_overlap_pairs, ROOMS
)


def solve_model1(
    appointments: pd.DataFrame,
    rooms: list[str] | None = None,
    delta_frac: float = 0.1,
    verbose: bool = False
) -> dict:
    """
    Solve the Feasibility Packing Model for a given set of appointments.

    Parameters
    ----------
    appointments : DataFrame with columns [appt_id, start_min, end_min, duration_min]
    rooms        : list of room IDs to consider (default: all 16)
    delta_frac   : robust buffer as fraction of appointment duration (Delta_a = delta_frac * d_a)
    verbose      : print solver output

    Returns
    -------
    dict with keys:
        'status'      : solver status string
        'feasible'    : bool
        'assignment'  : dict {appt_id → room_id}
        'rooms_used'  : list of rooms that are occupied
        'obj_value'   : number of rooms used
    """
    if rooms is None:
        rooms = ROOMS

    appts = appointments.reset_index(drop=True)
    if appts.empty:
        return {"status": "NoAppointments", "feasible": True,
                "assignment": {}, "rooms_used": [], "obj_value": 0}

    A = list(appts["appt_id"])
    R = rooms

    # Robust buffers (minutes)
    delta = {row["appt_id"]: int(delta_frac * row["duration_min"])
             for _, row in appts.iterrows()}

    # Overlapping pairs (with robust buffers applied)
    overlaps = compute_overlap_pairs(appts, delta=delta)

    # ── Build PuLP model ────────────────────────────────────────────────────
    mdl = pulp.LpProblem("Model1_FeasibilityPacking", pulp.LpMinimize)

    # Decision variables
    x = {(a, r): pulp.LpVariable(f"x_{a}_{r}", cat="Binary") for a in A for r in R}
    # Room open indicator (improved objective)
    y = {r: pulp.LpVariable(f"y_{r}", cat="Binary") for r in R}

    # ── Objective: minimise number of rooms opened ──────────────────────────
    mdl += pulp.lpSum(y[r] for r in R)

    # ── Constraints ─────────────────────────────────────────────────────────

    # 1. Each appointment assigned to exactly one room
    for a in A:
        mdl += pulp.lpSum(x[a, r] for r in R) == 1, f"assign_{a}"

    # 2. No two overlapping appointments in the same room
    for (a, b) in overlaps:
        for r in R:
            mdl += x[a, r] + x[b, r] <= 1, f"conflict_{a}_{b}_{r}"

    # 3. Link y_r to x_ar: y_r = 1 iff any appointment uses room r
    for r in R:
        for a in A:
            mdl += x[a, r] <= y[r], f"link_{a}_{r}"

    # ── Solve ────────────────────────────────────────────────────────────────
    solver = pulp.PULP_CBC_CMD(msg=1 if verbose else 0)
    mdl.solve(solver)

    status = pulp.LpStatus[mdl.status]
    feasible = status == "Optimal"

    assignment = {}
    rooms_used = []
    if feasible:
        for a in A:
            for r in R:
                if pulp.value(x[a, r]) is not None and pulp.value(x[a, r]) > 0.5:
                    assignment[a] = r
                    break
        rooms_used = [r for r in R if pulp.value(y[r]) is not None and pulp.value(y[r]) > 0.5]

    return {
        "status": status,
        "feasible": feasible,
        "assignment": assignment,
        "rooms_used": rooms_used,
        "obj_value": pulp.value(mdl.objective) if feasible else None,
    }


def run_model1_all_days(
    appointments: pd.DataFrame,
    delta_frac: float = 0.1,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Run Model 1 for each (date, day_of_week) combination.

    Returns a DataFrame merging appointment data with assigned rooms.
    """
    results = []
    for (date, day), grp in appointments.groupby(["date", "day_of_week"]):
        sol = solve_model1(grp, delta_frac=delta_frac, verbose=verbose)
        if not sol["feasible"]:
            print(f"  [Model1] INFEASIBLE for {date} ({day})")
            continue
        for appt_id, room in sol["assignment"].items():
            results.append({"appt_id": appt_id, "assigned_room_m1": room})
        print(f"  [Model1] {date} ({day}): {sol['obj_value']:.0f} rooms used, "
              f"status={sol['status']}")

    if not results:
        return appointments.copy()

    res_df = pd.DataFrame(results)
    return appointments.merge(res_df, on="appt_id", how="left")


if __name__ == "__main__":
    appts = load_all_appointments()
    # Run on Week 1, Monday only for quick test
    mon = appts[(appts["week"] == 1) & (appts["day_of_week"] == "Monday")]
    print(f"Monday Week 1: {len(mon)} appointments")
    sol = solve_model1(mon, verbose=False)
    print(f"Status: {sol['status']}, Rooms used: {sol['obj_value']}")
    print("Sample assignments:", dict(list(sol["assignment"].items())[:5]))
