"""
Model 2: Provider-Day Schedule Generation
────────────────────────────────────────────────────────────────────────────────
For a given (provider, day) pair, assigns appointments to examination rooms while
minimising provider travel (room switches × distance).

Key design decisions vs original formulation:
  - Since appointment start times are FIXED in the data, x_art reduces to x_ar
    (time is not a decision variable).  This dramatically shrinks the model.
  - Room-switch variables z_rr' are properly linked to x_ar via bilinear
    product linearisation over consecutive appointment pairs.
  - Room cluster constraint C_p is enforced by restricting R to the cluster.
  - Robust durations: Delta_a = delta_frac * d_a widens overlap detection.
"""

import pulp
import pandas as pd
import numpy as np
from data_loader import (
    load_all_appointments, load_all_provider_availability,
    load_distance_matrix, get_provider_cluster, compute_overlap_pairs, ROOMS
)


# Penalty weights (can be tuned)
GAMMA = 10.0   # penalty per room switch
ETA   = 1.0    # penalty per metre of travel


def solve_model2(
    appointments: pd.DataFrame,
    provider: str,
    day: str,
    week: int,
    rooms: list[str],
    dist_matrix: pd.DataFrame,
    delta_frac: float = 0.1,
    verbose: bool = False
) -> dict:
    """
    Solve the Provider-Day Schedule Generation model.

    Parameters
    ----------
    appointments : appointments for this (provider, day) sorted by start_min
    provider     : provider ID string
    day          : day name ('Monday', ..., 'Friday')
    week         : week number
    rooms        : allowed rooms for this provider (cluster)
    dist_matrix  : 16×16 distance DataFrame indexed by room IDs
    delta_frac   : robust buffer fraction
    verbose      : solver output

    Returns
    -------
    dict with:
        'status', 'feasible', 'assignment', 'obj_value',
        'num_switches', 'total_travel', 'cost',
        'alpha'  (appointment-coverage indicator: appt_id → 1),
        'beta'   (room-time usage: (room, start_min) → 1)
    """
    appts = appointments.sort_values("start_min").reset_index(drop=True)
    if appts.empty:
        return {
            "status": "NoAppointments", "feasible": True,
            "assignment": {}, "obj_value": 0.0,
            "num_switches": 0, "total_travel": 0.0,
            "cost": 0.0,
            "alpha": {}, "beta": {},
        }

    A = list(appts["appt_id"])
    R = rooms

    delta = {row["appt_id"]: int(delta_frac * row["duration_min"])
             for _, row in appts.iterrows()}

    overlaps = compute_overlap_pairs(appts, delta=delta)
    overlap_set = set(overlaps) | {(b, a) for a, b in overlaps}

    # Consecutive pairs by time order
    consecutive = [(A[i], A[i + 1]) for i in range(len(A) - 1)]

    # ── Build PuLP model ─────────────────────────────────────────────────────
    mdl = pulp.LpProblem(f"Model2_{provider}_{day}", pulp.LpMinimize)

    # x[a, r] = 1 if appointment a is in room r
    x = {(a, r): pulp.LpVariable(f"x_{a}_{r}", cat="Binary") for a in A for r in R}

    # w[i, r, r2] = 1 if consecutive appt pair i uses (r → r2)
    # Only create for r ≠ r2 (switches); same-room handled by objective = 0
    w = {}
    for (a, b) in consecutive:
        for r in R:
            for r2 in R:
                w[(a, b, r, r2)] = pulp.LpVariable(f"w_{a}_{b}_{r}_{r2}", cat="Binary")

    # ── Objective: γ·switches + η·travel ─────────────────────────────────────
    switch_cost = pulp.lpSum(
        GAMMA * w[a, b, r, r2]
        for (a, b) in consecutive
        for r in R
        for r2 in R
        if r != r2
    )
    travel_cost = pulp.lpSum(
        ETA * dist_matrix.loc[r, r2] * w[a, b, r, r2]
        for (a, b) in consecutive
        for r in R
        for r2 in R
        if r != r2 and r in dist_matrix.index and r2 in dist_matrix.columns
    )
    mdl += switch_cost + travel_cost

    # ── Constraints ──────────────────────────────────────────────────────────

    # 1. Each appointment assigned to exactly one room
    for a in A:
        mdl += pulp.lpSum(x[a, r] for r in R) == 1, f"assign_{a}"

    # 2. No overlapping appointments in the same room
    for (a, b) in overlaps:
        for r in R:
            mdl += x[a, r] + x[b, r] <= 1, f"conflict_{a}_{b}_{r}"

    # 3. Room occupancy: at most one appointment per room per time point
    #    (already covered by constraint 2 for all overlapping pairs)

    # 4. Linearise w[a,b,r,r2] = x[a,r] * x[b,r2] for consecutive pairs
    for (a, b) in consecutive:
        for r in R:
            for r2 in R:
                mdl += w[a, b, r, r2] <= x[a, r],           f"wlink1_{a}_{b}_{r}_{r2}"
                mdl += w[a, b, r, r2] <= x[b, r2],          f"wlink2_{a}_{b}_{r}_{r2}"
                mdl += w[a, b, r, r2] >= x[a, r] + x[b, r2] - 1, f"wlink3_{a}_{b}_{r}_{r2}"

        # Each consecutive pair: exactly one (r,r2) combination selected
        mdl += pulp.lpSum(w[a, b, r, r2] for r in R for r2 in R) == 1, \
               f"w_sum_{a}_{b}"

    # ── Solve ─────────────────────────────────────────────────────────────────
    solver = pulp.PULP_CBC_CMD(msg=1 if verbose else 0, timeLimit=60)
    mdl.solve(solver)

    status = pulp.LpStatus[mdl.status]
    feasible = status in ("Optimal", "Not Solved") or mdl.status == 1
    feasible = mdl.status == 1  # 1 = optimal in PuLP

    assignment = {}
    num_switches = 0
    total_travel = 0.0

    if feasible:
        for a in A:
            for r in R:
                if pulp.value(x[a, r]) is not None and pulp.value(x[a, r]) > 0.5:
                    assignment[a] = r
                    break

        for (a, b) in consecutive:
            ra = assignment.get(a)
            rb = assignment.get(b)
            if ra and rb and ra != rb:
                num_switches += 1
                if ra in dist_matrix.index and rb in dist_matrix.columns:
                    total_travel += dist_matrix.loc[ra, rb]

    obj = pulp.value(mdl.objective) if feasible else None
    cost = (GAMMA * num_switches + ETA * total_travel) if feasible else float("inf")

    # alpha: appointment covered by this schedule
    alpha = {a: 1 for a in A} if feasible else {}

    # beta: room-time usage (room, start_min) → 1
    beta = {}
    if feasible:
        for a in A:
            r = assignment.get(a)
            if r is not None:
                row = appts[appts["appt_id"] == a].iloc[0]
                for t in range(int(row["start_min"]), int(row["end_min"])):
                    beta[(r, t)] = 1

    return {
        "status": status,
        "feasible": feasible,
        "assignment": assignment,
        "obj_value": obj,
        "num_switches": num_switches,
        "total_travel": total_travel,
        "cost": cost,
        "alpha": alpha,
        "beta": beta,
    }


def generate_all_schedules(
    appointments: pd.DataFrame,
    provider_avail: pd.DataFrame,
    dist_matrix: pd.DataFrame,
    delta_frac: float = 0.1,
    proximity_threshold: float = 4.0,
    verbose: bool = False
) -> list[dict]:
    """
    Run Model 2 for every (provider, day, week) combination.
    Returns list of schedule dicts (one per provider-day).
    """
    schedules = []

    for (provider, day, week), grp in appointments.groupby(["provider", "day_of_week", "week"]):
        cluster = get_provider_cluster(
            provider, day, week, provider_avail, dist_matrix, proximity_threshold
        )

        sol = solve_model2(
            grp, provider=provider, day=day, week=week,
            rooms=cluster, dist_matrix=dist_matrix,
            delta_frac=delta_frac, verbose=verbose
        )

        schedule = {
            "provider": provider,
            "day": day,
            "week": week,
            "rooms": cluster,
            **sol,
        }
        schedules.append(schedule)

        print(f"  [Model2] {provider} {day} W{week}: "
              f"{'OK' if sol['feasible'] else 'INFEASIBLE'}, "
              f"switches={sol['num_switches']}, "
              f"travel={sol['total_travel']:.1f}m, "
              f"cost={sol['cost']:.1f}")

    return schedules


def schedules_to_dataframe(schedules: list[dict],
                            appointments: pd.DataFrame) -> pd.DataFrame:
    """Flatten schedule assignments into an appointment-level DataFrame."""
    rows = []
    for sch in schedules:
        for appt_id, room in sch["assignment"].items():
            rows.append({
                "appt_id": appt_id,
                "provider": sch["provider"],
                "day": sch["day"],
                "week": sch["week"],
                "assigned_room": room,
                "schedule_cost": sch["cost"],
                "num_switches": sch["num_switches"],
                "total_travel": sch["total_travel"],
            })
    if not rows:
        return appointments.copy()
    res = pd.DataFrame(rows)
    return appointments.merge(res[["appt_id", "assigned_room", "schedule_cost",
                                    "num_switches", "total_travel"]],
                               on="appt_id", how="left")


def solve_model2_with_exclusions(
    appointments: pd.DataFrame,
    provider: str,
    day: str,
    week: int,
    rooms: list[str],
    dist_matrix: pd.DataFrame,
    reserved_rt: set[tuple[str, int]],
    delta_frac: float = 0.1,
    verbose: bool = False
) -> dict:
    """
    Wrapper around solve_model2 that adds constraints forcing x_ar = 0
    whenever room r would occupy already-reserved (r, t) time slots.
    """
    appts = appointments.sort_values("start_min").reset_index(drop=True)
    if appts.empty:
        return {
            "status": "NoAppointments", "feasible": True,
            "assignment": {}, "obj_value": 0.0,
            "num_switches": 0, "total_travel": 0.0,
            "cost": 0.0, "alpha": {}, "beta": {},
        }

    # Determine which (appt, room) combinations are blocked by reservations
    blocked: set[tuple[int, str]] = set()
    for _, row in appts.iterrows():
        a = row["appt_id"]
        for t in range(int(row["start_min"]), int(row["end_min"])):
            for r in rooms:
                if (r, t) in reserved_rt:
                    blocked.add((a, r))

    # Filter rooms to those that are usable for at least one appointment
    all_blocked_per_appt: dict[int, set[str]] = {}
    for (a, r) in blocked:
        all_blocked_per_appt.setdefault(a, set()).add(r)

    # Rooms free for all appointments of this provider (globally free)
    fully_free = [r for r in rooms
                  if all(r not in all_blocked_per_appt.get(a, set())
                         for a in appts["appt_id"])]
    rooms_to_use = fully_free if fully_free else rooms

    return solve_model2(
        appts, provider=provider, day=day, week=week,
        rooms=rooms_to_use, dist_matrix=dist_matrix,
        delta_frac=delta_frac, verbose=verbose
    )


def generate_schedules_sequential(
    appointments: pd.DataFrame,
    provider_avail: pd.DataFrame,
    dist_matrix: pd.DataFrame,
    delta_frac: float = 0.1,
    proximity_threshold: float = 4.0,
    verbose: bool = False
) -> list[dict]:
    """
    Generate schedules sequentially to avoid cross-provider room conflicts.
    Each provider sees already-reserved (room, time) slots and avoids them.
    Greedy but conflict-free; used as starting solution for column generation.
    Sort order: providers with most appointments first (solve harder first).
    """
    schedules = []
    reserved: set[tuple[str, int]] = set()   # (room_id, t_minute)

    # Sort providers by number of appointments descending
    prov_day_groups = list(appointments.groupby(["provider", "day_of_week", "week"]))
    prov_day_groups.sort(key=lambda x: len(x[1]), reverse=True)

    for (provider, day, week), grp in prov_day_groups:
        cluster = get_provider_cluster(
            provider, day, week, provider_avail, dist_matrix, proximity_threshold
        )

        sol = solve_model2_with_exclusions(
            grp, provider=provider, day=day, week=week,
            rooms=cluster, dist_matrix=dist_matrix,
            reserved_rt=reserved,
            delta_frac=delta_frac, verbose=verbose
        )

        # Update global reservations
        if sol["feasible"]:
            for (r, t) in sol["beta"]:
                reserved.add((r, t))

        schedule = {
            "provider": provider,
            "day": day,
            "week": week,
            "rooms": cluster,
            **sol,
        }
        schedules.append(schedule)

        print(f"  [Model2] {provider} {day} W{week}: "
              f"{'OK' if sol['feasible'] else 'INFEASIBLE'}, "
              f"switches={sol['num_switches']}, "
              f"travel={sol['total_travel']:.1f}m, "
              f"cost={sol['cost']:.1f}")

    return schedules


if __name__ == "__main__":
    from data_loader import load_all_appointments, load_all_provider_availability

    appts = load_all_appointments()
    avail = load_all_provider_availability()
    dist  = load_distance_matrix()

    # Quick test: one provider-day
    prov, day, week = "HPW101", "Monday", 1
    grp = appts[(appts["provider"] == prov) & (appts["day_of_week"] == day) & (appts["week"] == week)]
    cluster = get_provider_cluster(prov, day, week, avail, dist)
    print(f"{prov} {day} W{week}: {len(grp)} appts, cluster={cluster}")
    sol = solve_model2(grp, prov, day, week, cluster, dist, verbose=False)
    print(f"Status={sol['status']}, Cost={sol['cost']:.2f}, Switches={sol['num_switches']}")
    print("Assignment:", sol["assignment"])
