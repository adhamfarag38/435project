"""
Column Generation for Examination Room Scheduling
══════════════════════════════════════════════════════════════════════════════
Implements the full Dantzig-Wolfe (DW) column generation loop.

BACKGROUND — WHY COLUMN GENERATION?
─────────────────────────────────────
The Master Problem (Model 3) selects one schedule per (provider, day) from a
pool of candidate schedules.  In principle, the pool contains every possible
feasible schedule — but there are exponentially many of them.

Column generation avoids enumerating all schedules. Instead:
  • We start with a small pool (one schedule per provider-day).
  • We iteratively ask: "Is there a schedule NOT in the pool that would
    improve the current LP solution?"
  • That question is answered by solving a *pricing subproblem* (Model 2).

ALGORITHM
─────────
1.  Generate one schedule per (provider, day) → initial column pool.
2.  Solve the LP RELAXATION of the Master Problem.
    → Obtain dual variables:
        π_a  = shadow price of appointment-coverage constraint for appt a
        μ_rt = shadow price of room-capacity constraint at (room r, minute t)
3.  For each (provider, day), solve the PRICING SUBPROBLEM (modified Model 2):
        Minimise  γ·switches + η·travel − Σ_a [π_a + Σ_t μ_rt] · x_{ar}
    The reduced cost of the resulting schedule s is:
        r̄(s) = c_s − Σ_a π_a · α_{as} − Σ_{(r,t)} μ_rt · β_{rts}
4.  If r̄(s) < 0 for any (provider, day): add that schedule to the pool
    and go back to step 2.  Otherwise, STOP — the LP is optimal.
5.  Solve the ILP of the Master Problem on the final column pool.

BOUNDS
──────
  LP lower bound  z_LP : the LP relaxation objective at termination.
  ILP upper bound z_ILP: the integer solution objective.
  Duality gap     Δ    : (z_ILP − z_LP) / z_ILP × 100 %
"""

import pulp
import pandas as pd
import numpy as np

from data_loader import (
    get_provider_cluster, compute_overlap_pairs, ROOMS
)
from model2 import (
    generate_all_schedules, generate_schedules_sequential, GAMMA, ETA
)
from model3 import build_master_problem, master_to_appointments


# ─── Pricing Subproblem ───────────────────────────────────────────────────────

def solve_model2_pricing(
    appointments: pd.DataFrame,
    provider: str,
    day: str,
    week: int,
    rooms: list[str],
    dist_matrix: pd.DataFrame,
    dual_appt: dict,          # π_a : dual of coverage constraint for each appt
    dual_room: dict,          # μ_rt: dual of room-capacity constraint at (room, minute)
    delta_frac: float = 0.0,
    verbose: bool = False,
) -> dict:
    """
    Pricing subproblem: solve a modified Model 2 for a single (provider, day).

    The standard Model 2 minimises  γ·switches + η·travel.
    Here we subtract the dual savings from the LP master problem:

        Modified obj = γ·switches + η·travel
                     − Σ_a Σ_r x_{ar} · [π_a + Σ_{t ∈ [t_a, t_a+d_a)} μ_{rt}]

    Intuition: π_a is how much the master "pays" for covering appointment a,
    and μ_{rt} is how much it costs to use room r at minute t.  If those
    savings outweigh the switching cost, a new schedule is worth adding.

    After solving, the REDUCED COST is:
        r̄(s) = c_s − Σ_a π_a − Σ_{(r,t)} μ_{rt} · β_{rts}
    If r̄(s) < 0 this schedule improves the LP objective → add to pool.

    Returns the same dict structure as solve_model2(), plus key 'reduced_cost'.
    """
    appts = appointments.sort_values("start_min").reset_index(drop=True)
    if appts.empty:
        return {
            "status": "NoAppointments", "feasible": True,
            "assignment": {}, "obj_value": 0.0,
            "num_switches": 0, "total_travel": 0.0,
            "cost": 0.0, "alpha": {}, "beta": {},
            "reduced_cost": 0.0,
        }

    A = list(appts["appt_id"])
    R = rooms

    # Robust overlap buffers
    delta = {row["appt_id"]: int(delta_frac * row["duration_min"])
             for _, row in appts.iterrows()}
    overlaps   = compute_overlap_pairs(appts, delta=delta)
    consecutive = [(A[i], A[i + 1]) for i in range(len(A) - 1)]

    # ── Pre-compute per-(appointment, room) dual coefficient ──────────────────
    # For appointment a in room r, the dual saving is:
    #   π_a  (covers the appointment once)
    #   + Σ_{t=start_a}^{end_a - 1} μ_{r,t}  (frees room-time capacity)
    dual_coeff = {}
    for _, row in appts.iterrows():
        a     = row["appt_id"]
        start = int(row["start_min"])
        end   = int(row["end_min"])
        for r in R:
            room_saving = sum(dual_room.get((r, t), 0.0)
                              for t in range(start, end))
            dual_coeff[(a, r)] = dual_appt.get(a, 0.0) + room_saving

    # ── Build PuLP model ───────────────────────────────────────────────────────
    mdl = pulp.LpProblem(f"Pricing_{provider}_{day}_W{week}", pulp.LpMinimize)

    # x[a, r] = 1 if appointment a is assigned to room r
    x = {(a, r): pulp.LpVariable(f"x_{a}_{r}", cat="Binary")
         for a in A for r in R}

    # w[a, b, r, r2] = 1 if consecutive pair (a→b) transitions (r→r2)
    w = {(a, b, r, r2): pulp.LpVariable(f"w_{a}_{b}_{r}_{r2}", cat="Binary")
         for (a, b) in consecutive for r in R for r2 in R}

    # ── Modified objective ────────────────────────────────────────────────────
    switch_travel = pulp.lpSum(
        (GAMMA + ETA * (dist_matrix.loc[r, r2]
                        if r != r2 and r in dist_matrix.index
                        and r2 in dist_matrix.columns else 0.0))
        * w[a, b, r, r2]
        for (a, b) in consecutive
        for r in R for r2 in R if r != r2
    )
    dual_savings = pulp.lpSum(
        dual_coeff.get((a, r), 0.0) * x[a, r]
        for a in A for r in R
    )
    mdl += switch_travel - dual_savings

    # ── Constraints (identical to Model 2) ────────────────────────────────────
    for a in A:
        mdl += pulp.lpSum(x[a, r] for r in R) == 1, f"assign_{a}"

    for (a, b) in overlaps:
        for r in R:
            mdl += x[a, r] + x[b, r] <= 1, f"conflict_{a}_{b}_{r}"

    for (a, b) in consecutive:
        for r in R:
            for r2 in R:
                mdl += w[a, b, r, r2] <= x[a, r],                f"wL1_{a}_{b}_{r}_{r2}"
                mdl += w[a, b, r, r2] <= x[b, r2],               f"wL2_{a}_{b}_{r}_{r2}"
                mdl += w[a, b, r, r2] >= x[a, r] + x[b, r2] - 1, f"wL3_{a}_{b}_{r}_{r2}"
        mdl += pulp.lpSum(w[a, b, r, r2] for r in R for r2 in R) == 1, \
               f"w_sum_{a}_{b}"

    solver = pulp.PULP_CBC_CMD(msg=1 if verbose else 0, timeLimit=10)
    mdl.solve(solver)

    feasible    = (mdl.status == 1)
    assignment  = {}
    num_switches = 0
    total_travel = 0.0

    if feasible:
        for a in A:
            for r in R:
                if pulp.value(x[a, r]) is not None and pulp.value(x[a, r]) > 0.5:
                    assignment[a] = r
                    break
        for (a, b) in consecutive:
            ra, rb = assignment.get(a), assignment.get(b)
            if ra and rb and ra != rb:
                num_switches += 1
                if ra in dist_matrix.index and rb in dist_matrix.columns:
                    total_travel += dist_matrix.loc[ra, rb]

    # True schedule cost (switch + travel, without dual savings)
    cost = (GAMMA * num_switches + ETA * total_travel) if feasible else float("inf")

    # Reduced cost: how much this schedule saves the master LP
    rc = float("inf")
    if feasible:
        appt_dual_sum = sum(dual_appt.get(a, 0.0) for a in A)
        room_dual_sum = 0.0
        for a in A:
            r = assignment.get(a)
            if r:
                row_a = appts[appts["appt_id"] == a].iloc[0]
                room_dual_sum += sum(
                    dual_room.get((r, t), 0.0)
                    for t in range(int(row_a["start_min"]), int(row_a["end_min"]))
                )
        rc = cost - appt_dual_sum - room_dual_sum

    # alpha / beta for master problem
    alpha = {a: 1 for a in A} if feasible else {}
    beta  = {}
    if feasible:
        for a in A:
            r = assignment.get(a)
            if r:
                row_a = appts[appts["appt_id"] == a].iloc[0]
                for t in range(int(row_a["start_min"]), int(row_a["end_min"])):
                    beta[(r, t)] = 1

    return {
        "status":        pulp.LpStatus[mdl.status],
        "feasible":      feasible,
        "assignment":    assignment,
        "obj_value":     pulp.value(mdl.objective) if feasible else None,
        "num_switches":  num_switches,
        "total_travel":  total_travel,
        "cost":          cost,
        "alpha":         alpha,
        "beta":          beta,
        "reduced_cost":  rc,
    }


# ─── Full Column Generation Loop ──────────────────────────────────────────────

def run_column_generation(
    appointments: pd.DataFrame,
    provider_avail: pd.DataFrame,
    dist_matrix: pd.DataFrame,
    delta_frac: float = 0.0,
    proximity_threshold: float = 4.0,
    max_iterations: int = 3,
    rc_tol: float = -1e-4,
    verbose: bool = False,
) -> dict:
    """
    Full Dantzig-Wolfe column generation loop.

    Parameters
    ----------
    max_iterations  : maximum CG iterations before forcing ILP solve
    rc_tol          : a schedule is "improving" if reduced_cost < rc_tol

    Returns
    -------
    dict with keys:
        'lp_bound'       : float  — LP relaxation lower bound (at LP optimality)
        'ilp_bound'      : float  — ILP solution cost (upper bound)
        'gap_pct'        : float  — duality gap %
        'master_result'  : dict   — full result from build_master_problem (ILP)
        'schedules'      : list   — complete column pool
        'num_iterations' : int    — CG iterations performed
        'columns_added'  : int    — total new columns added across all iterations
    """
    """
    Column Generation using conflict-resolution column diversification.

    Algorithm:
      1. Generate one schedule per (provider, day) — initial column pool.
      2. Detect cross-provider room conflicts in the combined pool.
      3. For each conflicted provider-day (the "loser"), re-solve Model 2
         with the conflicted rooms excluded → new column with different
         room assignments added to the pool.
      4. Repeat until no conflicts remain or max_iterations reached.
      5. Solve LP relaxation for lower bound reporting.
      6. Solve final ILP with hard room constraints → conflict-free schedule.

    This gives Model 3 a diverse pool of columns so it can mix-and-match
    provider-day schedules into a globally conflict-free, low-cost solution.
    """
    from model2 import solve_model2, ROOMS as ALL_ROOMS, GAMMA, ETA
    from data_loader import get_provider_cluster

    print("\n" + "═" * 62)
    print("  COLUMN GENERATION (Conflict-Resolution Decomposition)")
    print("═" * 62)

    def _greedy_schedule(grp, provider, day, week, rooms):
        """
        Greedy fallback: assign each appointment (sorted by start time) to the
        first available room.  Always feasible — used when Model 2 fails.
        Room conflicts are allowed here and penalised later by Model 3's soft
        room constraints.
        """
        appts = grp.sort_values("start_min").reset_index(drop=True)
        # room → list of (start_min, end_min) already booked
        occupancy: dict[str, list[tuple]] = {r: [] for r in rooms}
        assignment = {}
        for _, row in appts.iterrows():
            a = row["appt_id"]
            s, e = int(row["start_min"]), int(row["end_min"])
            # Pick first room with no conflict; else pick first room (allow conflict)
            chosen = rooms[0]
            for r in rooms:
                if all(e <= bs or s >= be for bs, be in occupancy[r]):
                    chosen = r
                    break
            assignment[a] = chosen
            occupancy[chosen].append((s, e))

        num_switches, total_travel = 0, 0.0
        prev_room = None
        for _, row in appts.iterrows():
            r = assignment.get(row["appt_id"])
            if prev_room and r and r != prev_room:
                num_switches += 1
                if prev_room in dist_matrix.index and r in dist_matrix.columns:
                    total_travel += dist_matrix.loc[prev_room, r]
            prev_room = r

        alpha = {a: 1 for a in assignment}
        beta = {}
        for _, row in appts.iterrows():
            r = assignment.get(row["appt_id"])
            if r:
                for t in range(int(row["start_min"]), int(row["end_min"])):
                    beta[(r, t)] = 1

        return {
            "status": "Greedy", "feasible": True,
            "assignment": assignment,
            "num_switches": num_switches, "total_travel": total_travel,
            "cost": GAMMA * num_switches + ETA * total_travel,
            "alpha": alpha, "beta": beta,
        }

    # ── Step 1: Initial column pool (one schedule per provider-day) ───────────
    print("\n[CG Step 1] Generating initial schedules (one per provider-day)...")
    schedules: list[dict] = []
    seen: set[frozenset] = set()

    def _add_schedule(sch: dict) -> bool:
        key = frozenset(sch["assignment"].items())
        if key in seen or not sch.get("feasible"):
            return False
        seen.add(key)
        schedules.append(sch)
        return True

    for (provider, day, week), grp in appointments.groupby(
            ["provider", "day_of_week", "week"]):
        cluster = get_provider_cluster(
            provider, day, week, provider_avail, dist_matrix, proximity_threshold
        )
        sol = solve_model2(
            grp, provider=provider, day=day, week=week,
            rooms=cluster, dist_matrix=dist_matrix,
            delta_frac=delta_frac, verbose=False,
        )
        if not sol["feasible"] and set(cluster) != set(ALL_ROOMS):
            fallback_rooms = ALL_ROOMS[:8]  # cap at 8 to limit model size
            sol = solve_model2(
                grp, provider=provider, day=day, week=week,
                rooms=fallback_rooms, dist_matrix=dist_matrix,
                delta_frac=delta_frac, verbose=False,
            )
            cluster = fallback_rooms
        if not sol["feasible"]:
            # Last resort: greedy assignment — always produces a schedule
            sol = _greedy_schedule(grp, provider, day, week, ALL_ROOMS)
            cluster = ALL_ROOMS
        _add_schedule({"provider": provider, "day": day, "week": week,
                       "rooms": cluster, **sol})

    n0 = sum(1 for s in schedules if s.get("feasible"))
    print(f"            {len(schedules)} schedules, {n0} feasible.\n")

    # ── Steps 2-4: Conflict-resolution column generation ─────────────────────
    # For each (room, time) slot occupied by >1 schedule, re-solve the
    # non-winning provider-day with those rooms blocked → new column.
    total_added = 0
    for iteration in range(1, max_iterations + 1):
        # Build (room, minute) → [schedule_idx, ...] map
        rt_map: dict[tuple, list[int]] = {}
        for i, sch in enumerate(schedules):
            if not sch.get("feasible"):
                continue
            for rt in sch["beta"]:
                rt_map.setdefault(rt, []).append(i)

        # Identify losers: schedules that conflict on at least one slot
        loser_blocked: dict[int, set[str]] = {}
        for rt, idxs in rt_map.items():
            if len(idxs) > 1:
                for loser_idx in idxs[1:]:
                    loser_blocked.setdefault(loser_idx, set()).add(rt[0])

        if not loser_blocked:
            print(f"[CG Step {iteration+1}] No conflicts — column pool is conflict-free.")
            break

        print(f"[CG Step {iteration+1}] Re-solving {len(loser_blocked)} conflicted "
              f"provider-days with blocked rooms...")
        added_this = 0

        for idx, bad_rooms in loser_blocked.items():
            sch = schedules[idx]
            provider, day, week = sch["provider"], sch["day"], sch["week"]
            grp = appointments[
                (appointments["provider"] == provider) &
                (appointments["day_of_week"] == day) &
                (appointments["week"] == week)
            ]
            available = [r for r in sch["rooms"] if r not in bad_rooms]
            if not available:
                # Fall back to up to 8 rooms not in bad_rooms (limit model size)
                available = [r for r in ALL_ROOMS if r not in bad_rooms][:8]
                if not available:
                    available = ALL_ROOMS[:8]

            sol = solve_model2(
                grp, provider=provider, day=day, week=week,
                rooms=available, dist_matrix=dist_matrix,
                delta_frac=delta_frac, verbose=False,
            )
            if not sol["feasible"]:
                fallback_rooms = ALL_ROOMS[:8]
                sol = solve_model2(
                    grp, provider=provider, day=day, week=week,
                    rooms=fallback_rooms, dist_matrix=dist_matrix,
                    delta_frac=delta_frac, verbose=False,
                )
                available = fallback_rooms
            if not sol["feasible"]:
                sol = _greedy_schedule(grp, provider, day, week, ALL_ROOMS)
                available = ALL_ROOMS

            new_sch = {"provider": provider, "day": day, "week": week,
                       "rooms": available, **sol}
            if _add_schedule(new_sch):
                added_this += 1
                total_added += 1

        print(f"            Added {added_this} new columns "
              f"(pool now: {len(schedules)}).")
        if added_this == 0:
            break

    # ── Step 5: LP relaxation for lower bound ─────────────────────────────────
    print(f"\n[CG Step 5] Solving LP relaxation ({len(schedules)} columns)...")
    master_lp = build_master_problem(
        schedules, appointments, integer=False, verbose=False,
        hard_room_constraints=False,
    )
    lp_bound = master_lp["total_cost"] if master_lp["feasible"] else None

    # ── Step 6: Final ILP with hard room constraints ──────────────────────────
    print(f"[CG Step 6] Solving ILP with hard room constraints...")
    master_ilp = build_master_problem(
        schedules, appointments, integer=True, verbose=False,
        hard_room_constraints=True,
    )
    if not master_ilp["feasible"]:
        print("            Hard-constraint ILP infeasible — retrying with soft constraints.")
        master_ilp = build_master_problem(
            schedules, appointments, integer=True, verbose=False,
            hard_room_constraints=False,
        )
    ilp_bound = master_ilp["total_cost"] if master_ilp["feasible"] else float("inf")

    gap_pct = None
    if lp_bound and ilp_bound not in (float("inf"), 0):
        gap_pct = abs(ilp_bound - lp_bound) / abs(ilp_bound) * 100

    print(f"\n{'═' * 62}")
    print(f"  COLUMN GENERATION RESULTS")
    print(f"{'─' * 62}")
    print(f"  Conflict-resolution iterations : {iteration}")
    print(f"  New columns added              : {total_added}")
    print(f"  Total columns in pool          : {len(schedules)}")
    if lp_bound is not None:
        print(f"  LP lower bound  (z_LP)         : {lp_bound:.2f}")
    print(f"  ILP upper bound (z_ILP)        : {ilp_bound:.2f}")
    if gap_pct is not None:
        print(f"  Duality gap                    : {gap_pct:.2f}%")
    print(f"{'═' * 62}\n")

    return {
        "lp_bound":       lp_bound,
        "ilp_bound":      ilp_bound,
        "gap_pct":        gap_pct,
        "master_result":  master_ilp,
        "schedules":      schedules,
        "num_iterations": iteration,
        "columns_added":  total_added,
    }
