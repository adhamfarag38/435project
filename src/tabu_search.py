"""
Tabu Search for Examination Room Scheduling
══════════════════════════════════════════════════════════════════════════════
Applies Tabu Search (TS) as a post-optimisation metaheuristic, starting from
the integer solution produced by Column Generation and attempting to improve
it by swapping individual appointment room assignments.

BACKGROUND — WHY TABU SEARCH?
───────────────────────────────
The column generation ILP finds a good solution, but:
  • The ILP solver may time out before proving optimality.
  • The column pool may not contain the globally optimal assignment.

Tabu Search is a guided local search that escapes local optima by:
  (1) Always moving to the best available neighbour (even if worse).
  (2) Forbidding recently visited moves via a TABU LIST to prevent cycling.
  (3) Overriding the tabu list via an ASPIRATION CRITERION when a move
      produces a solution better than the global best.

SOLUTION REPRESENTATION
────────────────────────
A solution is a flat dict:
    assignment: {appt_id → room_id}
covering all appointments that received a room in the master ILP solution.

NEIGHBOURHOOD STRUCTURE
────────────────────────
N(x) = { x' : x' differs from x by reassigning exactly ONE appointment
               to a different room in that provider's allowed cluster }

OBJECTIVE FUNCTION  (penalised cost)
──────────────────────────────────────
f(x) = γ · switches(x)  +  η · travel(x)  +  P · conflicts(x)

  switches(x): number of consecutive room changes per provider-day
  travel(x)  : total metres walked between consecutive rooms
  conflicts(x): number of (room, time-slot) double-bookings
  P = 10 000  (large penalty to strongly discourage infeasibility)

INCREMENTAL COST UPDATE
────────────────────────
Re-computing f(x) from scratch for every candidate move is expensive.
Instead, when appointment a moves from old_room to new_room, only two
things change:

  ΔSwitch: only the two transitions adjacent to a in the sequence
           (pred → a) and (a → succ) are affected.

  ΔConflict: only appointments that OVERLAP IN TIME with a can create
             or remove a conflict with a.  We precompute overlap_adj[a]
             (the set of time-overlapping appointments) once before the loop.

Full cost is recomputed from scratch every RECOMPUTE_EVERY iterations to
prevent floating-point drift from accumulating.

TABU LIST
──────────
A FIFO queue (collections.deque, maxlen = k_max) stores (appt_id, old_room)
pairs — the move that was just made.  A parallel set (tabu_set) allows O(1)
membership testing (deque lookup is O(n)).

ASPIRATION CRITERION
─────────────────────
A tabu move is ALLOWED if the resulting cost beats the global best:
    if candidate_cost < best_cost: override tabu status.

STOPPING CRITERIA
──────────────────
  • max_iterations: hard limit on total TS iterations.
  • max_no_improve: stop early if the global best has not improved for
    this many consecutive iterations.
"""

import pandas as pd
import numpy as np
from collections import deque

from data_loader import ROOMS, compute_overlap_pairs, get_provider_cluster

# Objective weights (must match Model 2 for consistency)
GAMMA       = 10.0    # penalty per room switch
ETA         = 1.0     # penalty per metre of travel
P_CONFLICT  = 1e4     # penalty per (room, time) double-booking
RECOMPUTE_EVERY = 50  # full recompute every N iterations to fix float drift


# ─── Precomputation ───────────────────────────────────────────────────────────

def _build_ts_state(
    selected_schedules: list[dict],
    appointments: pd.DataFrame,
    provider_avail: pd.DataFrame,
    dist_matrix: pd.DataFrame,
    proximity_threshold: float,
) -> dict:
    """
    Build all precomputed data structures needed by the TS inner loop.

    Returns a dict with:
      assignment      : {appt_id → room_id}  mutable working copy
      appt_data       : {appt_id → {start, end, provider, day, week}}
      consecutive_map : {(prov,day,week) → [appt_id, ...] sorted by start_min}
      position_map    : {appt_id → index in its provider-day sequence}  O(1) lookup
      overlap_adj     : {appt_id → set of overlapping appt_ids}  for conflict delta
      allowed_rooms   : {appt_id → [room, ...]}  provider cluster
    """
    # Flat assignment dict from selected schedules
    assignment = {}
    for sch in selected_schedules:
        for appt_id, room in sch["assignment"].items():
            assignment[appt_id] = room

    # Per-appointment data (avoid repeated DataFrame lookups in the inner loop)
    appt_data = {}
    for _, row in appointments.iterrows():
        appt_data[row["appt_id"]] = {
            "start_min":  int(row["start_min"]),
            "end_min":    int(row["end_min"]),
            "provider":   row["provider"],
            "day":        row["day_of_week"],
            "week":       row["week"],
        }

    # Provider-day sequences (sorted by start time) — ONLY assigned appointments.
    # Including unassigned appointments creates None-room gaps that silently
    # suppress switch costs between adjacent assigned appointments, causing
    # compute_full_cost to undercount relative to the ILP's schedule costs.
    assigned_ids = set(assignment.keys())
    assigned_appts = appointments[appointments["appt_id"].isin(assigned_ids)]
    consecutive_map = {}
    position_map    = {}
    for (prov, day, week), grp in assigned_appts.groupby(
            ["provider", "day_of_week", "week"]):
        seq = list(grp.sort_values("start_min")["appt_id"])
        key = (prov, day, week)
        consecutive_map[key] = seq
        for idx, a in enumerate(seq):
            position_map[a] = idx

    # Overlap adjacency list — only among assigned appointments
    overlap_adj = {a: set() for a in assigned_ids}
    rows = assigned_appts.to_dict("records")
    for i, ra in enumerate(rows):
        for rb in rows[i + 1:]:
            if ra["start_min"] < rb["end_min"] and rb["start_min"] < ra["end_min"]:
                overlap_adj[ra["appt_id"]].add(rb["appt_id"])
                overlap_adj[rb["appt_id"]].add(ra["appt_id"])

    # Allowed rooms per appointment (provider cluster)
    allowed_rooms = {}
    for appt_id, info in appt_data.items():
        cluster = get_provider_cluster(
            info["provider"], info["day"], info["week"],
            provider_avail, dist_matrix, proximity_threshold
        )
        allowed_rooms[appt_id] = cluster

    return {
        "assignment":      assignment,
        "appt_data":       appt_data,
        "consecutive_map": consecutive_map,
        "position_map":    position_map,
        "overlap_adj":     overlap_adj,
        "allowed_rooms":   allowed_rooms,
    }


# ─── Cost Functions ───────────────────────────────────────────────────────────

def _transition_cost(r1, r2, dist_matrix) -> float:
    """Switch + travel cost for moving from room r1 to room r2."""
    if r1 is None or r2 is None or r1 == r2:
        return 0.0
    dist = dist_matrix.loc[r1, r2] if (r1 in dist_matrix.index
                                        and r2 in dist_matrix.columns) else 0.0
    return GAMMA + ETA * dist


def compute_full_cost(
    assignment: dict,
    appt_data: dict,
    consecutive_map: dict,
    dist_matrix: pd.DataFrame,
) -> float:
    """
    Compute f(x) = γ·switches + η·travel + P·conflicts from scratch.
    Called at initialisation and periodically to correct floating-point drift.
    """
    total = 0.0

    # Switch + travel costs
    for seq in consecutive_map.values():
        for i in range(len(seq) - 1):
            r1 = assignment.get(seq[i])
            r2 = assignment.get(seq[i + 1])
            total += _transition_cost(r1, r2, dist_matrix)

    # Conflict costs: (room, time) double-bookings
    room_time: dict[tuple, int] = {}
    for appt_id, room in assignment.items():
        info = appt_data[appt_id]
        for t in range(info["start_min"], info["end_min"]):
            key = (room, t)
            room_time[key] = room_time.get(key, 0) + 1

    for count in room_time.values():
        if count > 1:
            total += P_CONFLICT * (count - 1)

    return total


def _incremental_delta(
    appt_id: int,
    new_room: str,
    assignment: dict,
    appt_data: dict,
    consecutive_map: dict,
    position_map: dict,
    overlap_adj: dict,
    dist_matrix: pd.DataFrame,
) -> float:
    """
    Compute the CHANGE in objective from reassigning appt_id to new_room.

    Only two things can change:
      (1) The two transitions adjacent to appt_id in the provider-day sequence.
      (2) Conflicts with appointments that overlap in time with appt_id.

    Returns delta (negative = improvement).
    """
    old_room = assignment.get(appt_id)
    if old_room == new_room:
        return float("inf")     # Not a real move

    info = appt_data[appt_id]
    key  = (info["provider"], info["day"], info["week"])
    seq  = consecutive_map.get(key, [appt_id])
    idx  = position_map.get(appt_id, 0)

    # ── ΔSwitch/Travel: only the two adjacent transitions change ──────────────
    pred_room = assignment.get(seq[idx - 1]) if idx > 0         else None
    succ_room = assignment.get(seq[idx + 1]) if idx < len(seq) - 1 else None

    old_trans = (_transition_cost(pred_room, old_room, dist_matrix)
                 + _transition_cost(old_room, succ_room, dist_matrix))
    new_trans = (_transition_cost(pred_room, new_room, dist_matrix)
                 + _transition_cost(new_room, succ_room, dist_matrix))
    delta_switch = new_trans - old_trans

    # ── ΔConflict: only time-overlapping appointments matter ──────────────────
    delta_conf = 0
    for other_id in overlap_adj.get(appt_id, set()):
        other_room = assignment.get(other_id)
        if other_room is None:
            continue
        was_conflict = (other_room == old_room)
        now_conflict = (other_room == new_room)
        delta_conf  += int(now_conflict) - int(was_conflict)

    return delta_switch + P_CONFLICT * delta_conf


# ─── Main Tabu Search ────────────────────────────────────────────────────────

def run_tabu_search(
    selected_schedules: list[dict],
    appointments: pd.DataFrame,
    provider_avail: pd.DataFrame,
    dist_matrix: pd.DataFrame,
    proximity_threshold: float = 4.0,
    k_max: int = 20,
    max_iterations: int = 300,
    max_no_improve: int = 75,
    verbose: bool = False,
) -> dict:
    """
    Tabu Search over single-appointment room-swap neighbourhood.

    Parameters
    ----------
    selected_schedules  : schedules selected by the CG ILP (starting point)
    k_max               : tabu list size (FIFO queue length)
    max_iterations      : hard iteration limit
    max_no_improve      : early stop after this many iters without improvement

    Returns
    -------
    dict with:
        'best_assignment'  : dict  — best {appt_id → room} found
        'best_cost'        : float — best objective value
        'initial_cost'     : float — starting cost (from ILP solution)
        'improvement_pct'  : float — % improvement over initial cost
        'final_switches'   : int
        'final_travel'     : float
        'final_conflicts'  : int
        'history'          : pd.DataFrame — per-iteration log
        'iterations_run'   : int
    """
    print("\n" + "═" * 62)
    print("  TABU SEARCH  (Post-Optimisation Metaheuristic)")
    print("═" * 62)
    print(f"  Tabu list size  k_max : {k_max}")
    print(f"  Max iterations        : {max_iterations}")
    print(f"  Early-stop (no impr.) : {max_no_improve}\n")

    # ── Precompute state ───────────────────────────────────────────────────────
    state = _build_ts_state(
        selected_schedules, appointments,
        provider_avail, dist_matrix, proximity_threshold
    )
    assignment      = state["assignment"]
    appt_data       = state["appt_data"]
    consecutive_map = state["consecutive_map"]
    position_map    = state["position_map"]
    overlap_adj     = state["overlap_adj"]
    allowed_rooms   = state["allowed_rooms"]

    scheduled_ids = list(assignment.keys())

    # ── Initialise ─────────────────────────────────────────────────────────────
    current_cost = compute_full_cost(
        assignment, appt_data, consecutive_map, dist_matrix
    )
    best_cost       = current_cost
    best_assignment = assignment.copy()

    print(f"[TS] Starting cost: {current_cost:.2f}  "
          f"({len(scheduled_ids)} appointments assigned)\n")

    # Sanity check: TS objective must be >= LP relaxation.
    # If current_cost is dramatically lower than the ILP value reported by CG,
    # the TS objective is computing a different quantity — stop and diagnose.
    # (This guard is informational; remove the assert if you want soft warnings.)

    # Tabu list: FIFO deque of (appt_id, old_room)
    tabu_deque: deque = deque(maxlen=k_max)
    tabu_set:   set   = set()           # mirror set for O(1) lookup

    history       = []
    no_improve    = 0
    iteration     = 0

    # ── Main loop ─────────────────────────────────────────────────────────────
    for iteration in range(1, max_iterations + 1):

        # Periodically recompute cost to eliminate floating-point drift
        if iteration % RECOMPUTE_EVERY == 0:
            current_cost = compute_full_cost(
                assignment, appt_data, consecutive_map, dist_matrix
            )

        best_cand_delta = float("inf")
        best_cand       = None           # (appt_id, old_room, new_room)

        # ── Evaluate all single-swap neighbours ────────────────────────────────
        for appt_id in scheduled_ids:
            old_room = assignment[appt_id]

            for new_room in allowed_rooms.get(appt_id, []):
                if new_room == old_room:
                    continue

                delta = _incremental_delta(
                    appt_id, new_room,
                    assignment, appt_data,
                    consecutive_map, position_map, overlap_adj,
                    dist_matrix,
                )
                cand_cost = current_cost + delta
                is_tabu   = (appt_id, old_room) in tabu_set

                # Aspiration criterion: allow tabu move if it beats global best
                aspiration = cand_cost < best_cost

                if (not is_tabu or aspiration) and delta < best_cand_delta:
                    best_cand_delta = delta
                    best_cand       = (appt_id, old_room, new_room)

        # If ALL moves are tabu and no aspiration applies, force best tabu move
        if best_cand is None:
            for appt_id in scheduled_ids:
                old_room = assignment[appt_id]
                for new_room in allowed_rooms.get(appt_id, []):
                    if new_room == old_room:
                        continue
                    delta = _incremental_delta(
                        appt_id, new_room,
                        assignment, appt_data,
                        consecutive_map, position_map, overlap_adj,
                        dist_matrix,
                    )
                    if delta < best_cand_delta:
                        best_cand_delta = delta
                        best_cand       = (appt_id, old_room, new_room)

        if best_cand is None:
            print(f"[TS] No valid move at iteration {iteration}. Stopping.")
            break

        # ── Apply best move ────────────────────────────────────────────────────
        appt_id, old_room, new_room = best_cand
        assignment[appt_id] = new_room
        current_cost       += best_cand_delta

        # ── Update tabu list (FIFO) ────────────────────────────────────────────
        if len(tabu_deque) == k_max:
            # The oldest entry is about to be evicted from the deque
            evicted = tabu_deque[0]
            tabu_set.discard(evicted)
        tabu_deque.append((appt_id, old_room))
        tabu_set.add((appt_id, old_room))

        # ── Update best solution ───────────────────────────────────────────────
        if current_cost < best_cost:
            best_cost       = current_cost
            best_assignment = assignment.copy()
            no_improve      = 0
            marker          = " ← new best"
        else:
            no_improve += 1
            marker      = ""

        history.append({
            "iteration":    iteration,
            "current_cost": current_cost,
            "best_cost":    best_cost,
            "delta":        best_cand_delta,
            "appt_moved":   appt_id,
            "new_room":     new_room,
            "tabu_list_sz": len(tabu_deque),
        })

        if verbose or iteration % 50 == 0 or iteration == 1:
            print(f"[TS] iter={iteration:4d} | "
                  f"current={current_cost:10.2f} | "
                  f"best={best_cost:10.2f} | "
                  f"no_impr={no_improve:3d}{marker}")

        # ── Early stopping ─────────────────────────────────────────────────────
        if no_improve >= max_no_improve:
            print(f"\n[TS] No improvement for {max_no_improve} iterations "
                  f"— stopping early.")
            break

    # ── Final full-cost recompute (canonical answer) ──────────────────────────
    best_cost = compute_full_cost(
        best_assignment, appt_data, consecutive_map, dist_matrix
    )

    # Decompose final cost for reporting
    switch_total = 0
    travel_total = 0.0
    for seq in consecutive_map.values():
        for i in range(len(seq) - 1):
            r1 = best_assignment.get(seq[i])
            r2 = best_assignment.get(seq[i + 1])
            if r1 and r2 and r1 != r2:
                switch_total += 1
                if r1 in dist_matrix.index and r2 in dist_matrix.columns:
                    travel_total += dist_matrix.loc[r1, r2]

    rt_usage: dict[tuple, int] = {}
    for appt_id, room in best_assignment.items():
        info = appt_data[appt_id]
        for t in range(info["start_min"], info["end_min"]):
            key = (room, t)
            rt_usage[key] = rt_usage.get(key, 0) + 1
    conflicts_total = sum(c - 1 for c in rt_usage.values() if c > 1)

    # Build result DataFrame
    init_cost = compute_full_cost(
        {appt_id: sch["assignment"][appt_id]
         for sch in selected_schedules
         for appt_id in sch["assignment"]},
        appt_data, consecutive_map, dist_matrix
    )
    improvement_pct = (init_cost - best_cost) / init_cost * 100 if init_cost > 0 else 0.0

    best_df = appointments.copy()
    best_df["assigned_room"] = best_df["appt_id"].map(best_assignment)

    print(f"\n{'═' * 62}")
    print(f"  TABU SEARCH RESULTS")
    print(f"{'─' * 62}")
    print(f"  Iterations run   : {iteration}")
    print(f"  Initial cost     : {init_cost:.2f}")
    print(f"  Best cost found  : {best_cost:.2f}")
    print(f"  Improvement      : {improvement_pct:.2f}%")
    print(f"  Final switches   : {switch_total}")
    print(f"  Final travel     : {travel_total:.1f} m")
    print(f"  Final conflicts  : {conflicts_total}")
    print(f"{'═' * 62}\n")

    return {
        "best_assignment":  best_assignment,
        "best_cost":        best_cost,
        "initial_cost":     init_cost,
        "improvement_pct":  improvement_pct,
        "final_switches":   switch_total,
        "final_travel":     travel_total,
        "final_conflicts":  conflicts_total,
        "best_df":          best_df,
        "history":          pd.DataFrame(history),
        "iterations_run":   iteration,
    }
