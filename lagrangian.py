"""
Lagrangian Relaxation for Examination Room Scheduling
══════════════════════════════════════════════════════════════════════════════
Applies Lagrangian Relaxation (LR) to Model 3 (Master Schedule Selection)
to obtain a rigorous LOWER BOUND on the optimal scheduling cost.

BACKGROUND — WHY LAGRANGIAN RELAXATION?
─────────────────────────────────────────
The Master Problem (Model 3) has two types of constraints:
  (A) Appointment coverage:      Σ_s α_{as} λ_s = 1   ∀ a   ← RELAX THESE
  (B) Room capacity (soft):      Σ_s β_{rts} λ_s ≤ 1  ∀ r,t
  (C) One schedule per provider: Σ_{s∈S(p,d)} λ_s = 1  ∀ p,d

By moving (A) into the objective with multipliers u_a ≥ 0, the relaxed
problem is easier to solve and its optimal value is a valid lower bound:

  L(u) = Σ_a u_a  +  min_{λ,σ} [ Σ_s (c_s − Σ_a u_a α_{as}) λ_s  +  M Σ σ_{rt} ]
                       subject to (B) and (C) and λ ∈ {0,1}, σ ≥ 0

  L(u) ≤ z*  for ANY u ≥ 0,  where z* is the true ILP optimal.

The LAGRANGIAN DUAL maximises L(u) over u ≥ 0:
  z_LD = max_{u ≥ 0} L(u)   ≤   z*

CUTTING PLANE (MASTER PROBLEM) METHOD
──────────────────────────────────────
Instead of the subgradient method, we solve the Lagrangian dual exactly
via the cutting plane (Kelley's) approach:

  Outer approximation of  max_{u≥0} L(u)  as a master LP:

      max  z_master
      s.t. z_master ≤ L(u^h)  ∀ h generated so far
           u_a ≥ 0

  Equivalently, in minimisation form (let θ = −z_master):

      min  Σ_a u_a + θ
      s.t. θ + Σ_a coverage_a^h · u_a ≥ SP_obj_raw^h   ∀ h
           u_a ≥ 0,   θ free

  where:
      coverage_a^h = Σ_s α_{as} λ_s^h        (how much appt a is covered)
      SP_obj_raw^h = sub_cost^h + Σ_a u^h_a · coverage_a^h
                   = raw schedule cost without the multiplier terms

  Each iteration:
    1. Solve subproblem at current u → (λ^h, L(u^h), coverage^h)
    2. Add cut to master LP
    3. Re-solve master LP → new u, new lower bound z_master
    4. z_lag  = min over h of L(u^h)   (best feasible upper bound on z_LD)
    5. z_master                         (lower bound on z_LD)
    6. Convergence: z_lag − z_master < tol

BOUNDS HIERARCHY
────────────────
  z_master  ≤  z_LD  ≤  z_LP  ≤  z*  ≤  z_ILP
"""

import pulp
import pandas as pd
import numpy as np


# ─── Lagrangian Subproblem ────────────────────────────────────────────────────

def solve_lagrangian_subproblem(
    schedules: list[dict],
    appointments: pd.DataFrame,
    u: dict,
    verbose: bool = False,
) -> dict:
    """
    Solve the Lagrangian subproblem: Model 3 WITHOUT appointment-coverage
    constraints, with modified schedule costs.

    Modified cost for schedule s:
        c_s^u = c_s − Σ_{a ∈ α_s} u_a

    The subproblem minimises:
        Σ_s c_s^u · λ_s + M · Σ_{(r,t)} σ_{rt}

    subject to:
        Σ_s β_{rts} λ_s − σ_{rt} ≤ 1   ∀ (r,t)    [room capacity, soft]
        Σ_{s ∈ S(p,d)} λ_s = 1           ∀ (p,d)    [one schedule per day]
        λ_s ∈ {0,1},  σ_{rt} ≥ 0

    Lagrangian value:
        L(u) = Σ_a u_a  +  (subproblem objective value)

    Returns
    -------
    dict with:
        'feasible'          : bool
        'L_u'               : float   — L(u), the lower bound at current u
        'subproblem_cost'   : float   — raw subproblem objective
        'lambda_values'     : dict    — {schedule_index → λ_s value}
        'selected_scheds'   : list    — feasible schedules used
    """
    feasible_scheds = [s for s in schedules if s.get("feasible", False)]
    if not feasible_scheds:
        return {
            "feasible": False, "L_u": float("inf"),
            "subproblem_cost": float("inf"),
            "lambda_values": {}, "selected_scheds": [],
        }

    S = list(range(len(feasible_scheds)))
    ROOM_PENALTY = 1e4

    # Modified costs: subtract multiplier savings
    mod_costs = []
    for sch in feasible_scheds:
        u_saving = sum(u.get(a, 0.0) for a in sch["alpha"])
        mod_costs.append(sch["cost"] - u_saving)

    # Provider-day groups (for one-schedule-per-day constraint)
    pd_groups = {}
    for s, sch in enumerate(feasible_scheds):
        key = (sch["provider"], sch["day"], sch["week"])
        pd_groups.setdefault(key, []).append(s)

    # Room-time slots used by more than one schedule (only these need capacity constraints)
    rt_usage: dict[tuple, list[int]] = {}
    for s, sch in enumerate(feasible_scheds):
        for rt in sch["beta"]:
            rt_usage.setdefault(rt, []).append(s)
    conflict_rts = {rt: users for rt, users in rt_usage.items() if len(users) > 1}

    # ── Build PuLP model ──────────────────────────────────────────────────────
    mdl = pulp.LpProblem("LR_Subproblem", pulp.LpMinimize)

    lam = {s: pulp.LpVariable(f"lam_{s}", cat="Binary") for s in S}
    room_slack = {}
    for (r, t), users in conflict_rts.items():
        sl = pulp.LpVariable(f"sl_{r}_{t}", lowBound=0)
        room_slack[(r, t)] = sl
        mdl += (pulp.lpSum(lam[s] for s in users) - sl <= 1,
                f"room_{r}_{t}")

    mdl += (
        pulp.lpSum(mod_costs[s] * lam[s] for s in S)
        + pulp.lpSum(ROOM_PENALTY * sl for sl in room_slack.values())
    )

    for (prov, day, week), s_list in pd_groups.items():
        tag = f"{prov}_{day}_W{week}".replace(" ", "_")
        mdl += pulp.lpSum(lam[s] for s in s_list) == 1, f"one_{tag}"

    solver = pulp.PULP_CBC_CMD(msg=1 if verbose else 0, timeLimit=60)
    mdl.solve(solver)

    if mdl.status != 1:
        return {
            "feasible": False, "L_u": float("inf"),
            "subproblem_cost": float("inf"),
            "lambda_values": {}, "selected_scheds": feasible_scheds,
        }

    lam_vals = {s: (pulp.value(lam[s]) or 0.0) for s in S}
    sub_cost  = pulp.value(mdl.objective)

    # L(u) = Σ_a u_a + subproblem objective
    u_constant = sum(u.get(a, 0.0)
                     for a in appointments["appt_id"].tolist())
    L_u = u_constant + sub_cost

    return {
        "feasible":         True,
        "L_u":              L_u,
        "subproblem_cost":  sub_cost,
        "lambda_values":    lam_vals,
        "selected_scheds":  feasible_scheds,
    }


# ─── Cutting Plane (Master Problem) Loop ──────────────────────────────────────

def run_lagrangian_relaxation(
    schedules: list[dict],
    appointments: pd.DataFrame,
    z_ub: float,
    max_iterations: int = 150,
    lambda_init: float = 2.0,
    lambda_factor: float = 0.5,
    patience: int = 15,
    gap_tol: float = 0.01,
    verbose: bool = False,
) -> dict:
    """
    Cutting plane (Kelley's) method for the Lagrangian dual of Model 3.

    Replaces the subgradient method with an exact outer-linearisation approach.
    The legacy parameters lambda_init, lambda_factor, patience are accepted for
    API compatibility but are unused by this implementation.

    Algorithm
    ---------
    Master LP (minimisation form):
        min  Σ_a u_a + θ
        s.t. θ + Σ_a coverage_a^h · u_a ≥ SP_obj_raw^h   [cut for iter h]
             u_a ≥ 0  ∀ a
             θ free (bounded: -1e8 ≤ θ ≤ 1e8)

    where SP_obj_raw^h = sub_cost^h + Σ_a u^h_a · coverage_a^h
          coverage_a^h = Σ_s α_{as} · λ_s^h

    Bounds tracked:
        z_lag    = min_h L(u^h)   — best feasible value of Lagrangian dual
        z_master = master LP obj  — lower bound on z_LD
        gap      = z_lag − z_master

    Convergence: gap < 1e-4  or  max_iterations reached.

    Parameters
    ----------
    z_ub          : best known upper bound (from Column Generation / ILP)
    max_iterations: maximum cutting plane iterations
    lambda_init   : unused (kept for API compatibility)
    lambda_factor : unused (kept for API compatibility)
    patience      : unused (kept for API compatibility)
    gap_tol       : stop if (z_UB − z_lag) / z_UB < gap_tol

    Returns
    -------
    dict with:
        'best_lower_bound' : float        — best L(u) = z_lag achieved
        'upper_bound'      : float        — z_ub passed in
        'final_gap_pct'    : float        — (z_UB − z_lag) / z_UB × 100
        'multipliers'      : dict         — final u_a values
        'history'          : pd.DataFrame — per-iteration log
        'num_iterations'   : int
    """
    print("\n" + "═" * 62)
    print("  LAGRANGIAN RELAXATION  (Cutting Plane Method)")
    print("═" * 62)
    print(f"  Upper bound z_UB : {z_ub:.4f}")
    print(f"  Max iterations   : {max_iterations}")
    print(f"  Conv. tolerance  : 1e-4 (z_lag − z_master)\n")

    all_appt_ids = list(appointments["appt_id"])

    # ── Initialise master LP ───────────────────────────────────────────────────
    master = pulp.LpProblem("LR_Master", pulp.LpMinimize)

    u_vars = {a: pulp.LpVariable(f"u_{a}", lowBound=0) for a in all_appt_ids}
    theta  = pulp.LpVariable("theta", lowBound=-1e8, upBound=1e8)

    master += pulp.lpSum(u_vars[a] for a in all_appt_ids) + theta

    # Start with u = 0 (warm-start point for first subproblem)
    u = {a: 0.0 for a in all_appt_ids}

    z_lag     = float("inf")   # best (minimum) L(u^h) seen — tracks z_LD upper side
    z_master  = -float("inf")  # master LP objective       — tracks z_LD lower side
    history   = []
    k         = 0
    cut_count = 0

    master_solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=30)

    for k in range(1, max_iterations + 1):

        # ── Step 1: Solve Lagrangian subproblem at current u ───────────────────
        sub = solve_lagrangian_subproblem(
            schedules, appointments, u, verbose=False
        )
        if not sub["feasible"]:
            print(f"[LR] k={k}: subproblem infeasible — stopping.")
            break

        L_u         = sub["L_u"]
        lam_vals    = sub["lambda_values"]
        feas_scheds = sub["selected_scheds"]
        sub_cost    = sub["subproblem_cost"]

        # ── Step 2: Compute coverage_a^h = Σ_s α_{as} · λ_s^h ────────────────
        coverage = {}
        for a in all_appt_ids:
            coverage[a] = sum(
                lam_vals.get(s, 0.0)
                for s, sch in enumerate(feas_scheds)
                if a in sch["alpha"]
            )

        # ── Step 3: Compute SP_obj_raw^h ──────────────────────────────────────
        # sub_cost = Σ_s (c_s - Σ_a u_a α_{as}) λ_s  +  M Σ σ_{rt}
        # SP_obj_raw = sub_cost + Σ_a u_a · coverage_a
        #            = raw schedule cost (without multiplier adjustment)
        SP_obj_raw = sub_cost + sum(u[a] * coverage[a] for a in all_appt_ids)

        # ── Step 4: Update z_lag ───────────────────────────────────────────────
        improved = L_u < z_lag
        if improved:
            z_lag = L_u

        # ── Step 5: Add linearisation cut to master LP ────────────────────────
        # θ + Σ_a coverage_a^h · u_a ≥ SP_obj_raw^h
        cut_name = f"cut_{k}"
        master += (
            theta + pulp.lpSum(coverage[a] * u_vars[a] for a in all_appt_ids)
            >= SP_obj_raw,
            cut_name,
        )
        cut_count += 1

        # ── Step 6: Solve master LP → new lower bound and new u ───────────────
        master.solve(master_solver)

        if master.status == 1:
            z_master = pulp.value(master.objective)
            u = {a: max(0.0, pulp.value(u_vars[a]) or 0.0)
                 for a in all_appt_ids}
        else:
            # Master infeasible/unbounded (should not happen after first cut)
            if verbose:
                print(f"[LR] k={k}: master LP status {master.status} — "
                      f"keeping previous u.")

        # ── Step 7: Compute gaps and log ──────────────────────────────────────
        primal_gap = z_lag - z_master   # gap between best L and master bound
        gap_pct    = (z_ub - z_lag) / abs(z_ub) * 100 if z_ub != 0 else float("inf")

        history.append({
            "iteration": k,
            "L_u":       L_u,
            "best_L":    z_lag,
            "gap_pct":   gap_pct,
            "z_master":  z_master,
            "improved":  improved,
        })

        if k == 1 or k % 10 == 0:
            print(f"[LR] k={k:4d} | L(u) = {L_u:12.4f} | "
                  f"z_lag = {z_lag:12.4f} | z_master = {z_master:12.4f} | "
                  f"gap = {gap_pct:6.2f}%")

        # ── Step 8: Convergence check ──────────────────────────────────────────
        if primal_gap < 1e-4:
            print(f"\n[LR] Cutting plane gap {primal_gap:.6f} < 1e-4 — converged.")
            break

        if gap_pct < gap_tol * 100:
            print(f"\n[LR] UB gap {gap_pct:.4f}% below tolerance — converged.")
            break

    final_gap = (z_ub - z_lag) / abs(z_ub) * 100 if z_ub != 0 else float("inf")

    print(f"\n{'═' * 62}")
    print(f"  LAGRANGIAN RELAXATION RESULTS  (Cutting Plane)")
    print(f"{'─' * 62}")
    print(f"  Iterations run      : {k}")
    print(f"  Cuts added          : {cut_count}")
    print(f"  Best lower bound    : {z_lag:.4f}  (z_lag)")
    print(f"  Master LP bound     : {z_master:.4f}  (z_master)")
    print(f"  Upper bound (z_UB)  : {z_ub:.4f}")
    print(f"  Final duality gap   : {final_gap:.2f}%  (vs z_UB)")
    print(f"{'═' * 62}\n")

    return {
        "best_lower_bound": z_lag,
        "upper_bound":      z_ub,
        "final_gap_pct":    final_gap,
        "multipliers":      u,
        "history":          pd.DataFrame(history),
        "num_iterations":   k,
    }
