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

SUBGRADIENT METHOD
──────────────────
The function L(u) is concave and piecewise linear — no gradient exists, but
a SUBGRADIENT always does:

  g_a = 1 − Σ_s α_{as} λ_s^k      (coverage violation for appointment a)

  g_a > 0: appt a is under-covered  → increase u_a (penalise more)
  g_a < 0: appt a is over-covered   → decrease u_a (penalise less)
  g_a = 0: constraint satisfied

POLYAK STEP SIZE
────────────────
  t_k = λ_k · (z_UB − L(u^k)) / ‖g^k‖²

  z_UB : best known feasible upper bound (ILP solution from Column Generation)
  λ_k  : scalar ∈ (0, 2], halved every `patience` iterations without improvement

UPDATE RULE  (subgradient ASCENT for maximising L)
────────────────────────────────────────────────────
  u_a^{k+1} = max(0,  u_a^k + t_k · g_a^k)

  Note: + sign because we are MAXIMISING L(u) and g is the subgradient.

CONVERGENCE
───────────
Stop when any of:
  • ‖g^k‖ ≈ 0  (all coverage constraints satisfied → primal feasibility)
  • gap = (z_UB − L(u)) / z_UB < ε
  • iteration limit reached
  • λ_k falls below minimum threshold (numerical precision limit)
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


# ─── Subgradient Loop ─────────────────────────────────────────────────────────

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
    Subgradient method for the Lagrangian dual of Model 3.

    Parameters
    ----------
    z_ub          : best known upper bound (ILP solution from Column Generation)
    lambda_init   : initial Polyak scaling factor λ ∈ (0, 2]
    lambda_factor : multiply λ by this when no improvement (default 0.5)
    patience      : iterations without improvement before reducing λ
    gap_tol       : stop if (z_UB − L(u)) / z_UB < gap_tol

    Returns
    -------
    dict with:
        'best_lower_bound' : float       — best L(u) achieved
        'final_gap_pct'    : float       — (z_UB − best_L) / z_UB × 100
        'multipliers'      : dict        — final u_a values
        'history'          : pd.DataFrame — per-iteration log
        'num_iterations'   : int
    """
    print("\n" + "═" * 62)
    print("  LAGRANGIAN RELAXATION  (Subgradient Method)")
    print("═" * 62)
    print(f"  Upper bound z_UB : {z_ub:.4f}")
    print(f"  Max iterations   : {max_iterations}")
    print(f"  Initial λ        : {lambda_init}")
    print(f"  Patience         : {patience} iters before halving λ\n")

    all_appt_ids = list(appointments["appt_id"])

    # Initialise multipliers at 0
    u = {a: 0.0 for a in all_appt_ids}

    lam_k         = lambda_init
    best_L        = -float("inf")
    no_improve    = 0
    history       = []
    k             = 0

    for k in range(1, max_iterations + 1):

        # ── Solve Lagrangian subproblem ────────────────────────────────────────
        sub = solve_lagrangian_subproblem(
            schedules, appointments, u, verbose=False
        )
        if not sub["feasible"]:
            print(f"[LR] k={k}: subproblem infeasible — stopping.")
            break

        L_u         = sub["L_u"]
        lam_vals    = sub["lambda_values"]
        feas_scheds = sub["selected_scheds"]

        # ── Update best lower bound ────────────────────────────────────────────
        improved = L_u > best_L
        if improved:
            best_L     = L_u
            no_improve = 0
        else:
            no_improve += 1

        gap_pct = (z_ub - best_L) / abs(z_ub) * 100 if z_ub != 0 else float("inf")

        history.append({
            "iteration": k,
            "L_u":       L_u,
            "best_L":    best_L,
            "gap_pct":   gap_pct,
            "lambda":    lam_k,
            "improved":  improved,
        })

        # Print every 10 iterations (and first + last)
        if k == 1 or k % 10 == 0:
            print(f"[LR] k={k:4d} | L(u) = {L_u:10.4f} | "
                  f"best = {best_L:10.4f} | gap = {gap_pct:6.2f}% | λ = {lam_k:.5f}")

        # ── Convergence checks ─────────────────────────────────────────────────
        if gap_pct < gap_tol * 100:
            print(f"\n[LR] Gap {gap_pct:.4f}% below tolerance — converged.")
            break

        # ── Compute subgradient ────────────────────────────────────────────────
        # g_a = 1 − Σ_s α_{as} · λ_s
        # g_a > 0: appointment a under-covered → u_a should rise
        # g_a < 0: appointment a over-covered  → u_a should fall
        g = {}
        for a in all_appt_ids:
            covered = sum(
                lam_vals.get(s, 0.0)
                for s, sch in enumerate(feas_scheds)
                if a in sch["alpha"]
            )
            g[a] = 1.0 - covered

        g_vec   = np.array([g[a] for a in all_appt_ids])
        norm_sq = float(np.dot(g_vec, g_vec))

        # Converged if subgradient is zero (all coverage constraints satisfied)
        if norm_sq < 1e-10:
            print(f"\n[LR] Subgradient ‖g‖ ≈ 0 at k={k} — all coverage "
                  f"constraints satisfied.")
            break

        # ── Polyak step size ───────────────────────────────────────────────────
        # t_k = λ_k · (z_UB − L(u^k)) / ‖g^k‖²
        numerator = z_ub - L_u
        t_k = lam_k * max(numerator, 0.0) / norm_sq

        # ── Multiplier update (subgradient ASCENT) ─────────────────────────────
        # u_a^{k+1} = max(0,  u_a^k + t_k · g_a^k)
        # We ADD because we are MAXIMISING L(u) and g is the subgradient of L.
        for a in all_appt_ids:
            u[a] = max(0.0, u[a] + t_k * g[a])

        # ── Reduce λ if no improvement for `patience` iterations ───────────────
        if no_improve >= patience:
            lam_k      *= lambda_factor
            no_improve  = 0
            if verbose:
                print(f"[LR]   No improvement for {patience} iters "
                      f"→ λ reduced to {lam_k:.6f}")
            if lam_k < 1e-8:
                print("[LR] λ too small — stopping (numerical limit).")
                break

    final_gap = (z_ub - best_L) / abs(z_ub) * 100 if z_ub != 0 else float("inf")

    print(f"\n{'═' * 62}")
    print(f"  LAGRANGIAN RELAXATION RESULTS")
    print(f"{'─' * 62}")
    print(f"  Iterations run      : {k}")
    print(f"  Best lower bound    : {best_L:.4f}")
    print(f"  Upper bound (z_UB)  : {z_ub:.4f}")
    print(f"  Final duality gap   : {final_gap:.2f}%")
    print(f"{'═' * 62}\n")

    return {
        "best_lower_bound": best_L,
        "upper_bound":      z_ub,
        "final_gap_pct":    final_gap,
        "multipliers":      u,
        "history":          pd.DataFrame(history),
        "num_iterations":   k,
    }
