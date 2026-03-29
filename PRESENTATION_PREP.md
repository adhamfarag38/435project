# MSE 435 Presentation Prep — Everything You Need to Know

---

## THE PROBLEM

The clinic has 16 examination rooms shared by up to 23 providers per day. Every appointment has a fixed start time, a fixed duration, and a fixed provider. The goal is to assign each appointment to a room so that:
- No two appointments are in the same room at the same time
- Total provider travel distance between rooms is minimized
- As many appointments as possible get a room (coverage)

This is hard because there are more providers than rooms on busy days, so not everyone can have a dedicated room. Providers have to share, which means they might move between rooms across the day (switching), and that travel has a cost.

---

## THE DATA

- Week 1: 501 total appointments → 427 active (74 deleted/cancelled)
- Week 2: 574 total appointments → 499 active (75 deleted/cancelled)
- 20 providers in Week 1 (HPW101–HPW120), 23 in Week 2 (HPW201–HPW224)
- 16 exam rooms (ER1–ER16)
- Appointment durations: 5 to 120 minutes, most common is 30 min (35%) then 15 min (23%)
- Day is split into 5-minute time slots
- No-show rate: ~7–8% of active appointments
- Week 1 Tuesday had only 4 appointments (1 provider) — it was Remembrance Day, a statutory holiday
- Room distances range from 0.5m (ER15–ER16, adjacent) to 13.0m (ER13–ER16, far end of clinic)
- On the busiest day (Week 2 Tuesday), 21 providers share 16 rooms — a one-room-per-provider policy is mathematically impossible

---

## THE THREE MODELS

### Model 1 — Feasibility Check
- Decision variables: x_ar (appointment a assigned to room r), y_r (room r is used)
- Objective: minimize number of rooms opened
- Constraints: each appointment gets exactly one room; no two overlapping appointments share a room; y_r = 1 if any appointment uses room r
- Purpose: just checks whether a feasible assignment exists at all. It is not the main scheduling model.
- Result: confirms a feasible packing exists for both weeks

### Model 2 — Provider-Day Schedule Generator
- Runs independently for each (provider, day) pair
- Decision variables: x_ar (appointment a in room r), w_abrr' (transition indicator — consecutive appointments a and b go from room r to r')
- Objective: minimize gamma × switches + eta × travel distance
  - gamma = 10 (penalty per room switch)
  - eta = 1 (penalty per metre of travel)
- Constraints: each appointment assigned to exactly one room in the provider's cluster; no two overlapping appointments in the same room; the w transition variables are linearized products of x variables (three constraints per consecutive pair)
- The bilinear term x_ar × x_br' cannot go directly into an LP — so it is linearized: w_abrr' ≤ x_ar, w_abrr' ≤ x_br', w_abrr' ≥ x_ar + x_br' − 1
- Output: one candidate schedule per (provider, day) — an assignment of each appointment to a room

### Model 3 — Master Schedule Selection
- Takes the pool of candidate schedules from Model 2 and selects the best combination
- Decision variables: lambda_s (binary — schedule s is selected), sigma_rt (slack — room r is double-booked at minute t)
- Objective: minimize sum of selected schedule costs + M × sum of room-time conflicts, where M = 10,000
- Constraints: each appointment covered by exactly one selected schedule; at most one provider per room per minute (soft, with slack sigma_rt); exactly one schedule selected per (provider, day)
- The sigma_rt slack is what makes the model always feasible — it absorbs unavoidable cross-provider room conflicts and penalizes them heavily
- Output: a globally optimal combination of provider-day schedules

---

## COLUMN GENERATION (DANTZIG-WOLFE DECOMPOSITION)

### Why it is needed
Model 3 selects from the set of all possible feasible schedules for each provider-day. There are exponentially many possible schedules (every possible room assignment sequence). You cannot enumerate them all upfront. Column generation builds the pool incrementally.

### How it works
1. Start with one schedule per (provider, day) — generated greedily by Model 2
2. Solve the LP relaxation of Model 3 (lambda can be fractional between 0 and 1)
3. Extract dual variables: pi_a (shadow price of appointment coverage) and mu_rt (shadow price of room capacity at minute t)
4. For each (provider, day), solve a modified Model 2 (the pricing subproblem): subtract dual savings from the objective
5. If the reduced cost of the new schedule is negative, it can improve the master LP — add it to the pool
6. Repeat until no schedule with negative reduced cost exists → LP optimality
7. Solve the final ILP on the complete pool

### Reduced cost formula
c_bar_s = c_s − sum(pi_a for a in schedule) − sum(mu_rt for (r,t) in schedule)
If c_bar_s < 0, add the schedule to the pool.

### What the bounds mean
- z_LP: the LP relaxation value from the restricted master problem. This is a valid lower bound ONLY if CG fully converged (no improving columns exist). If CG stopped early due to numerical stagnation, z_LP may be too high and is not a certified lower bound.
- z_ILP: the integer solution from the restricted pool — always a valid upper bound (feasible solution)
- Our CG terminates if the LP bound doesn't improve by more than 1e-6 between iterations — this is a stagnation stop, not a proven optimality stop. So our z_LP should be treated as an upper bound estimate, not a certified lower bound.

### The critical error flagged by the professor
We reported z_LP ≤ z_TS ≤ z_ILP as the bound hierarchy, but z_TS = 28.9M < z_LP = z_ILP = 36.8M. This contradicts the claim that z_LP is a lower bound. The correct statement is: the Tabu Search found a better feasible solution than the CG integer solution, which is possible because the CG column pool was incomplete. The correct hierarchy is z_LR ≤ z* ≤ z_TS ≤ z_ILP, where z_LP is not a certified lower bound in our implementation.

---

## LAGRANGIAN RELAXATION

### What it does
Takes the appointment coverage constraints (each appointment must be covered by exactly one schedule) and moves them into the objective with a penalty multiplier u_a ≥ 0. This gives a relaxed problem that is easier to solve and whose optimal value is a lower bound on z*.

### The Lagrangian function
L(u) = sum(u_a) + min over lambda of [sum((c_s − sum(u_a * alpha_as)) * lambda_s) + M × sum(sigma_rt)]
subject only to room capacity and one-schedule-per-day constraints.

### Why it gives a lower bound
Relaxing a constraint makes the feasible set larger. A larger feasible set can only decrease or maintain the minimum. So L(u) ≤ z* for any u ≥ 0.

### The Lagrangian dual
z_LD = max over u≥0 of L(u). This is the tightest possible lower bound from this relaxation.

### How we solve it — cutting plane method
We maintain a master LP: min sum(u_a) + theta, subject to theta ≥ sum(cov_a * u_a) + SP_obj for each iteration's subproblem. At each iteration we add a new cut. The algorithm terminates when z_lag − z_master < 1e-4.

### Results
z_LR = 36,800,551 (Week 1) = 43,000,706 (Week 2). These are the certified lower bounds.

---

## TABU SEARCH

### Purpose
Post-optimisation: start from the CG ILP solution and try to find a better feasible schedule by exploring room swaps that the ILP solver might have missed.

### How it works
- Solution representation: a flat dictionary {appointment_id → room_id}
- Neighbourhood: all single-appointment room swaps (move one appointment to a different room in its provider's cluster)
- At each iteration: evaluate all possible swaps, pick the best non-tabu one
- Tabu list: a FIFO queue of size k_max = 20 storing (appointment, old_room) pairs — prevents immediately undoing recent moves
- Aspiration criterion: override the tabu list if a move produces a solution better than the global best ever seen
- Stops after 300 iterations or 75 consecutive iterations without improvement

### Objective function
f(x) = gamma × switches + eta × travel + P × room-minute conflicts
where P = 10,000 (same as M in Model 3)

### Incremental updates
Recomputing the full objective from scratch for every candidate move would be O(appointments²). Instead, only the two transitions adjacent to the moved appointment change (for switch/travel), and only time-overlapping appointments matter for conflict deltas.

### Results
z_TS = 28,901,341 (Week 1), 35,301,772 (Week 2) — 21–19% improvement over z_ILP

### Why z_TS < z_ILP is valid
The TS operates on individual appointment-room assignments, not restricted to the column pool. It can find combinations that no single schedule in the pool achieves. This is why it beats z_ILP — the column pool was incomplete.

---

## THE SIX POLICIES

### Policy A — Single Room (Baseline)
Each provider is fixed to their pre-assigned home room for the entire day. No optimisation. Mirrors current clinical practice.
- Coverage: 85.7% (W1), 87.8% (W2)
- Why less than 100%: on some days, multiple providers share the same home room and there is no way to avoid conflicts without moving someone

### Policy B — Cluster of Nearby Rooms
Providers can use any room within 4 metres of their home room. Model 2 minimises switches and travel within this cluster. Model 3 resolves cross-provider conflicts globally.
- Coverage: 96.5% (W1), 94.4% (W2)
- The 4m threshold was chosen based on the room proximity matrix — it allows 2–4 additional rooms per provider without requiring large travel distances

### Policy C — Robust Duration Buffer
Same as B, but adds a 10% buffer to each appointment duration when checking for conflicts. A 30-minute appointment is treated as 33 minutes for overlap detection.
- Coverage: 84.1% (W1), 74.5% (W2)
- Performs worst because the buffer makes the schedule more conservative — rooms appear "busy" longer, so fewer appointments can be packed in, and this costs more than it saves from conflict avoidance

### Policy D — No-Show Overbooking
Adjusts effective durations using historical no-show rates: d_hat = (1 − rho) × d, where rho is the provider's no-show probability (~7–8%). Shorter effective durations allow more appointments to be squeezed in.
- Coverage: 91.1% (W1), 86.6% (W2)
- Lower than B because the duration adjustments sometimes backfire — when a patient does show up, a short slot can create a real conflict

### Policy E — Day Blocking
Providers marked as unavailable on a given day have all their appointments removed before scheduling. Respects day-off schedules.
- Coverage: 84.3% (W1), 82.2% (W2)
- The blocked appointments are counted as unscheduled against the total, which drags coverage down

### Policy F — Admin Time Buffer (Recommended)
Admin blocks (morning huddle 9:00–9:30, noon prep 11:30–12:00, lunch 12:00–13:00, afternoon close 16:30–17:00) are treated as flexible overflow. If an appointment straddles the start of an admin block, its effective end time is clipped to the block boundary. This reduces apparent overlap and gives the scheduler more room flexibility.
- Coverage: 96.5% (W1), 97.2% (W2)
- Wins over B because it achieves equal or better coverage with fewer room switches — the admin buffer absorbs overruns that would otherwise force a provider to move rooms

### Why F beats B even at equal coverage
Coverage is the same in Week 1 (both 96.5%) but F has lower average switches (0.79 vs higher for B). The 15 unscheduled appointments in Week 1 cannot be helped by admin buffering — they are infeasible because too many providers compete for the same rooms during peak hours. The improvement from F is in schedule quality, not capacity.

---

## PROXIMITY THRESHOLD — WHY 4 METRES

The room distance matrix shows distances ranging from 0.5m to 13.0m. A 4m threshold includes 2–5 additional rooms for most providers (ER rooms near each other in a cluster). Going wider (e.g., 8m) would allow more room flexibility but increase travel cost significantly. 4m was chosen as the operational sweet spot — providers can move between nearby rooms without meaningfully disrupting their day.

---

## KEY NUMBERS TO MEMORISE

- 16 rooms, up to 23 providers per day
- 427 active appointments Week 1, 499 Week 2
- No-show rate ~7–8%, cancellation rate ~14–15%
- gamma = 10 (switch penalty), eta = 1 (travel penalty per metre), M = 10,000 (room conflict penalty)
- z_LR = 36.8M (W1) — certified lower bound
- z_ILP = 36.8M (W1) — CG integer solution (upper bound)
- z_TS = 28.9M (W1) — Tabu Search solution (better upper bound, 21% improvement)
- Policy F coverage: 96.5% (W1), 97.2% (W2)
- Policy A coverage: 85.7% (W1), 87.8% (W2) — the baseline you beat
- Average switches per provider-day under Policy F: 0.79 (W1), 0.91 (W2)
- Total travel under Policy F: 50.2m (W1), 54.4m (W2)

---

## QUESTIONS YOU MIGHT BE ASKED

### On the problem
Q: Why not just assign each provider a permanent room for the whole week?
A: On busy days, especially Week 2 Tuesday (21 providers, 16 rooms), there are more providers than rooms. A permanent assignment is mathematically infeasible. Even on less busy days, a shared-room approach reduces idle room time.

Q: Why minimise travel distance rather than number of switches?
A: We actually minimise both — gamma × switches + eta × travel. Switches are weighted at 10 and travel at 1 per metre because a room switch has a fixed cognitive and time cost beyond just the walking distance. These weights can be tuned.

Q: What happens to the 15 unscheduled appointments under Policy F?
A: They belong to provider-days where Model 2 returned infeasible (HPW104 Wednesday and HPW115 Thursday are the known cases). The cluster rooms available to those providers were fully occupied by overlapping appointments from other providers. Short of assigning them a room outside the 4m cluster (which we could try with a larger threshold), they cannot be scheduled.

### On the models
Q: Why do you need three models? Why not solve it all at once?
A: One big model would have millions of variables (every possible room assignment for every appointment across all providers and days simultaneously). The decomposition makes it tractable. Model 2 handles the within-provider complexity, Model 3 handles the cross-provider conflicts, and column generation links them.

Q: Why is the linearisation of w needed in Model 2?
A: The switch cost depends on whether consecutive appointments a and b are in different rooms — that is, x_ar × x_br'. A product of two binary variables is bilinear and makes the problem non-linear (non-convex). The three linearisation constraints replace the product with a new binary variable w_abrr' that equals 1 if and only if both x_ar = 1 and x_br' = 1.

Q: What is the integrality gap?
A: The gap between the LP relaxation (z_LP) and the integer solution (z_ILP). A gap of 0 means the LP solution is already integer — the relaxation was tight. In our results z_LP = z_ILP, which means no integrality gap within the restricted column pool.

Q: Why does Model 3 use a soft room capacity constraint instead of a hard one?
A: A hard constraint would make the model infeasible on days where no combination of the generated schedules can avoid all cross-provider room conflicts. The soft constraint with sigma_rt slack ensures the model always finds a solution and instead penalises conflicts heavily (M = 10,000). In a full column generation run with enough columns, sigma_rt should be driven to zero.

### On column generation
Q: How do you know when column generation has converged?
A: Properly, when no pricing subproblem returns a schedule with negative reduced cost — meaning no schedule outside the current pool can improve the LP objective. In our implementation we also stop if the LP bound changes by less than 1e-6 between iterations (stagnation stop), which is a practical but not theoretically certified stopping criterion.

Q: What is a dual variable and why does the pricing subproblem use them?
A: The dual variable pi_a for an appointment coverage constraint tells you how much the LP objective would decrease if you relaxed that constraint by one unit — i.e., how much the LP "values" covering that appointment. The pricing subproblem subtracts these values from the schedule cost to compute reduced cost. If a schedule's reduced cost is negative, adding it to the pool can decrease the master LP objective.

Q: Why does the restricted LP give an upper bound on the true LP optimal (not a lower bound)?
A: Adding more schedules to the pool gives the LP more choices, which can only decrease or maintain the minimum. So more columns → lower LP objective. The restricted LP (fewer columns) therefore gives a value ≥ the true LP optimal. The restricted LP is an upper bound on the true LP relaxation value, not a lower bound.

### On Lagrangian relaxation
Q: Why does dualising the coverage constraints give a lower bound?
A: When you move a constraint into the objective with a penalty, you are relaxing it — the feasible set gets larger (you can now violate the constraint if the penalty is worth it). A larger feasible set means the minimum can only go down. So the relaxed objective ≤ the true objective for any feasible solution → lower bound.

Q: Why is z_LR ≤ z_LP?
A: Both are lower bounds on z*, but the LP bound is tighter (closer to z*) because the LP relaxation is a direct relaxation of the ILP, while Lagrangian only dualises the coverage constraints and keeps the rest of the structure. In our results they happen to be equal (both 36.8M), which suggests the Lagrangian dual is tight.

Q: What is the cutting plane method used for?
A: To solve the Lagrangian dual (maximise L(u) over u ≥ 0). At each iteration we solve the subproblem, add a linearisation cut to a master LP, and re-solve. This progressively tightens the bound on the optimal multipliers u.

### On Tabu Search
Q: Why use Tabu Search after already having an ILP solution?
A: The ILP was solved on a restricted column pool. The pool may not contain the globally optimal schedule for every provider-day. Tabu Search bypasses this restriction by working directly on appointment-room assignments and can find better solutions that no combination of pooled schedules achieves.

Q: Why can the Tabu Search beat z_ILP if z_ILP is supposed to be optimal?
A: z_ILP is optimal only within the restricted column pool. The true optimal might be lower. The TS explores a richer space not constrained by the pool.

Q: What is the aspiration criterion?
A: An override rule: if a move is on the tabu list (normally forbidden), it is allowed anyway if it would produce a solution better than the best solution found so far. This prevents the tabu list from blocking a genuinely good move.

Q: What is the tabu list preventing exactly?
A: Cycling — repeatedly undoing and redoing the same move. After moving appointment A from room R1 to R2, the pair (A, R1) goes on the tabu list. This prevents immediately moving A back to R1, which would undo all progress.

### On results and policies
Q: Why does Policy C perform so poorly?
A: The 10% duration buffer makes each appointment appear longer to the conflict detection. This reduces effective room availability more than it gains from conflict avoidance — there are fewer opportunities to pack appointments without conflicts, so more appointments go unscheduled.

Q: Why do Policies B and F have the same coverage?
A: The 15 unscheduled appointments are infeasible because of room availability during peak hours, not because of timing conflicts near admin block boundaries. Policy F's admin buffer resolves the latter but not the former. The improvement from F shows up in switch count and travel distance, not in coverage.

Q: Is Policy F operationally realistic?
A: Yes. Admin huddle and prep time are already built into the clinic schedule as flexible time. Using them as overflow buffers does not require changing clinical practice — it just means that if an appointment runs slightly long into admin time, it is absorbed rather than creating a cascade of room conflicts.

Q: What is room utilisation and why is it only 22–24%?
A: Room utilisation is the percentage of total available room-minutes that are actually occupied by appointments. 16 rooms × 480 min/day × 5 days = 38,400 room-minutes available per week. With ~427 appointments averaging ~30 min each, that is only ~12,810 appointment-minutes, giving roughly 33% theoretical maximum. After accounting for scheduling constraints and idle time between appointments, 22–24% is realistic and consistent with typical outpatient clinic utilisation.

Q: Could you increase coverage beyond 96.5%?
A: Yes, a few ways: (1) allow rooms outside the 4m proximity cluster in extreme cases, (2) run more column generation iterations to find better schedules, (3) apply overbooking more carefully than Policy D does, or (4) request the clinic restructure its room assignments (give high-volume providers like HPW114 a room cluster with fewer shared-room conflicts).

### On the bound hierarchy (the critical error)
Q: Your professor said z_TS < z_LP is an error — how do you respond?
A: The professor is correct to flag it. We incorrectly labelled z_LP as a lower bound. In our implementation, z_LP comes from the restricted master LP which may not have fully converged — it is an upper bound estimate on the true LP optimal, not a certified lower bound. The only certified lower bound in our results is z_LR from the Lagrangian relaxation. The Tabu Search legitimately found a better feasible solution (z_TS = 28.9M) than the restricted-pool ILP (z_ILP = 36.8M), which is valid. The corrected hierarchy is z_LR ≤ z* ≤ z_TS ≤ z_ILP.

---

## CODE STRUCTURE (IN CASE THEY ASK)

- data_loader.py — loads and cleans appointment/provider/distance data, defines admin blocks and room lists
- model1.py — feasibility packing (PuLP ILP)
- model2.py — provider-day schedule generation (PuLP ILP), also has sequential greedy version
- model3.py — master schedule selection (PuLP ILP), extracts dual variables for LP case
- column_generation.py — full Dantzig-Wolfe loop, calls model2 pricing subproblem
- lagrangian.py — cutting plane method for Lagrangian dual
- tabu_search.py — post-optimisation metaheuristic
- main.py — orchestrates all six policies and the advanced CG→LR→TS pipeline
- visualization.py — Gantt charts and KPI bar charts
- generate_policy_f_schedule.py — standalone script to regenerate Policy F charts and CSVs

All ILP models use PuLP with the CBC solver. Time limits: 60 seconds per Model 2 subproblem, 120 seconds for Model 3.
