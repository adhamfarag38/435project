function result = model3(schedules, all_appt_ids, integer_solve)
% MODEL3  Master Schedule Selection Model.
%
% result = model3(schedules, all_appt_ids, integer_solve)
%
% Selects the optimal combination of provider-day schedules:
%   min  sum_s  c_s * lambda_s
%   s.t. sum_{s: alpha_as=1} lambda_s = 1    for all a in A   (coverage)
%        sum_{s: beta_rts=1} lambda_s <= 1   for all r,t      (room capacity)
%        sum_{s in S(p,d)}   lambda_s = 1    for all p,d      (one-per-day)
%        lambda_s in {0,1}
%
% Inputs:
%   schedules      - struct array from model2 runs (see model2.m output)
%                    each element also has fields: provider, day, week
%   all_appt_ids   - vector of all appointment IDs that must be covered
%   integer_solve  - true for ILP (default), false for LP relaxation
%
% Output: struct with .feasible, .selected_idx, .total_cost, .status

if nargin < 3, integer_solve = true; end

% Filter to feasible schedules
feasible_idx = find([schedules.feasible]);
if isempty(feasible_idx)
    result.feasible = false;
    result.selected_idx = [];
    result.total_cost = inf;
    result.status = -1;
    fprintf('[Model3] No feasible schedules available.\n');
    return;
end

S = length(feasible_idx);
fprintf('[Model3] Using %d feasible schedules.\n', S);

%% ── Build appointment coverage map ──────────────────────────────────────────
% all_appt_ids that appear in at least one schedule
covered_appts = [];
for s = 1:S
    sch = schedules(feasible_idx(s));
    covered_appts = union(covered_appts, sch.alpha(:)');
end
A_cover = intersect(all_appt_ids(:)', covered_appts);
nA = length(A_cover);
appt_map = containers.Map(num2cell(A_cover), num2cell(1:nA));

%% ── Provider-day groups ──────────────────────────────────────────────────────
pd_keys = {};
pd_members = {};
for s = 1:S
    sch = schedules(feasible_idx(s));
    key = sprintf('%s_%s_W%d', sch.provider, sch.day, sch.week);
    found = false;
    for k = 1:numel(pd_keys)
        if strcmp(pd_keys{k}, key)
            pd_members{k}(end+1) = s;
            found = true; break;
        end
    end
    if ~found
        pd_keys{end+1} = key; %#ok<AGROW>
        pd_members{end+1} = s; %#ok<AGROW>
    end
end
nPD = numel(pd_keys);

%% ── Collect room-time conflicts across schedules ─────────────────────────────
% beta_rt_s: for each schedule s, which (room, t) slots it uses
% Represent as a list of (s, room_str, t) triples
rt_map = containers.Map('KeyType','char','ValueType','any');
for s = 1:S
    sch = schedules(feasible_idx(s));
    for ai = 1:numel(sch.beta)
        b = sch.beta{ai};
        if isempty(b), continue; end
        room = b{1}; t_start = b{2}; t_end = b{3};
        for t = t_start:t_end-1
            key = sprintf('%s_%d', room, t);
            if isKey(rt_map, key)
                rt_map(key) = [rt_map(key), s];
            else
                rt_map(key) = s;
            end
        end
    end
end
rt_keys = keys(rt_map);
% Only keep slots used by >1 schedule (capacity matters only then)
conflict_rt = {};
for k = 1:numel(rt_keys)
    if numel(rt_map(rt_keys{k})) > 1
        conflict_rt{end+1} = rt_keys{k}; %#ok<AGROW>
    end
end
nRT = numel(conflict_rt);

%% ── Variable: lambda_s (one per feasible schedule) ───────────────────────────
nVars = S;
f = zeros(S, 1);
for s = 1:S
    f(s) = schedules(feasible_idx(s)).cost;
end

if integer_solve
    intcon = 1:S;
else
    intcon = [];
end
lb = zeros(S,1);
ub = ones(S,1);

%% ── Equality constraints ─────────────────────────────────────────────────────
Aeq = zeros(nA + nPD, S);
beq = ones(nA + nPD, 1);

% Appointment coverage
for a_idx = 1:nA
    a = A_cover(a_idx);
    for s = 1:S
        sch = schedules(feasible_idx(s));
        if ismember(a, sch.alpha)
            Aeq(a_idx, s) = 1;
        end
    end
end

% One schedule per provider-day
for k = 1:nPD
    for s = pd_members{k}
        Aeq(nA + k, s) = 1;
    end
end

%% ── Inequality constraints: room capacity ─────────────────────────────────────
A_ineq = zeros(nRT, S);
b_ineq = ones(nRT, 1);

for ci = 1:nRT
    s_list = rt_map(conflict_rt{ci});
    for s = s_list
        A_ineq(ci, s) = 1;
    end
end

%% ── Solve ────────────────────────────────────────────────────────────────────
opts = optimoptions('intlinprog', 'Display', 'off', 'MaxTime', 120);
[x_sol, fval, exitflag] = intlinprog(f, intcon, A_ineq, b_ineq, Aeq, beq, lb, ub, opts);

result.status   = exitflag;
result.feasible = (exitflag == 1);
result.selected_idx  = [];
result.total_cost    = inf;

if result.feasible
    result.total_cost = fval;
    selected_local = find(x_sol > 0.5);
    result.selected_idx = feasible_idx(selected_local);
    fprintf('[Model3] Optimal cost=%.2f, selected %d schedules.\n', ...
            fval, numel(result.selected_idx));
else
    fprintf('[Model3] Solver exit flag=%d (not optimal).\n', exitflag);
end

end
