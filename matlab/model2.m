function result = model2(appointments, rooms, dist_matrix, delta_frac, gamma, eta)
% MODEL2  Provider-Day Schedule Generation Model.
%
% result = model2(appointments, rooms, dist_matrix, delta_frac, gamma, eta)
%
% For a given (provider, day) set of appointments, assigns each to a room
% from the allowed cluster, minimising weighted room switches and travel.
%
% Inputs:
%   appointments - table sorted by start_min (for one provider-day)
%   rooms        - cell array of allowed room IDs (provider cluster)
%   dist_matrix  - 16×16 distance matrix; rows/cols correspond to ER1..ER16
%   delta_frac   - robust duration buffer fraction (default 0.1)
%   gamma        - room-switch penalty (default 10)
%   eta          - travel distance weight (default 1)
%
% Output:  struct with fields:
%   .feasible, .assignment, .num_switches, .total_travel, .cost
%   .alpha (appt_id → 1 coverage map as containers.Map)
%   .beta  (nA×1 cell of {room, start_min, end_min} used time-slots)
%   .status

if nargin < 4, delta_frac = 0.1; end
if nargin < 5, gamma = 10; end
if nargin < 6, eta = 1; end

% Sort by start time
appointments = sortrows(appointments, 'start_min');
nA = height(appointments);
nR = numel(rooms);

ALL_ROOMS = {'ER1','ER2','ER3','ER4','ER5','ER6','ER7','ER8',...
             'ER9','ER10','ER11','ER12','ER13','ER14','ER15','ER16'};

% Map room IDs to distance matrix indices
room_idx = zeros(nR,1);
for r = 1:nR
    idx = find(strcmp(ALL_ROOMS, rooms{r}), 1);
    room_idx(r) = idx;
end

if nA == 0
    result = empty_result(appointments);
    return;
end

%% ── Overlap detection ────────────────────────────────────────────────────────
delta = round(delta_frac * appointments.duration_min);
overlap = false(nA, nA);
for i = 1:nA
    for j = i+1:nA
        if appointments.start_min(i) < appointments.end_min(j) + delta(j) && ...
           appointments.start_min(j) < appointments.end_min(i) + delta(i)
            overlap(i,j) = true; overlap(j,i) = true;
        end
    end
end

%% ── Variable layout ─────────────────────────────────────────────────────────
% x_ar : appointment a in room r  — index (a-1)*nR + r     [1 .. nA*nR]
% w_ijrr2: consecutive pair i→j in rooms r→r2              [nA*nR+1 .. nA*nR + (nA-1)*nR^2]
%
% Consecutive pairs: (1,2),(2,3),...,(nA-1,nA)
nCons  = nA - 1;
nX     = nA * nR;
nW     = nCons * nR * nR;
nVars  = nX + nW;

intcon = 1:nVars;
lb = zeros(nVars, 1);
ub = ones(nVars, 1);

% Helper: x index for (a,r) : 1-based
x_idx = @(a,r) (a-1)*nR + r;
% Helper: w index for (cons_pair k, r, r2) : 1-based, k in 1..nCons
w_idx = @(k,r,r2) nX + (k-1)*nR*nR + (r-1)*nR + r2;

%% ── Objective ────────────────────────────────────────────────────────────────
f = zeros(nVars, 1);
for k = 1:nCons
    for r = 1:nR
        for r2 = 1:nR
            if r ~= r2
                ri = room_idx(r); r2i = room_idx(r2);
                dist_val = 0;
                if ri > 0 && r2i > 0
                    dist_val = dist_matrix(ri, r2i);
                end
                f(w_idx(k,r,r2)) = gamma + eta * dist_val;
            end
        end
    end
end

%% ── Constraints ─────────────────────────────────────────────────────────────
A_ineq = [];
b_ineq = [];
Aeq    = [];
beq    = [];

% 1. Assignment: sum_r x_ar = 1
for a = 1:nA
    row = zeros(1, nVars);
    for r = 1:nR, row(x_idx(a,r)) = 1; end
    Aeq(end+1,:) = row; %#ok<AGROW>
    beq(end+1)   = 1;   %#ok<AGROW>
end

% 2. Conflict: x_ar + x_br <= 1 for overlapping pairs
for i = 1:nA
    for j = i+1:nA
        if overlap(i,j)
            for r = 1:nR
                row = zeros(1, nVars);
                row(x_idx(i,r)) = 1;
                row(x_idx(j,r)) = 1;
                A_ineq(end+1,:) = row; %#ok<AGROW>
                b_ineq(end+1)   = 1;  %#ok<AGROW>
            end
        end
    end
end

% 3. Linearise w_ijrr2 = x_ir * x_jr2 (consecutive pair k = i, j = i+1)
for k = 1:nCons
    i = k; j = k + 1;
    % w <= x_ir
    for r = 1:nR
        for r2 = 1:nR
            row = zeros(1, nVars);
            row(w_idx(k,r,r2)) =  1;
            row(x_idx(i,r))    = -1;
            A_ineq(end+1,:) = row; %#ok<AGROW>
            b_ineq(end+1)   = 0;  %#ok<AGROW>
        end
    end
    % w <= x_jr2
    for r = 1:nR
        for r2 = 1:nR
            row = zeros(1, nVars);
            row(w_idx(k,r,r2)) =  1;
            row(x_idx(j,r2))   = -1;
            A_ineq(end+1,:) = row; %#ok<AGROW>
            b_ineq(end+1)   = 0;  %#ok<AGROW>
        end
    end
    % w >= x_ir + x_jr2 - 1
    for r = 1:nR
        for r2 = 1:nR
            row = zeros(1, nVars);
            row(w_idx(k,r,r2)) = -1;
            row(x_idx(i,r))    =  1;
            row(x_idx(j,r2))   =  1;
            A_ineq(end+1,:) = row; %#ok<AGROW>
            b_ineq(end+1)   = 1;  %#ok<AGROW>
        end
    end
    % sum_{r,r2} w_kijrr2 = 1
    row = zeros(1, nVars);
    for r = 1:nR
        for r2 = 1:nR
            row(w_idx(k,r,r2)) = 1;
        end
    end
    Aeq(end+1,:) = row; %#ok<AGROW>
    beq(end+1)   = 1;   %#ok<AGROW>
end

b_ineq = b_ineq(:);
beq    = beq(:);

%% ── Solve ────────────────────────────────────────────────────────────────────
opts = optimoptions('intlinprog', 'Display', 'off', 'MaxTime', 60);
[x_sol, fval, exitflag] = intlinprog(f, intcon, A_ineq, b_ineq, Aeq, beq, lb, ub, opts);

result.status   = exitflag;
result.feasible = (exitflag == 1);
result.assignment  = cell(nA, 1);
result.num_switches = 0;
result.total_travel = 0;
result.cost         = inf;
result.alpha        = [];
result.beta         = {};

if result.feasible
    result.cost = fval;
    x_mat = reshape(x_sol(1:nX), nR, nA)';
    for a = 1:nA
        [~, r_idx_sol] = max(x_mat(a,:));
        result.assignment{a} = rooms{r_idx_sol};
    end

    % Count switches and travel
    for k = 1:nCons
        r1 = result.assignment{k};
        r2 = result.assignment{k+1};
        if ~strcmp(r1, r2)
            result.num_switches = result.num_switches + 1;
            ri1 = find(strcmp(ALL_ROOMS, r1),1);
            ri2 = find(strcmp(ALL_ROOMS, r2),1);
            if ~isempty(ri1) && ~isempty(ri2)
                result.total_travel = result.total_travel + dist_matrix(ri1, ri2);
            end
        end
    end

    % alpha: appointment IDs
    result.alpha = appointments.appt_id;

    % beta: {room, start_min, end_min} per appointment
    result.beta = cell(nA, 1);
    for a = 1:nA
        result.beta{a} = {result.assignment{a}, ...
                           appointments.start_min(a), appointments.end_min(a)};
    end
end

end  % model2


function r = empty_result(appts)
    r.status = 1; r.feasible = true;
    r.assignment = {}; r.num_switches = 0;
    r.total_travel = 0; r.cost = 0;
    r.alpha = appts.appt_id; r.beta = {};
end
