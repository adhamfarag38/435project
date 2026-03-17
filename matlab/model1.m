function result = model1(appointments, rooms, delta_frac)
% MODEL1  Feasibility Packing Model.
%
% result = model1(appointments, rooms, delta_frac)
%
% Assigns appointments to examination rooms while:
%   - Respecting time conflicts (robust duration buffers applied)
%   - Minimising number of distinct rooms used  (improved objective vs original)
%
% Inputs:
%   appointments - table with start_min, duration_min, end_min, appt_id
%   rooms        - cell array of room IDs e.g. {'ER1','ER2',...,'ER16'}
%   delta_frac   - robust buffer fraction (default 0.1)
%
% Output:
%   result - struct with fields:
%     .feasible      - logical
%     .assignment    - n×1 cell array of room IDs (matched to appointments rows)
%     .rooms_used    - cell array of rooms actually used
%     .obj_value     - number of rooms used
%     .status        - solver exit flag (1 = optimal)

if nargin < 2 || isempty(rooms)
    rooms = {'ER1','ER2','ER3','ER4','ER5','ER6','ER7','ER8',...
             'ER9','ER10','ER11','ER12','ER13','ER14','ER15','ER16'};
end
if nargin < 3, delta_frac = 0.1; end

nA = height(appointments);
nR = numel(rooms);

if nA == 0
    result.feasible   = true;
    result.assignment = {};
    result.rooms_used = {};
    result.obj_value  = 0;
    result.status     = 1;
    return;
end

%% ── Robust overlap detection ─────────────────────────────────────────────────
delta = round(delta_frac * appointments.duration_min);  % nA×1
overlap = false(nA, nA);
for i = 1:nA
    for j = i+1:nA
        a_end = appointments.end_min(i) + delta(i);
        b_end = appointments.end_min(j) + delta(j);
        if appointments.start_min(i) < b_end && appointments.start_min(j) < a_end
            overlap(i,j) = true;
            overlap(j,i) = true;
        end
    end
end

%% ── Variable layout ─────────────────────────────────────────────────────────
% Variables: [x(1,1),...,x(nA,nR), y(1),...,y(nR)]
% x_ar at index (a-1)*nR + r
% y_r  at index nA*nR + r

nVars = nA*nR + nR;
intcon = 1:nVars;  % all binary

f = [zeros(nA*nR, 1); ones(nR, 1)];  % minimise sum y_r

%% ── Equality constraints: sum_r x_ar = 1 for each a ─────────────────────────
Aeq = zeros(nA, nVars);
beq = ones(nA, 1);
for a = 1:nA
    for r = 1:nR
        Aeq(a, (a-1)*nR + r) = 1;
    end
end

%% ── Inequality constraints ───────────────────────────────────────────────────
A_ineq = [];
b_ineq = [];

% Conflict: x_ar + x_br <= 1 for overlapping pairs (a,b), for each room r
for i = 1:nA
    for j = i+1:nA
        if overlap(i,j)
            for r = 1:nR
                row = zeros(1, nVars);
                row((i-1)*nR + r) = 1;
                row((j-1)*nR + r) = 1;
                A_ineq(end+1,:) = row; %#ok<AGROW>
                b_ineq(end+1)   = 1;  %#ok<AGROW>
            end
        end
    end
end

% Link y_r >= x_ar: x_ar - y_r <= 0
for a = 1:nA
    for r = 1:nR
        row = zeros(1, nVars);
        row((a-1)*nR + r) = 1;
        row(nA*nR + r)     = -1;
        A_ineq(end+1,:) = row; %#ok<AGROW>
        b_ineq(end+1)   = 0;  %#ok<AGROW>
    end
end

b_ineq = b_ineq(:);

lb = zeros(nVars, 1);
ub = ones(nVars, 1);

%% ── Solve with intlinprog ────────────────────────────────────────────────────
opts = optimoptions('intlinprog', 'Display', 'off');
[x_sol, fval, exitflag] = intlinprog(f, intcon, A_ineq, b_ineq, Aeq, beq, lb, ub, opts);

result.status   = exitflag;
result.feasible = (exitflag == 1);
result.assignment = cell(nA, 1);
result.rooms_used = {};
result.obj_value  = NaN;

if result.feasible
    result.obj_value = fval;
    x_mat = reshape(x_sol(1:nA*nR), nR, nA)';  % nA x nR
    for a = 1:nA
        [~, r_idx] = max(x_mat(a,:));
        result.assignment{a} = rooms{r_idx};
    end
    y_vals = x_sol(nA*nR+1:end);
    result.rooms_used = rooms(y_vals > 0.5);
end

end
