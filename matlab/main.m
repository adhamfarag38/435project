%% main.m
% Examination Room Scheduling Optimization Engine — MATLAB
% ─────────────────────────────────────────────────────────
% Runs Models 1, 2, and 3 for Week 1 and Week 2, compares policies,
% and generates Gantt charts + KPI tables.

clear; clc; close all;

addpath(fileparts(mfilename('fullpath')));  % ensure this folder is on path

GAMMA = 10;     % room-switch penalty
ETA   = 1;      % travel-distance weight
DELTA_FRAC = 0.10;  % 10% robust duration buffer
PROXIMITY_THRESHOLD = 4.0;  % metres

WEEK = 1;   % Change to 2 for Week 2

fprintf('=======================================================\n');
fprintf(' Examination Room Scheduling — Week %d\n', WEEK);
fprintf('=======================================================\n\n');

%% ── 1. Load data ─────────────────────────────────────────────────────────────
fprintf('[1] Loading data...\n');
[appointments, provider_avail, dist_matrix] = data_loader(WEEK);
fprintf('    Appointments: %d\n', height(appointments));
fprintf('    Providers:    %d\n', numel(unique(appointments.provider)));

ALL_ROOMS = {'ER1','ER2','ER3','ER4','ER5','ER6','ER7','ER8',...
             'ER9','ER10','ER11','ER12','ER13','ER14','ER15','ER16'};

%% ── 2. Model 1: Feasibility check ───────────────────────────────────────────
fprintf('\n[2] Model 1 — Feasibility Packing per day...\n');
dates = unique(appointments.date);
for di = 1:numel(dates)
    day_appts = appointments(strcmp(appointments.date, dates{di}), :);
    if height(day_appts) == 0, continue; end
    day_name = char(day_appts.day_of_week(1));
    res1 = model1(day_appts, ALL_ROOMS, 0.0);
    if res1.feasible
        fprintf('    %s (%s): feasible, %d rooms used\n', ...
                char(dates{di}), day_name, numel(res1.rooms_used));
    else
        fprintf('    %s (%s): INFEASIBLE (exit=%d)\n', ...
                char(dates{di}), day_name, res1.status);
    end
end

%% ── 3. Model 2: Generate provider-day schedules ───────────────────────────
fprintf('\n[3] Model 2 — Provider-Day Schedule Generation...\n');

providers = unique(appointments.provider);
days_of_week = {'Monday','Tuesday','Wednesday','Thursday','Friday'};

all_schedules = struct([]);
sched_count = 0;

for pi = 1:numel(providers)
    prov = char(providers(pi));
    for di = 1:numel(days_of_week)
        day = days_of_week{di};

        % Get appointments for this (provider, day)
        mask = strcmp(appointments.provider, prov) & ...
               strcmp(appointments.day_of_week, day);
        pd_appts = appointments(mask, :);
        if height(pd_appts) == 0, continue; end

        % Get provider cluster
        cluster = get_provider_cluster(prov, day, WEEK, provider_avail, ...
                                        dist_matrix, PROXIMITY_THRESHOLD, ALL_ROOMS);

        % Solve Model 2
        res2 = model2(pd_appts, cluster, dist_matrix, DELTA_FRAC, GAMMA, ETA);

        sched_count = sched_count + 1;
        all_schedules(sched_count).provider    = prov;
        all_schedules(sched_count).day         = day;
        all_schedules(sched_count).week        = WEEK;
        all_schedules(sched_count).feasible    = res2.feasible;
        all_schedules(sched_count).assignment  = res2.assignment;
        all_schedules(sched_count).num_switches = res2.num_switches;
        all_schedules(sched_count).total_travel = res2.total_travel;
        all_schedules(sched_count).cost        = res2.cost;
        all_schedules(sched_count).alpha       = res2.alpha;
        all_schedules(sched_count).beta        = res2.beta;
        all_schedules(sched_count).appt_ids    = pd_appts.appt_id;

        fprintf('    %s %s: %s, switches=%d, travel=%.1fm\n', ...
                prov, day, ...
                ternary(res2.feasible,'OK','INFEAS'), ...
                res2.num_switches, res2.total_travel);
    end
end

fprintf('    Generated %d schedules (%d feasible).\n', ...
        sched_count, sum([all_schedules.feasible]));

%% ── 4. Model 3: Master Schedule Selection ────────────────────────────────────
fprintf('\n[4] Model 3 — Master Schedule Selection...\n');

res3 = model3(all_schedules, appointments.appt_id, true);

if res3.feasible
    fprintf('    Total cost = %.2f\n', res3.total_cost);
    fprintf('    Selected %d schedules.\n', numel(res3.selected_idx));
else
    fprintf('    Master problem not solved optimally (status=%d).\n', res3.status);
end

%% ── 5. Extract final assignments ─────────────────────────────────────────────
appt_room = repmat({''}, height(appointments), 1);

for si = res3.selected_idx
    sch = all_schedules(si);
    for ai = 1:numel(sch.appt_ids)
        appt_id = sch.appt_ids(ai);
        idx = find(appointments.appt_id == appt_id, 1);
        if ~isempty(idx) && ~isempty(sch.assignment)
            appt_room{idx} = sch.assignment{ai};
        end
    end
end

appointments.assigned_room = string(appt_room);
assigned_count = sum(appointments.assigned_room ~= "");
fprintf('\n    Appointments with assigned rooms: %d / %d (%.1f%%)\n', ...
        assigned_count, height(appointments), ...
        100*assigned_count/height(appointments));

%% ── 6. KPI Table ─────────────────────────────────────────────────────────────
fprintf('\n[5] Key Performance Indicators\n');
fprintf('%-30s  %10s  %10s  %10s\n', 'Metric', 'Value', '', '');
fprintf('%s\n', repmat('-',1,55));

n_feasible  = sum([all_schedules.feasible]);
avg_switches = mean([all_schedules([all_schedules.feasible]).num_switches]);
total_travel = sum([all_schedules([all_schedules.feasible]).total_travel]);
coverage_pct = 100 * assigned_count / height(appointments);

fprintf('%-30s  %10.1f%%\n', 'Coverage Rate', coverage_pct);
fprintf('%-30s  %10.2f\n',   'Avg Switches per Provider-Day', avg_switches);
fprintf('%-30s  %10.1f m\n', 'Total Provider Travel', total_travel);
fprintf('%-30s  %10.2f\n',   'Total Scheduling Cost', res3.total_cost);

%% ── 7. Gantt Chart ───────────────────────────────────────────────────────────
fprintf('\n[6] Generating Gantt charts...\n');
days_available = unique(appointments.day_of_week);
for di = 1:numel(days_available)
    day = char(days_available(di));
    day_appts = appointments(strcmp(appointments.day_of_week, day) & ...
                              appointments.assigned_room ~= "", :);
    if height(day_appts) == 0, continue; end
    plot_gantt(day_appts, day, WEEK);
end

%% ── 8. Save results ──────────────────────────────────────────────────────────
out_fname = sprintf('results_week%d.csv', WEEK);
writetable(appointments, out_fname);
fprintf('\nResults saved to %s\n', out_fname);
fprintf('Done.\n');


%% ══════════════════════════════════════════════════════════════════════════════
%  Helper functions
%% ══════════════════════════════════════════════════════════════════════════════

function cluster = get_provider_cluster(prov, day, week, provider_avail, ...
                                          dist_matrix, threshold, ALL_ROOMS)
    mask = strcmp(provider_avail.provider, prov) & ...
           strcmp(provider_avail.day, day);
    if ~any(mask) || ~provider_avail.available(find(mask,1))
        cluster = ALL_ROOMS;
        return;
    end
    home_room = char(provider_avail.room_am(find(mask,1)));
    if isempty(home_room)
        home_room = char(provider_avail.room_pm(find(mask,1)));
    end
    if isempty(home_room)
        cluster = ALL_ROOMS; return;
    end
    hr_idx = find(strcmp(ALL_ROOMS, home_room), 1);
    if isempty(hr_idx)
        cluster = ALL_ROOMS; return;
    end
    cluster = {};
    for r = 1:numel(ALL_ROOMS)
        if dist_matrix(hr_idx, r) <= threshold
            cluster{end+1} = ALL_ROOMS{r}; %#ok<AGROW>
        end
    end
    if isempty(cluster), cluster = {home_room}; end
end


function plot_gantt(day_appts, day, week)
    providers = unique(day_appts.provider);
    nP = numel(providers);
    colors = lines(16);

    ALL_ROOMS = {'ER1','ER2','ER3','ER4','ER5','ER6','ER7','ER8',...
                 'ER9','ER10','ER11','ER12','ER13','ER14','ER15','ER16'};

    fig = figure('Name', sprintf('%s Week%d', day, week), ...
                 'Position', [100 100 1200 max(400, nP*30)]);
    hold on;

    for pi = 1:nP
        prov = char(providers(pi));
        p_appts = day_appts(strcmp(day_appts.provider, prov), :);
        for ai = 1:height(p_appts)
            room = char(p_appts.assigned_room(ai));
            ri   = find(strcmp(ALL_ROOMS, room), 1);
            if isempty(ri), ri = 1; end
            color = colors(ri, :);
            rectangle('Position', ...
                [p_appts.start_min(ai), nP - pi, ...
                 p_appts.duration_min(ai), 0.85], ...
                'FaceColor', color, 'EdgeColor', 'k', 'LineWidth', 0.5);
            if p_appts.duration_min(ai) >= 15
                text(p_appts.start_min(ai)+1, nP - pi + 0.4, room, ...
                     'FontSize', 6, 'Color', 'w', 'FontWeight', 'bold');
            end
        end
    end

    set(gca, 'YTick', 0.5:nP-0.5, 'YTickLabel', flip(providers), ...
             'FontSize', 8);
    xlabel('Time (minutes from midnight)');
    ylabel('Provider');
    title(sprintf('Provider Schedule — %s, Week %d', day, week));

    % X-axis labels
    x_ticks = 480:30:1020;
    labels   = arrayfun(@(m) sprintf('%02d:%02d', floor(m/60), mod(m,60)), ...
                         x_ticks, 'UniformOutput', false);
    set(gca, 'XTick', x_ticks, 'XTickLabel', labels, ...
             'XTickLabelRotation', 45);
    xlim([480 1020]); ylim([0 nP]);

    % Admin blocks
    xpatch = [540 570 570 540; 690 720 720 690; 720 780 780 720; 990 1020 1020 990]';
    ypatch = repmat([0; 0; nP; nP], 1, 4);
    patch(xpatch, ypatch, [1 0 0], 'FaceAlpha', 0.08, 'EdgeColor', 'none');

    grid on; box on;
    fname = sprintf('gantt_%s_W%d.png', day, week);
    saveas(fig, fname);
    fprintf('    Saved %s\n', fname);
    close(fig);
end


function s = ternary(cond, a, b)
    if cond, s = a; else, s = b; end
end
