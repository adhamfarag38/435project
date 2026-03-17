function [appointments, provider_avail, dist_matrix] = data_loader(week)
% DATA_LOADER  Load and preprocess Examination Room Scheduling data.
%
% [appointments, provider_avail, dist_matrix] = data_loader(week)
%
% Inputs:
%   week  - 1 or 2
%
% Outputs:
%   appointments   - table with fields: appt_id, date, day_of_week, provider,
%                    start_min, duration_min, end_min, no_show
%   provider_avail - table: provider, day, room_am, room_pm, available
%   dist_matrix    - 16x16 symmetric distance matrix (ER1..ER16)

if nargin < 1, week = 1; end

data_path = '../data/';

%% ── Appointments ────────────────────────────────────────────────────────────
fname = sprintf('%sAppointmentDataWeek%d.csv', data_path, week);
raw = readtable(fname, 'VariableNamingRule', 'preserve');

% Standardise column names
raw.Properties.VariableNames = lower(strtrim(strrep(...
    raw.Properties.VariableNames, ' ', '_')));

% Drop deleted and cancelled
if ismember('deleted_appts', raw.Properties.VariableNames)
    raw = raw(~strcmpi(strtrim(raw.deleted_appts), 'Y'), :);
end
if ismember('cancelled_appts', raw.Properties.VariableNames)
    raw = raw(~strcmpi(strtrim(raw.cancelled_appts), 'Y'), :);
end

n = height(raw);
appt_id     = (1:n)';
start_min   = zeros(n, 1);
duration_min = zeros(n, 1);
no_show     = false(n, 1);
day_of_week = strings(n, 1);

for i = 1:n
    % Parse start time
    t_str = char(raw.appt_time(i));
    parts = strsplit(t_str, ':');
    start_min(i) = str2double(parts{1})*60 + str2double(parts{2});

    % Duration
    d = raw.appt_duration(i);
    if isnumeric(d)
        duration_min(i) = max(5, d);
    else
        duration_min(i) = 15;
    end

    % No-show flag
    ns_val = char(raw.no_show_appts(i));
    no_show(i) = strcmpi(strtrim(ns_val), 'Y');

    % Day of week from date string  'MM-DD-YYYY'
    d_str = char(raw.appt_date(i));
    try
        dt = datetime(d_str, 'InputFormat', 'MM-dd-yyyy');
        day_of_week(i) = string(datestr(dt, 'dddd'));
    catch
        day_of_week(i) = "Unknown";
    end
end

end_min = start_min + duration_min;

appointments = table(appt_id, string(raw.patient_id), string(raw.appt_date), ...
    day_of_week, string(raw.primary_provider), ...
    start_min, duration_min, end_min, no_show, ...
    'VariableNames', {'appt_id','patient_id','date','day_of_week',...
                      'provider','start_min','duration_min','end_min','no_show'});

%% ── Provider availability ────────────────────────────────────────────────────
fname2 = sprintf('%sProviderRoomAssignmentWeek%d.csv', data_path, week);
raw2 = readtable(fname2, 'VariableNamingRule', 'preserve');

days = {'Monday','Tuesday','Wednesday','Thursday','Friday'};
records = {};
for i = 1:height(raw2)
    prov = char(raw2{i,1});
    if isempty(strtrim(prov)), continue; end

    for di = 1:numel(days)
        day_name = days{di};
        if ~ismember(day_name, raw2.Properties.VariableNames), continue; end
        raw_val = char(raw2{i, day_name});
        [room_am, room_pm, avail] = parse_room_cell(raw_val);
        records(end+1,:) = {string(prov), string(day_name), ...
                             string(room_am), string(room_pm), avail}; %#ok<AGROW>
    end
end

provider_avail = cell2table(records, ...
    'VariableNames', {'provider','day','room_am','room_pm','available'});

%% ── Room distance matrix ─────────────────────────────────────────────────────
fname3 = sprintf('%sroom_proximity_matrix.csv', data_path);
raw3 = readtable(fname3, 'VariableNamingRule', 'preserve', 'ReadRowNames', true);

rooms = {'ER1','ER2','ER3','ER4','ER5','ER6','ER7','ER8',...
         'ER9','ER10','ER11','ER12','ER13','ER14','ER15','ER16'};

nr = numel(rooms);
dist_matrix = zeros(nr, nr);
row_names = raw3.Properties.RowNames;
col_names = raw3.Properties.VariableNames;

for i = 1:nr
    ri = find(strcmpi(row_names, rooms{i}), 1);
    if isempty(ri), continue; end
    for j = 1:nr
        ci = find(strcmpi(col_names, rooms{j}), 1);
        if isempty(ci), continue; end
        val = raw3{ri, ci};
        if isnumeric(val) && ~isnan(val)
            dist_matrix(i, j) = val;
        end
    end
end
% Make symmetric
dist_matrix = (dist_matrix + dist_matrix') / 2;

end  % data_loader


%% ── Helper: parse room cell ─────────────────────────────────────────────────
function [room_am, room_pm, available] = parse_room_cell(raw_val)
    raw_val = strtrim(raw_val);
    unavail = {'','N/A','CLOSED','NO ROOM AVAILABLE','NO ROOM'};
    if any(strcmpi(raw_val, unavail))
        room_am = ''; room_pm = ''; available = false; return;
    end
    available = true;
    % Split on '/' for AM/PM rooms
    parts = strsplit(raw_val, '/');
    room_am = extract_room_number(strtrim(parts{1}));
    if numel(parts) >= 2
        room_pm = extract_room_number(strtrim(parts{2}));
    else
        room_pm = room_am;
    end
end

function room_id = extract_room_number(s)
    % Remove (AM)/(PM) qualifiers
    s = regexprep(s, '\s*\(AM\)\s*|\s*\(PM\)\s*', '', 'ignorecase');
    tok = regexp(s, '\d+', 'match');
    if isempty(tok)
        room_id = '';
    else
        room_id = sprintf('ER%s', tok{1});
    end
end
