"""
Data loading and preprocessing for the Examination Room Scheduling problem.
Handles appointment data, provider availability, and room distance matrix.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re

DATA_PATH = "data/"

# ─── Operating hours (minutes from midnight) ─────────────────────────────────
LUNCH_START      = 12 * 60        # 720
LUNCH_END        = 13 * 60        # 780
NOON_ADMIN_START = 11 * 60 + 30   # 690
NOON_ADMIN_END   = 12 * 60        # 720

ADMIN_BLOCKS = {
    "weekday": {          # Monday–Thursday
        "morning":   (9 * 60,      9 * 60 + 30),   # 540–570
        "noon":      (11 * 60 + 30, 12 * 60),       # 690–720
        "lunch":     (12 * 60,     13 * 60),        # 720–780
        "afternoon": (16 * 60 + 30, 17 * 60),       # 990–1020
    },
    "friday": {
        "morning":   (8 * 60,      8 * 60 + 30),    # 480–510
        "noon":      (11 * 60 + 30, 12 * 60),       # 690–720
        "lunch":     (12 * 60,     13 * 60),        # 720–780
        "afternoon": (15 * 60,     15 * 60 + 30),   # 900–930
    },
}

# Rooms 1-16
ROOMS = [f"ER{i}" for i in range(1, 17)]

# Day-name to weekday number (Monday=0 ... Friday=4)
DAY_INDEX = {"Monday": 0, "Tuesday": 1, "Wednesday": 2,
             "Thursday": 3, "Friday": 4}


def time_to_minutes(t: str) -> int:
    """Convert 'HH:MM:SS' or 'HH:MM' string to minutes from midnight."""
    parts = str(t).strip().split(":")
    return int(parts[0]) * 60 + int(parts[1])


def minutes_to_str(m: int) -> str:
    return f"{m // 60:02d}:{m % 60:02d}"


def _parse_room_name(raw) -> str | None:
    """
    Normalise raw room string → 'ER{n}' or None if unavailable.
    Handles: 'Room 5', 'RM 5 (AM)', 'Room 5 (PM)', 'Room 11 (AM) / Room 6 (PM)', etc.
    Returns the *first* room mentioned (AM slot if AM/PM split).
    """
    if pd.isna(raw):
        return None
    s = str(raw).strip()
    if s in ("", "N/A", "CLOSED", "NO ROOM AVAILABLE", "NO ROOM"):
        return None
    # Take the part before '/' (AM room)
    s = s.split("/")[0].strip()
    # Strip qualifiers
    s = re.sub(r"\s*\(AM\)\s*|\s*\(PM\)\s*", "", s, flags=re.IGNORECASE).strip()
    # Match digit(s)
    m = re.search(r"\d+", s)
    return f"ER{m.group()}" if m else None


def _parse_room_name_pm(raw) -> str | None:
    """Return PM room if the cell contains 'Room X (AM) / Room Y (PM)' pattern."""
    if pd.isna(raw):
        return None
    s = str(raw)
    parts = s.split("/")
    if len(parts) < 2:
        return _parse_room_name(raw)
    pm_part = re.sub(r"\s*\(AM\)\s*|\s*\(PM\)\s*", "", parts[1], flags=re.IGNORECASE).strip()
    m = re.search(r"\d+", pm_part)
    return f"ER{m.group()}" if m else None


def _is_am_only(raw) -> bool:
    if pd.isna(raw):
        return False
    return bool(re.search(r"\(AM\)", str(raw), re.IGNORECASE)) and "/" not in str(raw)


def _is_pm_only(raw) -> bool:
    if pd.isna(raw):
        return False
    return bool(re.search(r"\(PM\)", str(raw), re.IGNORECASE)) and "/" not in str(raw)


# ─── Load appointments ────────────────────────────────────────────────────────

def load_appointments(week: int = 1) -> pd.DataFrame:
    """
    Returns cleaned appointment DataFrame with columns:
        appt_id, patient_id, date, day_of_week, provider,
        start_min, duration_min, end_min, no_show, week
    Deleted and cancelled appointments are dropped.
    """
    fname = DATA_PATH + f"AppointmentDataWeek{week}.csv"
    df = pd.read_csv(fname)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Rename for consistency
    df = df.rename(columns={
        "patient_id": "patient_id",
        "appt_date": "date",
        "primary_provider": "provider",
        "apptstatussingleview": "status",
        "cancelled_appts": "cancelled",
        "deleted_appts": "deleted",
        "no_show_appts": "no_show",
        "appt_time": "appt_time",
        "appt_duration": "duration_min",
    })
    # Strip stray spaces from column names that may still exist
    df.columns = [c.strip("_") for c in df.columns]

    # Drop deleted and cancelled
    df = df[df["deleted"].str.strip().str.upper() != "Y"]
    df = df[df["cancelled"].str.strip().str.upper() != "Y"]
    df = df.reset_index(drop=True)

    # Parse times
    df["start_min"] = df["appt_time"].apply(time_to_minutes)
    df["duration_min"] = pd.to_numeric(df["duration_min"], errors="coerce").fillna(15).astype(int)
    df["end_min"] = df["start_min"] + df["duration_min"]

    # Parse date → day of week
    df["date"] = df["date"].astype(str).str.strip()
    df["date_parsed"] = pd.to_datetime(df["date"], format="%m-%d-%Y", errors="coerce")
    df["day_of_week"] = df["date_parsed"].dt.day_name()

    # No-show flag
    df["no_show"] = df["no_show"].astype(str).str.strip().str.upper() == "Y"

    df["week"] = week
    df["appt_id"] = df.index  # unique integer id

    # Keep only relevant columns
    cols = ["appt_id", "patient_id", "date", "day_of_week", "provider",
            "start_min", "duration_min", "end_min", "no_show", "week"]
    return df[cols].copy()


def load_all_appointments() -> pd.DataFrame:
    w1 = load_appointments(1)
    w2 = load_appointments(2)
    w2["appt_id"] += len(w1)
    return pd.concat([w1, w2], ignore_index=True)


# ─── Load provider availability ───────────────────────────────────────────────

def load_provider_availability(week: int = 1) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
        provider, provider_type, day, available, room_am, room_pm, week
    where room_am / room_pm are ER-normalised room ids (or None).
    """
    fname = DATA_PATH + f"ProviderRoomAssignmentWeek{week}.csv"
    df = pd.read_csv(fname)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=["Primary Provider"])
    df = df[df["Primary Provider"].astype(str).str.strip() != ""]

    # Identify type column (Week 1 uses 'TYPE', Week 2 uses 'Provider Type')
    type_col = "TYPE" if "TYPE" in df.columns else "Provider Type"

    records = []
    for _, row in df.iterrows():
        provider = str(row["Primary Provider"]).strip()
        ptype = str(row[type_col]).strip() if not pd.isna(row[type_col]) else "UNKNOWN"

        for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
            if day not in df.columns:
                continue
            raw = row[day]
            room_am = _parse_room_name(raw)
            room_pm = _parse_room_name_pm(raw)
            am_only = _is_am_only(raw)
            pm_only = _is_pm_only(raw)

            available = room_am is not None or room_pm is not None

            records.append({
                "provider": provider,
                "provider_type": ptype,
                "day": day,
                "available": available,
                "room_am": room_am,
                "room_pm": room_pm,
                "am_only": am_only,
                "pm_only": pm_only,
                "week": week,
            })

    return pd.DataFrame(records)


def load_all_provider_availability() -> pd.DataFrame:
    w1 = load_provider_availability(1)
    w2 = load_provider_availability(2)
    return pd.concat([w1, w2], ignore_index=True)


# ─── Load room distance matrix ───────────────────────────────────────────────

def load_distance_matrix() -> pd.DataFrame:
    """Returns symmetric 16x16 distance matrix indexed and columned by ER1..ER16."""
    df = pd.read_csv(DATA_PATH + "room_proximity_matrix.csv", index_col=0)
    df.index = [str(i).strip() for i in df.index]
    df.columns = [str(c).strip() for c in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    # Make symmetric
    df = (df + df.T) / 2
    for r in ROOMS:
        if r not in df.index:
            df.loc[r] = 0
            df[r] = 0
    df = df.loc[ROOMS, ROOMS]
    return df


# ─── Provider-day appointments helper ────────────────────────────────────────

def get_provider_day_appointments(appointments: pd.DataFrame,
                                   provider: str, day: str) -> pd.DataFrame:
    """Return appointments for a specific provider on a given day-of-week."""
    mask = (appointments["provider"] == provider) & (appointments["day_of_week"] == day)
    return appointments[mask].sort_values("start_min").copy()


# ─── No-show probability ─────────────────────────────────────────────────────

def compute_noshow_rate(appointments: pd.DataFrame) -> dict[str, float]:
    """
    Compute per-provider no-show probability from historical data.
    Falls back to global rate for providers with too few observations.
    """
    global_rate = appointments["no_show"].mean()
    rates = {}
    for prov, grp in appointments.groupby("provider"):
        n = len(grp)
        rate = grp["no_show"].mean() if n >= 5 else global_rate
        rates[prov] = rate
    return rates


# ─── Admin-time overlap check ─────────────────────────────────────────────────

def is_admin_time(start_min: int, end_min: int, day: str) -> bool:
    """Check if [start_min, end_min) overlaps any admin/lunch block."""
    key = "friday" if day == "Friday" else "weekday"
    for _, (blk_start, blk_end) in ADMIN_BLOCKS[key].items():
        if start_min < blk_end and end_min > blk_start:
            return True
    return False


# ─── Overlap detection ───────────────────────────────────────────────────────

def compute_overlap_pairs(appointments: pd.DataFrame,
                           delta: dict[int, int] | None = None) -> list[tuple[int, int]]:
    """
    Return list of (appt_id_a, appt_id_b) pairs that overlap in time.
    delta: optional robust buffer (minutes) per appt_id.
    """
    overlaps = []
    rows = appointments.to_dict("records")
    for i, a in enumerate(rows):
        da = delta.get(a["appt_id"], 0) if delta else 0
        for b in rows[i + 1:]:
            db = delta.get(b["appt_id"], 0) if delta else 0
            # Overlap if NOT (a ends before b starts OR b ends before a starts)
            # With robust buffers: a uses [t_a, t_a + d_a + delta_a]
            a_end = a["end_min"] + da
            b_end = b["end_min"] + db
            if a["start_min"] < b_end and b["start_min"] < a_end:
                overlaps.append((a["appt_id"], b["appt_id"]))
    return overlaps


# ─── Provider cluster rooms ──────────────────────────────────────────────────

def get_provider_cluster(provider: str, day: str, week: int,
                          provider_avail: pd.DataFrame,
                          dist_matrix: pd.DataFrame,
                          proximity_threshold: float = 4.0) -> list[str]:
    """
    Return rooms within proximity_threshold of the provider's assigned room.
    If no assigned room, return all rooms.
    """
    row = provider_avail[
        (provider_avail["provider"] == provider) &
        (provider_avail["day"] == day) &
        (provider_avail["week"] == week)
    ]
    if row.empty or not row.iloc[0]["available"]:
        return ROOMS

    home_room = row.iloc[0]["room_am"] or row.iloc[0]["room_pm"]
    if home_room is None or home_room not in dist_matrix.index:
        return ROOMS

    cluster = [r for r in ROOMS
               if r in dist_matrix.columns and
               dist_matrix.loc[home_room, r] <= proximity_threshold]
    return cluster if cluster else [home_room]


if __name__ == "__main__":
    appts = load_all_appointments()
    avail = load_all_provider_availability()
    dist  = load_distance_matrix()
    print(f"Total appointments: {len(appts)}")
    print(f"Provider-day availability entries: {len(avail)}")
    print(f"Room distance matrix: {dist.shape}")
    print(appts.head())
