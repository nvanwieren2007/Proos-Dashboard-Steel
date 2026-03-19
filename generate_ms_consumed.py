"""
generate_ms_consumed.py
=======================
Standalone script: reads the MS ERP job-operations CSV, applies forward-
scheduling against the capacity limits, and writes the result as
ms_consumed_data.csv (and optionally Current_MS.csv).

Usage
-----
  python generate_ms_consumed.py                        # uses defaults below
  python generate_ms_consumed.py "path\\to\\erp.csv"   # custom ERP file path

Output files (written next to this script)
------------------------------------------
  ms_consumed_data.csv   — drop this into the Proos-Capacity-Dashboard repo
  Current_MS.csv         — a copy for your reference / manual upload

"""

import sys
from pathlib import Path
from datetime import datetime, date, timedelta
import pandas as pd

# ─── Configuration ─────────────────────────────────────────────────────────

_SCRIPT_DIR   = Path(__file__).parent
_REPO_DIR     = Path(r"C:\Users\nvanwieren\TestingPython\Proos-Capacity-Dashboard")

_DEFAULT_ERP  = Path(r"C:\Users\nvanwieren\Downloads\Job Operations from PO source data.csv")
_LIMITS_CSV   = _SCRIPT_DIR / "capacity_limits_template.csv"
_OUT_DATA     = _SCRIPT_DIR / "ms_consumed_data.csv"
_OUT_CURRENT  = _SCRIPT_DIR / "Current_MS.csv"
_REPO_DATA    = _REPO_DIR   / "ms_consumed_data.csv"

# ─── Must match capacity_dashboard.py exactly ──────────────────────────────

OPERATIONS = [
    "Laser",
    "Press_Brake",
    "Weld",
    "Resistance_Weld",
    "Powder_Coat",
    "Assembly",
    "Packing",
    "PEM",
]

_ERP_OP_MAP = {
    "Laser WIP":        "Laser",
    "Form WIP 1":       "Press_Brake",
    "Form FG":          "Press_Brake",
    "Pem WIP 1":        "PEM",
    "Pem WIP 2":        "PEM",
    "Pem FG":           "PEM",
    "OS Paint WIP":     "Powder_Coat",
    "OS Paint FG":      "Powder_Coat",
    "Assemble WIP 1":   "Assembly",
    "Secondary Ops FG": "Assembly",
    "Weld WIP 1":       "Weld",
    "Weld FG":          "Weld",
}

# ─── Helpers ───────────────────────────────────────────────────────────────

def _advance_iso_week(year: int, week: int) -> tuple:
    d = date.fromisocalendar(year, week, 1) + timedelta(weeks=1)
    iso = d.isocalendar()
    return int(iso.year), int(iso.week)


def get_cap(capacity: pd.DataFrame, company: str, operation: str,
            week: int = None, year: int = None) -> float:
    df = capacity[capacity["Company"] == company]
    if df.empty or operation not in df.columns:
        return 0.0
    has_wk = "Week" in df.columns and "Year" in df.columns
    if has_wk and week is not None and year is not None:
        override = df[
            (df["Year"].fillna(0).astype(int) == year) &
            (df["Week"].fillna(0).astype(int) == week)
        ]
        if not override.empty:
            return float(override[operation].values[0])
    if has_wk:
        default = df[
            (df["Year"].fillna(0).astype(int) == 0) |
            (df["Week"].fillna(0).astype(int) == 0)
        ]
        if not default.empty:
            return float(default[operation].values[0])
    return float(df[operation].values[0])


# ─── Core parser (forward-schedule mode) ───────────────────────────────────

def parse_erp_to_consumed(erp_df: pd.DataFrame,
                           capacity: pd.DataFrame,
                           today_iso_week: int,
                           today_year: int) -> pd.DataFrame:
    """
    Forward-schedule all open MS jobs against capacity limits starting from
    today.  Each operation sequences independently (they run in parallel).
    Jobs are processed soonest-due-first so urgent work gets earlier slots.
    """
    df = erp_df.copy()
    df = df[df["jobOperationStatus"] != "Completed"].copy()
    df = df[df["operationCode"].isin(_ERP_OP_MAP)].copy()
    df["workcenter"] = df["operationCode"].map(_ERP_OP_MAP)

    df["quantityCompleted"] = pd.to_numeric(df["quantityCompleted"], errors="coerce").fillna(0)
    df["quantity"]          = pd.to_numeric(df["quantity"],          errors="coerce").fillna(0)
    df["remaining_qty"]     = (df["quantity"] - df["quantityCompleted"]).clip(lower=0)
    df["rate"]      = pd.to_numeric(df["rate"],      errors="coerce").replace(0, float("nan"))
    df["setupTime"] = pd.to_numeric(df["setupTime"], errors="coerce").fillna(0)
    df["hours"]     = df["setupTime"] + (df["remaining_qty"] / df["rate"]).fillna(0)
    df["dueDate_dt"] = pd.to_datetime(df["dueDate"], utc=True, errors="coerce")

    _MAX_WEEKS = 208
    week_hours: dict = {}

    for op in OPERATIONS:
        op_jobs = df[df["workcenter"] == op].sort_values("dueDate_dt", na_position="last")
        if op_jobs.empty:
            continue

        ptr_yr, ptr_wk = today_year, today_iso_week
        cap_used: dict = {}
        weeks_advanced = 0

        for _, job in op_jobs.iterrows():
            remaining = float(job["hours"])
            if remaining <= 1e-9:
                continue

            while remaining > 1e-9:
                cap_limit = get_cap(capacity, "MS", op, ptr_wk, ptr_yr)
                used      = cap_used.get((ptr_yr, ptr_wk), 0.0)
                available = cap_limit - used

                if available <= 1e-9:
                    ptr_yr, ptr_wk = _advance_iso_week(ptr_yr, ptr_wk)
                    weeks_advanced += 1
                    if weeks_advanced > _MAX_WEEKS:
                        break
                    continue

                take = min(remaining, available)
                cap_used[(ptr_yr, ptr_wk)] = used + take

                key = (ptr_yr, ptr_wk)
                if key not in week_hours:
                    week_hours[key] = {o: 0.0 for o in OPERATIONS}
                week_hours[key][op] += take
                remaining -= take

    if not week_hours:
        print("⚠  No schedulable hours found — check operationCode mapping.")
        return pd.DataFrame(columns=["Company", "Year", "Week"] + OPERATIONS)

    rows = []
    for (yr, wk) in sorted(week_hours):
        row = {"Company": "MS", "Year": yr, "Week": wk}
        for op in OPERATIONS:
            row[op] = round(week_hours[(yr, wk)].get(op, 0.0), 4)
        rows.append(row)
    return pd.DataFrame(rows)


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    erp_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_ERP

    # Validate inputs
    if not erp_path.exists():
        print(f"❌  ERP file not found:\n    {erp_path}")
        sys.exit(1)
    if not _LIMITS_CSV.exists():
        print(f"❌  Capacity limits file not found:\n    {_LIMITS_CSV}")
        sys.exit(1)

    print(f"📂  ERP source : {erp_path}")
    print(f"📂  Limits     : {_LIMITS_CSV}")

    erp_df   = pd.read_csv(erp_path)
    capacity = pd.read_csv(_LIMITS_CSV)

    today    = date.today().isocalendar()
    iso_week = int(today.week)
    iso_year = int(today.year)
    print(f"📅  Scheduling from ISO week {iso_week} / {iso_year}")

    # Parse
    result = parse_erp_to_consumed(erp_df, capacity, iso_week, iso_year)

    if result.empty:
        print("No rows generated — nothing written.")
        sys.exit(0)

    print(f"\n✅  {len(result)} week-rows generated across "
          f"weeks {result['Week'].min()}–{result['Week'].max()} "
          f"({result['Year'].iloc[0]})")
    print(result.to_string(index=False))

    # Write outputs
    result.to_csv(_OUT_DATA,    index=False)
    result.to_csv(_OUT_CURRENT, index=False)
    print(f"\n💾  Written: {_OUT_DATA}")
    print(f"💾  Written: {_OUT_CURRENT}")

    # Also copy to the repo directory if it exists
    if _REPO_DIR.exists():
        result.to_csv(_REPO_DATA, index=False)
        print(f"💾  Written: {_REPO_DATA}")
        print("\n📌  Next steps:")
        print(f"    cd \"{_REPO_DIR}\"")
        print( "    git add ms_consumed_data.csv")
        print( "    git commit -m \"Update MS consumed data\"")
        print( "    git push origin main")
    else:
        print(f"\n⚠   Repo dir not found ({_REPO_DIR}) — skipped repo copy.")
        print(f"    Manually copy {_OUT_DATA.name} into the repo and push.")


if __name__ == "__main__":
    main()
