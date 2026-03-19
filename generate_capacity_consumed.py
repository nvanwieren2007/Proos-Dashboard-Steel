# generate_capacity_consumed.py
#
# Reads the Proos manufacturing ERP job operations export and produces a
# week-by-week capacity-consumed summary across all manufacturing workcenters.
#
# Usage:  python generate_capacity_consumed.py
# Output: C:/Users/nvanwieren/Downloads/capacity_consumed.csv

import pandas as pd
from datetime import date, timedelta

# ── CONFIG ───────────────────────────────────────────────────────────────────
CSV_PATH    = r'C:\Users\nvanwieren\Downloads\open_job_ops 3-17-2026.csv'
IGNORE_PATH = r'C:\Users\nvanwieren\Downloads\Ignore_Parts.xlsx'
OUTPUT_PATH = r'C:\Users\nvanwieren\Downloads\capacity_consumed.csv'

TODAY      = date(2026, 3, 17)
START_DATE = date(2026, 3, 18)   # first day to schedule (tomorrow)
OVERDUE_WORKDAYS = 20            # spread overdue items across this many workdays

WORKCENTERS = [
    'Laser', 'Press_Brake', 'Weld', 'Resistance_Weld',
    'Powder_Coat', 'Assembly', 'Packing', 'PEM',
]

# ── OPERATION CODE → WORKCENTER ──────────────────────────────────────────────
OP_MAP = {
    'Laser WIP':             'Laser',
    'Laser FG':              'Laser',
    'Form WIP 1':            'Press_Brake',
    'Form FG':               'Press_Brake',
    'Pem WIP 1':             'PEM',
    'Pem FG':                'PEM',
    'Pem WIP 2':             'PEM',
    'OS Paint WIP':          'Powder_Coat',
    'OS Paint FG':           'Powder_Coat',
    'Assemble WIP 1':        'Assembly',
    'Secondary Ops FG':      'Assembly',
    'Secondary Ops WIP':     'Assembly',
    'Assemble FG STD':       'Assembly',
    'Assemble FG SMP':       'Assembly',
    'Countersink WIP':       'Assembly',
    'Tap FG':                'Assembly',
    'Cut to Length':         'Assembly',
    'Weld FG':               'Weld',
    'Weld WIP 1':            'Weld',
    'Resistance Weld WIP 1': 'Resistance_Weld',
    'Pack FG SMP':           'Packing',
    'Pick & Pack FG':        'Packing',
}

# Operation codes dropped before processing (outsourced / receive / etc.)
OPS_TO_DROP = [
    'OS Assemble FG', 'Receive (Ea)', 'OS Pem FG', 'OS Form FG',
    'OS Weld FG', 'OS Machine FG', 'OS Laser FG', 'OS Grind FG',
]

# ── HELPERS ──────────────────────────────────────────────────────────────────

def workdays_before_due(start: date, due: date) -> list[date]:
    """
    Return all Mon–Fri workdays from start up to (not including) due.
    Guarantees at least one workday is returned.
    """
    days = []
    cur = start
    while cur < due:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    return days if days else [start]


def next_n_workdays(start: date, n: int) -> list[date]:
    """Return the next n Mon–Fri workdays beginning on start (inclusive)."""
    days, cur = [], start
    while len(days) < n:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    return days


# ── LOAD & FILTER ─────────────────────────────────────────────────────────────

df = pd.read_csv(CSV_PATH)

ignore_parts = set(pd.read_excel(IGNORE_PATH)['Ignore_Parts'].dropna())
df = df[~df['partNumber'].isin(ignore_parts)]
df = df[~df['operationCode'].isin(OPS_TO_DROP)]
df = df[df['operationCode'].isin(OP_MAP)]

# Remaining quantity (clipped at 0 to avoid negatives from data quirks)
df['qty_remaining'] = (
    df['quantity'].fillna(0) - df['quantityCompleted'].fillna(0)
).clip(lower=0)

# Drop rows with nothing left to process
df = df[df['qty_remaining'] > 0]

# Drop rows where rate is zero or missing (can't compute hours)
df = df[df['rate'].notna() & (df['rate'] > 0)]

# Parse due dates; missing due dates treated as overdue
df['dueDate'] = pd.to_datetime(df['dueDate'], errors='coerce').dt.date

print(f'Rows to schedule: {len(df)}')

# ── SPREAD HOURS INTO WEEKS ───────────────────────────────────────────────────

# capacity[(year, iso_week, workcenter)] = total decimal hours consumed
capacity: dict[tuple, float] = {}

for _, row in df.iterrows():
    workcenter  = OP_MAP[row['operationCode']]
    # Hours = remaining quantity / rate (pieces per hour)
    total_hours = (row['qty_remaining'] / row['rate']) + row['setupTime']  # add setup time to total hours

    if total_hours <= 0:
        continue

    due = row['dueDate']

    if pd.isna(due) or due <= TODAY:
        # Overdue or no due date: spread over next OVERDUE_WORKDAYS workdays
        workdays = next_n_workdays(START_DATE, OVERDUE_WORKDAYS)
    else:
        # Spread evenly from START_DATE up to (not including) due date
        workdays = workdays_before_due(START_DATE, due)

    hours_per_day = total_hours / len(workdays)

    for d in workdays:
        iso_year, iso_week, _ = d.isocalendar()
        key = (int(iso_year), int(iso_week), workcenter)
        capacity[key] = capacity.get(key, 0.0) + hours_per_day

# ── BUILD OUTPUT DATAFRAME ────────────────────────────────────────────────────

week_set = sorted({(y, w) for y, w, _ in capacity})

rows = []
for year, week in week_set:
    r = {'Year': year, 'Week': week}
    for wc in WORKCENTERS:
        r[wc] = round(capacity.get((year, week, wc), 0.0), 2)
    rows.append(r)

out = pd.DataFrame(rows, columns=['Year', 'Week'] + WORKCENTERS)
out.to_csv(OUTPUT_PATH, index=False)

print(f'\nOutput: {len(out)} weeks written to:\n  {OUTPUT_PATH}')
print()
print(out.to_string(index=False))
