"""
SFAB Outsource Schedule Generator
----------------------------------
Reads an ERP export (CSV or Excel) and produces 7 per-operation schedule CSVs
ready to be loaded into the scheduling dashboard.

Assumptions / clarifications applied
-------------------------------------
- No "Index" column exists in the export; "PO No" is used as
  PO_Number_or_Job_Number.  Change PO_NUMBER_COL below if needed.
- Packing source column is "Pack Time Hr" (actual name in the export).
- Target week  = ISO week of (Due Date − 7 days), i.e. the week BEFORE due.
- Nothing is scheduled earlier than the current week.
- Level-loading: each job is placed in the least-loaded week within its valid
  window, spreading work evenly across available weeks.
- No week is intentionally given more than WEEKLY_HOUR_CAP hours; if all weeks
  in a window are already at the cap the least-loaded week is used anyway.
- On-time items: valid window = [current_week … target_week].
- Overdue items: valid window = [current_week+1 … current_week+OVERDUE_WEEKS] (weeks 13–20 for week 12 today).
- Rows where Balance ≤ 0 or operation time ≤ 0 are excluded.
- Hours Remaining = (Production Time × Quantity) + Setup Time
"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from datetime import datetime, timedelta

import pandas as pd

# ── tuneable constants ────────────────────────────────────────────────────────

DOWNLOADS_DIR   = os.path.join(os.path.expanduser("~"), "Downloads")
DEFAULT_FILE    = os.path.join(DOWNLOADS_DIR, "Updated Final Rolled Up Correctly.csv")

SOURCE_VALUE    = "SFAB"
PO_NUMBER_COL   = "PO No"          # change to "Item No" if that is "Index"
PART_COL        = "Part No"
QTY_COL         = "Balance"
DUE_DATE_COL    = "Due Date"
SUPPLIER_COL    = "Supplier"

WEEKLY_HOUR_CAP  = 50.0   # maximum hours to willingly assign per station per week
OVERDUE_WEEKS    = 8      # weeks in the scheduling window for overdue items

SETUP_TIMES: dict[str, float] = {
    "Laser":    0.250,
    "Bend":     0.330,
    "Assembly": 0.167,
    "Weld":     0.250,
    "PEM":      0.083,
    "PC":       0.167,
    "Packing":  0.167,
}

OPERATION_COLUMNS: dict[str, str] = {
    "Laser":    "Laser Time Hr",
    "Bend":     "Bend Time Hr",
    "Assembly": "Assembly Time Hr",
    "Weld":     "Weld Time Hr",
    "PEM":      "PEM Time Hr",
    "PC":       "PC Time Hr",
    "Packing":  "Pack Time Hr",
}

OUTPUT_COLUMNS = [
    "Company",
    "Year",
    "Week",
    "Source",
    "PO_Number_or_Job_Number",
    "Part Number",
    "Quantity",
    "Production Time",
    "Setup Time",
    "Hours Remaining",
]


# ── core logic ────────────────────────────────────────────────────────────────

def _week_before_due(due_val) -> tuple[int | None, int | None]:
    """Return (ISO year, ISO week) for the week that is 7 days before *due_val*."""
    try:
        if pd.isna(due_val):
            return None, None
    except (TypeError, ValueError):
        return None, None
    try:
        target = pd.to_datetime(due_val).to_pydatetime() - timedelta(days=7)
        iso = target.isocalendar()
        return int(iso[0]), int(iso[1])
    except Exception:
        return None, None


def _iso_week_offset(base: datetime, weeks: int) -> tuple[int, int]:
    """Return the ISO (year, week) that is *weeks* weeks after *base*."""
    iso = (base + timedelta(weeks=weeks)).isocalendar()
    return int(iso[0]), int(iso[1])


def _weeks_in_range(
    today: datetime, end_yw: tuple[int, int]
) -> list[tuple[int, int]]:
    """
    Return every ISO (year, week) from today's week up to *end_yw* inclusive.
    If *end_yw* is already in the past the list contains only today's week.
    """
    current_yw = _iso_week_offset(today, 0)
    if end_yw < current_yw:
        return [current_yw]
    window: list[tuple[int, int]] = []
    cursor = today
    while True:
        yw = _iso_week_offset(cursor, 0)
        window.append(yw)
        if yw >= end_yw:
            break
        cursor += timedelta(weeks=1)
    return window


def _pick_best_week(
    window: list[tuple[int, int]],
    week_load: dict[tuple[int, int], float],
    cap: float = WEEKLY_HOUR_CAP,
) -> tuple[int, int]:
    """
    Return the week in *window* with the least assigned hours.
    Weeks still under *cap* are always preferred over those at or above it.
    """
    under_cap = [w for w in window if week_load.get(w, 0.0) < cap]
    candidates = under_cap if under_cap else window
    return min(candidates, key=lambda w: week_load.get(w, 0.0))


def build_operation_df(df: pd.DataFrame, operation: str, today: datetime) -> pd.DataFrame:
    src_col = OPERATION_COLUMNS[operation]
    setup   = SETUP_TIMES[operation]

    current_yw = _iso_week_offset(today, 0)
    current_year, current_week = current_yw
    # Overdue items start scheduling from next week (not the current week)
    overdue_window = [_iso_week_offset(today, i) for i in range(1, OVERDUE_WEEKS + 1)]

    # Filter to rows that have a positive operation time AND a positive balance
    time_numeric = pd.to_numeric(df[src_col], errors="coerce")
    bal_numeric  = pd.to_numeric(df[QTY_COL],  errors="coerce")
    mask = (time_numeric > 0) & (bal_numeric > 0)
    subset = df[mask]

    # week_load tracks cumulative hours assigned so far for this operation
    week_load: dict[tuple[int, int], float] = {}

    on_time: list[tuple[tuple[int, int], dict]] = []  # (target_yw, record)
    overdue: list[dict] = []

    for _, row in subset.iterrows():
        year, week = _week_before_due(row.get(DUE_DATE_COL))
        qty       = float(pd.to_numeric(row[QTY_COL],  errors="coerce"))
        prod_time = float(pd.to_numeric(row[src_col],   errors="coerce"))
        hours_rem = round((prod_time * qty) + setup, 4)

        record = {
            "Company":                 str(row.get(SUPPLIER_COL, "") or ""),
            "Year":                    None,
            "Week":                    None,
            "Source":                  SOURCE_VALUE,
            "PO_Number_or_Job_Number": str(row.get(PO_NUMBER_COL, "") or ""),
            "Part Number":             str(row.get(PART_COL, "") or ""),
            "Quantity":                qty,
            "Production Time":         prod_time,
            "Setup Time":              setup,
            "Hours Remaining":         hours_rem,
        }

        if year is None or (year, week) < current_yw:
            overdue.append(record)
        else:
            on_time.append(((year, week), record))

    # Sort on-time jobs by target week (earliest deadline first) so tight
    # deadlines claim their preferred weeks before later ones fill them.
    on_time.sort(key=lambda x: x[0])

    results: list[dict] = []

    # Assign on-time jobs: valid window = [current_week … target_week]
    for target_yw, record in on_time:
        window = _weeks_in_range(today, target_yw)
        chosen = _pick_best_week(window, week_load)
        record["Year"] = chosen[0]
        record["Week"] = chosen[1]
        week_load[chosen] = week_load.get(chosen, 0.0) + record["Hours Remaining"]
        results.append(record)

    # Assign overdue jobs: valid window = next OVERDUE_WEEKS weeks
    for record in overdue:
        chosen = _pick_best_week(overdue_window, week_load)
        record["Year"] = chosen[0]
        record["Week"] = chosen[1]
        week_load[chosen] = week_load.get(chosen, 0.0) + record["Hours Remaining"]
        results.append(record)

    result = pd.DataFrame(results, columns=OUTPUT_COLUMNS)
    return result.sort_values(["Year", "Week"], ascending=True).reset_index(drop=True)


def process_file(filepath: str, log_fn) -> list[str]:
    ext = os.path.splitext(filepath)[1].lower()
    log_fn(f"Reading: {filepath}")

    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)

    log_fn(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

    today = datetime.today()
    cur_week = int(today.isocalendar()[1])
    log_fn(f"Today: {today.strftime('%Y-%m-%d')}  —  ISO Week {cur_week}\n")

    written: list[str] = []
    for operation in OPERATION_COLUMNS:
        op_df    = build_operation_df(df, operation, today)
        out_path = os.path.join(DOWNLOADS_DIR, f"SFAB_{operation}_Schedule.csv")
        op_df.to_csv(out_path, index=False)
        log_fn(f"  {operation:<10s}  {len(op_df):>4d} rows  →  {os.path.basename(out_path)}")
        written.append(out_path)

    log_fn(f"\nAll {len(written)} files written to:\n  {DOWNLOADS_DIR}")
    return written


# ── Tkinter UI ────────────────────────────────────────────────────────────────

class SFABScheduleApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("SFAB Outsource Schedule Generator")
        self.root.geometry("720x500")
        self.root.resizable(True, True)

        self.filepath = tk.StringVar(
            value=DEFAULT_FILE if os.path.exists(DEFAULT_FILE) else ""
        )
        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # File selector
        file_frame = ttk.LabelFrame(self.root, text="ERP Export File", padding=10)
        file_frame.pack(fill=tk.X, padx=12, pady=(12, 6))
        file_frame.columnconfigure(1, weight=1)

        ttk.Label(file_frame, text="File (CSV / Excel):").grid(
            row=0, column=0, sticky=tk.W
        )
        ttk.Entry(file_frame, textvariable=self.filepath).grid(
            row=0, column=1, sticky=tk.EW, padx=6
        )
        ttk.Button(file_frame, text="Browse…", command=self._browse).grid(
            row=0, column=2
        )

        # Info label
        ttk.Label(
            self.root,
            text=(
                "Generates 7 schedule CSVs (Laser, Bend, Assembly, Weld, PEM, PC, Packing)\n"
                f"Output folder: {DOWNLOADS_DIR}"
            ),
            justify=tk.LEFT,
            foreground="gray",
        ).pack(anchor=tk.W, padx=14, pady=(0, 4))

        # Run button
        self.run_btn = ttk.Button(
            self.root,
            text="Generate Schedule CSVs",
            command=self._run,
        )
        self.run_btn.pack(pady=6)

        # Log area
        log_frame = ttk.LabelFrame(self.root, text="Log", padding=6)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        self.log_box = scrolledtext.ScrolledText(
            log_frame,
            state=tk.DISABLED,
            font=("Consolas", 9),
            wrap=tk.WORD,
        )
        self.log_box.pack(fill=tk.BOTH, expand=True)

    # ── event handlers ────────────────────────────────────────────────────────

    def _browse(self) -> None:
        path = filedialog.askopenfilename(
            title="Select ERP Export File",
            initialdir=DOWNLOADS_DIR,
            filetypes=[
                ("CSV / Excel", "*.csv *.xlsx *.xls"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.filepath.set(path)

    def _log(self, msg: str) -> None:
        self.log_box.config(state=tk.NORMAL)
        self.log_box.insert(tk.END, msg + "\n")
        self.log_box.see(tk.END)
        self.log_box.config(state=tk.DISABLED)
        self.root.update_idletasks()

    def _run(self) -> None:
        path = self.filepath.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showerror("File not found", "Please select a valid data file.")
            return

        # Clear log and disable button while running
        self.log_box.config(state=tk.NORMAL)
        self.log_box.delete("1.0", tk.END)
        self.log_box.config(state=tk.DISABLED)
        self.run_btn.config(state=tk.DISABLED)

        def worker() -> None:
            try:
                process_file(path, self._log)
                self.root.after(
                    0,
                    lambda: messagebox.showinfo(
                        "Done", "All 7 schedule CSVs written to your Downloads folder."
                    ),
                )
            except Exception as exc:
                self._log(f"\nERROR: {exc}")
                self.root.after(
                    0, lambda: messagebox.showerror("Error", str(exc))
                )
            finally:
                self.root.after(0, lambda: self.run_btn.config(state=tk.NORMAL))

        threading.Thread(target=worker, daemon=True).start()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    SFABScheduleApp(root)
    root.mainloop()
