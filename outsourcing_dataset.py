"""
Outsourcing Dataset Builder
----------------------------
Merges labour-time data from per-quote CSVs into a source part list.

Source file  – CSV or Excel with columns: Part_Cleaned, Quote
Quote folder – directory of cleaned CSVs whose filenames end with the
               quote number (e.g. "RFQ_3991.csv" for quote "3991").

For each row the tool finds the matching quote CSV, locates the first row
whose ItemNumber *contains* the part number (substring match), and copies
the eight time columns into the source file.
"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd

TIME_COLUMNS = [
    "Laser Time Hr",
    "Bend Time Hr",
    "Assembly Time Hr",
    "Weld Time Hr",
    "PEM Time Hr",
    "PC Time Hr",
    "Pack Time Hr",
    "Drill Press Hr",
]


class OutsourcingDatasetApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Outsourcing Dataset Builder")
        self.root.geometry("740x560")
        self.root.resizable(True, True)

        self.source_file = tk.StringVar()
        self.csv_folder = tk.StringVar()

        self._build_ui()

    # ------------------------------------------------------------------ UI --

    def _build_ui(self) -> None:
        # ---- input path selectors ----
        input_frame = ttk.LabelFrame(self.root, text="Inputs", padding=10)
        input_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        input_frame.columnconfigure(1, weight=1)

        ttk.Label(input_frame, text="Source File (CSV / Excel):").grid(
            row=0, column=0, sticky=tk.W, pady=4
        )
        ttk.Entry(input_frame, textvariable=self.source_file).grid(
            row=0, column=1, sticky=tk.EW, padx=6, pady=4
        )
        ttk.Button(input_frame, text="Browse…", command=self._browse_source).grid(
            row=0, column=2, pady=4
        )

        ttk.Label(input_frame, text="Quote CSVs Folder:").grid(
            row=1, column=0, sticky=tk.W, pady=4
        )
        ttk.Entry(input_frame, textvariable=self.csv_folder).grid(
            row=1, column=1, sticky=tk.EW, padx=6, pady=4
        )
        ttk.Button(input_frame, text="Browse…", command=self._browse_folder).grid(
            row=1, column=2, pady=4
        )

        # ---- action row ----
        btn_frame = ttk.Frame(self.root, padding=(10, 2))
        btn_frame.pack(fill=tk.X)

        self.run_btn = ttk.Button(
            btn_frame, text="Build Dataset", command=self._start_processing
        )
        self.run_btn.pack(side=tk.LEFT, pady=6)

        self.progress = ttk.Progressbar(btn_frame, mode="indeterminate", length=220)
        self.progress.pack(side=tk.LEFT, padx=12, pady=6)

        # ---- log area ----
        log_frame = ttk.LabelFrame(self.root, text="Log", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.log_text = tk.Text(log_frame, state=tk.DISABLED, wrap=tk.WORD, height=20)
        sb = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    # -------------------------------------------------------- file browsing --

    def _browse_source(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Source File",
            filetypes=[
                ("CSV and Excel files", "*.csv *.xlsx *.xls"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.source_file.set(path)

    def _browse_folder(self) -> None:
        path = filedialog.askdirectory(title="Select Folder of Quote CSVs")
        if path:
            self.csv_folder.set(path)

    # -------------------------------------------------------------- logging --

    def _log(self, msg: str) -> None:
        """Thread-safe: schedules the append on the main thread."""
        self.root.after(0, self._append_log, msg)

    def _append_log(self, msg: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    # ----------------------------------------------------------- processing --

    def _start_processing(self) -> None:
        if not self.source_file.get():
            messagebox.showwarning("Missing Input", "Please select a source file.")
            return
        if not self.csv_folder.get():
            messagebox.showwarning("Missing Input", "Please select a quotes folder.")
            return

        self.run_btn.configure(state=tk.DISABLED)
        self.progress.start(10)
        threading.Thread(target=self._process, daemon=True).start()

    def _process(self) -> None:
        source_path = self.source_file.get()
        folder_path = self.csv_folder.get()

        # ---- load source file ----
        self._log(f"Loading source file: {os.path.basename(source_path)}")
        try:
            ext = os.path.splitext(source_path)[1].lower()
            df = (
                pd.read_excel(source_path)
                if ext in (".xlsx", ".xls")
                else pd.read_csv(source_path)
            )
        except Exception as exc:
            self._log(f"ERROR loading source file: {exc}")
            self._finish(None)
            return

        for required in ("Part_Cleaned", "Quote"):
            if required not in df.columns:
                self._log(f"ERROR: Source file is missing required column '{required}'.")
                self._finish(None)
                return

        # Ensure time columns exist in the output (NaN where no match is found)
        for col in TIME_COLUMNS:
            if col not in df.columns:
                df[col] = None

        # ---- index quote CSVs: filename stem → full path ----
        quote_index: dict[str, str] = {
            os.path.splitext(fname)[0]: os.path.join(folder_path, fname)
            for fname in os.listdir(folder_path)
            if fname.lower().endswith(".csv")
        }
        self._log(f"Found {len(quote_index)} CSV file(s) in the quotes folder.\n")

        # Cache loaded DataFrames so each file is read only once
        csv_cache: dict[str, pd.DataFrame] = {}

        matched = 0
        missing_quotes: set[str] = set()
        no_item_match: list[str] = []

        for idx, row in df.iterrows():
            part = self._clean_id(row["Part_Cleaned"])
            quote = self._clean_id(row["Quote"])

            if not part or not quote:
                continue

            # Find the CSV whose filename stem ends with the quote number
            csv_path = next(
                (path for stem, path in quote_index.items() if stem.endswith(quote)),
                None,
            )

            if csv_path is None:
                if quote not in missing_quotes:
                    self._log(f"  No CSV found for quote '{quote}'.")
                    missing_quotes.add(quote)
                continue

            # Load (or retrieve from cache)
            if csv_path not in csv_cache:
                try:
                    csv_cache[csv_path] = pd.read_csv(csv_path)
                    self._log(f"  Loaded: {os.path.basename(csv_path)}")
                except Exception as exc:
                    self._log(f"  ERROR reading {os.path.basename(csv_path)}: {exc}")
                    csv_cache[csv_path] = pd.DataFrame()  # sentinel – skip on retry

            qdf = csv_cache[csv_path]

            if qdf.empty:
                continue  # error was already logged on first load attempt

            if "ItemNumber" not in qdf.columns:
                self._log(
                    f"  WARNING: 'ItemNumber' column not found in "
                    f"{os.path.basename(csv_path)}."
                )
                continue

            # Substring match: ItemNumber contains the part number.
            # Normalise whitespace first so non-breaking spaces / extra spaces
            # in values like "Solid113- 124855" don't break the search.
            normalised_items = (
                qdf["ItemNumber"]
                .astype(str)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )
            mask = normalised_items.str.contains(part, na=False, regex=False)
            matches = qdf[mask]

            if matches.empty:
                no_item_match.append(f"{part}  (quote {quote})")
                continue

            # Copy time columns from the first matching row
            match_row = matches.iloc[0]
            for col in TIME_COLUMNS:
                if col in qdf.columns:
                    df.at[idx, col] = match_row[col]

            matched += 1

        # ---- summary ----
        self._log(f"\nCompleted: {matched} / {len(df)} rows matched.")

        if missing_quotes:
            self._log(f"\nQuote CSVs not found ({len(missing_quotes)}):")
            for q in sorted(missing_quotes):
                self._log(f"  • {q}")

        if no_item_match:
            self._log(f"\nNo ItemNumber match for {len(no_item_match)} part(s):")
            for p in no_item_match[:30]:
                self._log(f"  • {p}")
            if len(no_item_match) > 30:
                self._log(f"  … and {len(no_item_match) - 30} more.")

        self._finish(df)

    # ---------------------------------------------------------------- utils --

    @staticmethod
    def _clean_id(value) -> str:
        """
        Normalise an ID value that pandas may have read as a float.
        e.g. 3991.0  →  '3991'
        """
        s = str(value).strip()
        if s.lower() in ("nan", "none", ""):
            return ""
        if "." in s:
            try:
                s = str(int(float(s)))
            except ValueError:
                pass
        return s

    def _finish(self, df) -> None:
        """Worker-thread call: dispatch the save dialog back onto the main thread."""
        self.root.after(0, self._do_save, df)

    def _do_save(self, df) -> None:
        self.progress.stop()
        self.run_btn.configure(state=tk.NORMAL)

        if df is None:
            return

        out_path = filedialog.asksaveasfilename(
            title="Save Output File",
            defaultextension=".csv",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx"),
            ],
        )
        if not out_path:
            self._append_log("Save cancelled.")
            return

        try:
            if out_path.lower().endswith(".xlsx"):
                df.to_excel(out_path, index=False)
            else:
                df.to_csv(out_path, index=False)
            self._append_log(f"Saved → {out_path}")
        except Exception as exc:
            self._append_log(f"ERROR saving file: {exc}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = OutsourcingDatasetApp(root)
    root.mainloop()
