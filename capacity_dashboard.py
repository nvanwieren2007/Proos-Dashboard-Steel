"""
Manufacturing Capacity Dashboard
=================================
Two companies (Proos / MS), 7 operations, 52-week calendar view.

Data sources
------------
• Consumed Capacity sheet  — one row per (Company, Year, Week)
  Columns: Company, Year, Week, Laser, Press_Brake, Weld,
           Resistance_Weld, Powder_Coat, Assembly, Packing

• Capacity Limits sheet    — one row per Company
  Columns: Company, Laser, Press_Brake, Weld,
           Resistance_Weld, Powder_Coat, Assembly, Packing

Both sheets are loaded from Google Sheets "Publish to web → CSV" URLs,
or from local CSV file uploads.

To launch
---------
    cd c:\\Users\\nvanwieren\\TestingPython\\VScode_test
    streamlit run capacity_dashboard.py
"""

import os
import base64
import requests as _requests
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

COMPANIES = ["Proos", "MS"]

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

OP_LABELS = {
    "Laser":           "Laser",
    "Press_Brake":     "Press Brake",
    "Weld":            "Weld",
    "Resistance_Weld": "Resistance Weld",
    "Powder_Coat":     "Powder Coat",
    "Assembly":        "Assembly",
    "Packing":         "Packing",
    "PEM":             "PEM",
}

COLORS = {
    "Proos":         "#1f77b4",   # blue
    "MS":            "#ff7f0e",   # orange
    "combined_line": "#2ca02c",   # green
    "true_cap":      "#9467bd",   # purple
}

# Operations to hide per view.  Comment out individual entries to re-enable.
_PROOS_HIDDEN_OPS = {
    "Powder_Coat",
    "Resistance_Weld",
    "Packing",
    "PEM",
}
_MS_HIDDEN_OPS = {
    "Powder_Coat",
    "Resistance_Weld",
    "Packing",
    "PEM",
    "Assembly",
}
_COMBINED_HIDDEN_OPS = {
    "Powder_Coat",
    "Resistance_Weld",
    "Packing",
    "PEM",
}

_NO_DATA_ANNOTATION = dict(
    text="No consumed capacity for current / future weeks",
    xref="paper", yref="paper",
    x=0.5, y=0.5, showarrow=False,
    font=dict(size=12, color="#aaaaaa"),
)

# ─────────────────────────────────────────────────────────────────────────────
# MS consumed data — GitHub-backed persistence
# ─────────────────────────────────────────────────────────────────────────────

_MS_DATA_FILE = Path(__file__).parent / "ms_consumed_data.csv"
_MS_GITHUB_PATH = "ms_consumed_data.csv"


def _github_headers() -> dict:
    token = st.secrets.get("github_token", "")
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
    }


def _save_ms_data_to_github(csv_bytes: bytes) -> tuple:
    """
    Commit ms_consumed_data.csv to the GitHub repo via the GitHub Contents API.
    Returns (success: bool, message: str).
    Requires 'github_token' and 'github_repo' in secrets.
    """
    token = st.secrets.get("github_token", "")
    repo  = st.secrets.get("github_repo", "")
    if not token or not repo:
        return False, "Add `github_token` and `github_repo` to Streamlit secrets to enable cloud persistence."

    api_url = f"https://api.github.com/repos/{repo}/contents/{_MS_GITHUB_PATH}"
    headers = _github_headers()

    # Fetch current SHA so we can overwrite the file (required by GitHub API)
    sha = None
    r = _requests.get(api_url, headers=headers, timeout=10)
    if r.status_code == 200:
        sha = r.json().get("sha")

    payload = {
        "message": f"Update MS consumed data ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})",
        "content": base64.b64encode(csv_bytes).decode(),
    }
    if sha:
        payload["sha"] = sha

    r = _requests.put(api_url, headers=headers, json=payload, timeout=15)
    if r.status_code in (200, 201):
        _MS_DATA_FILE.write_bytes(csv_bytes)   # also keep a local copy
        return True, "Saved to GitHub — data will persist across server restarts."
    return False, f"GitHub API returned {r.status_code}: {r.json().get('message', r.text[:120])}"


@st.cache_data(ttl=60)
def _load_ms_data_from_github() -> pd.DataFrame:
    """Read ms_consumed_data.csv from the GitHub raw URL (cached 60 s)."""
    repo = st.secrets.get("github_repo", "")
    if not repo:
        return pd.DataFrame()
    url = f"https://raw.githubusercontent.com/{repo}/main/{_MS_GITHUB_PATH}"
    try:
        return pd.read_csv(url)
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def current_iso_week() -> int:
    return datetime.now().isocalendar()[1]


def current_iso_year() -> int:
    return datetime.now().year


@st.cache_data(ttl=300)
def load_csv_url(url: str) -> pd.DataFrame:
    return pd.read_csv(url)


# Expected columns in every per-operation Google Sheet tab
_DETAIL_COLS = [
    "Company", "Year", "Week", "Source",
    "Job Number", "Part Number", "Quantity",
    "Production Rate", "Setup Time", "Hours Remaining",
]


@st.cache_data(ttl=300)
def load_consumed_per_op_urls(op_urls: dict) -> pd.DataFrame:
    """
    Fetch one CSV per operation tab.  Each tab is expected to have:
      Company, Year, Week, Source, Job Number, Part Number,
      Quantity, Production Rate, Setup Time, Hours Remaining

    Returns a long-format DataFrame with an extra 'Operation' column.
    One row per job line.  Aggregation to wide format happens in the
    caller so all detail is preserved for display.
    """
    frames = []
    for op, url in op_urls.items():
        if not url:
            continue
        try:
            df = load_csv_url(url)
        except Exception as e:
            st.warning(f"Could not load {OP_LABELS.get(op, op)} tab: {e}")
            continue

        # Validate required columns — Company and Year are optional;
        # if missing we fill in sensible defaults.
        required = ("Week", "Hours Remaining")
        missing  = [c for c in required if c not in df.columns]
        if missing:
            st.warning(
                f"{OP_LABELS.get(op, op)} tab is missing columns: "
                f"{', '.join(missing)} — skipped."
            )
            continue

        # Normalize Company: absent → "MS"; any variant containing "MS" → "MS"
        if "Company" not in df.columns:
            df["Company"] = "MS"
        else:
            df["Company"] = df["Company"].apply(
                lambda v: "MS" if isinstance(v, str) and "MS" in v.upper() else v
            )

        # Default Year to current year if absent
        if "Year" not in df.columns:
            df["Year"] = current_iso_year()

        # Keep all expected detail columns that exist in the sheet
        keep = [c for c in _DETAIL_COLS if c in df.columns]
        df   = df[keep].copy()
        df["Operation"] = op
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=_DETAIL_COLS + ["Operation"])

    return pd.concat(frames, ignore_index=True)


def coerce_numerics(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


def active_rows(
    consumed: pd.DataFrame,
    operation: str,
    current_week: int,
    current_year: int,
    company: str = None,
) -> pd.DataFrame:
    """
    Return rows where:
      • week >= current_week  (respecting Year if present)
      • the operation column has consumed hours > 0
      • optionally filtered to a single company
    """
    df = consumed.copy()
    if company:
        df = df[df["Company"] == company]
    if "Year" in df.columns:
        df = df[
            (df["Year"] > current_year)
            | ((df["Year"] == current_year) & (df["Week"] >= current_week))
        ]
    else:
        df = df[df["Week"] >= current_week]
    if operation in df.columns:
        df = df[df[operation] > 0]
    return df


def get_cap(
    capacity: pd.DataFrame,
    company: str,
    operation: str,
    week: int = None,
    year: int = None,
) -> float:
    """
    Look up capacity for a company+operation.
    If the limits sheet has Year+Week columns, first try to find a
    week-specific override row; fall back to the company default
    (the row where Year and Week are blank / 0).
    """
    df = capacity[capacity["Company"] == company]
    if df.empty or operation not in df.columns:
        return 0.0

    has_week_col = "Week" in df.columns and "Year" in df.columns

    if has_week_col and week is not None and year is not None:
        # Try exact week+year override first
        override = df[
            (df["Year"].fillna(0).astype(int) == year) &
            (df["Week"].fillna(0).astype(int) == week)
        ]
        if not override.empty:
            return float(override[operation].values[0])

    # Fall back to the default row (Year and Week blank / 0)
    if has_week_col:
        default = df[
            (df["Year"].fillna(0).astype(int) == 0) |
            (df["Week"].fillna(0).astype(int) == 0)
        ]
        if not default.empty:
            return float(default[operation].values[0])

    # Legacy single-row-per-company format
    return float(df[operation].values[0])


def build_cap_series(
    capacity: pd.DataFrame,
    company: str,
    operation: str,
    weeks: list,
    year: int,
) -> list:
    """Return one capacity ceiling per week, respecting weekly overrides."""
    return [get_cap(capacity, company, operation, week=w, year=year) for w in weeks]


def _to_step_series(weeks: list, vals: list):
    """
    Convert parallel (weeks, vals) lists into (xs, ys) for a crisp step-line
    where each value spans exactly one bar width (week − 0.5 → week + 0.5).
    A vertical transition is inserted at the shared boundary between two weeks
    that have different values, so holiday dips snap back on the very next week
    with zero bleed-over.
    """
    if not weeks:
        return [], []
    xs, ys = [], []
    for i, (w, cap) in enumerate(zip(weeks, vals)):
        left  = w - 0.5
        right = w + 0.5
        if i == 0:
            xs += [left, right]
            ys += [cap,  cap ]
        else:
            prev = vals[i - 1]
            if cap != prev:
                # Two points at the same x = vertical edge at the boundary
                xs += [left, left, right]
                ys += [prev, cap,  cap ]
            else:
                xs.append(right)
                ys.append(cap)
    return xs, ys


# ─────────────────────────────────────────────────────────────────────────────
# Chart builder
# ─────────────────────────────────────────────────────────────────────────────

def build_chart(
    consumed: pd.DataFrame,
    capacity: pd.DataFrame,
    operation: str,
    view: str,
    current_week: int,
    current_year: int,
) -> go.Figure:
    fig = go.Figure()
    op_label = OP_LABELS[operation]
    # MS graphs always show the next 30 weeks; others show at least 10
    min_end_week = current_week + 29 if view == "MS Capacity" else current_week + 9

    # ── Single-company views ──────────────────────────────────────────────────
    if view in ("Proos Capacity", "MS Capacity"):
        company = "Proos" if view == "Proos Capacity" else "MS"
        color   = COLORS[company]

        df = active_rows(consumed, operation, current_week, current_year, company)

        if df.empty or operation not in df.columns:
            # No consumption data — draw capacity line and zero bars over the full window
            cap_weeks = list(range(current_week, min_end_week + 1))
            cap_vals  = build_cap_series(capacity, company, operation, cap_weeks, current_year)
            if any(v > 0 for v in cap_vals):
                cap_xs, cap_ys = _to_step_series(cap_weeks, cap_vals)
                fig.add_trace(go.Scatter(
                    x=cap_xs, y=cap_ys,
                    mode="lines",
                    name=f"{company} capacity",
                    line=dict(color=color, dash="dash", width=2.5),
                    showlegend=True,
                ))
            else:
                fig.add_annotation(**_NO_DATA_ANNOTATION)
        else:
            agg = df.groupby("Week")[operation].sum().sort_index()

            # For MS, always span the full 30-week window with zero-filled bars
            # so empty future weeks appear explicitly on the chart.
            cap_weeks = list(range(current_week, max(max(agg.index.tolist()), min_end_week) + 1))
            bar_vals  = [float(agg.get(w, 0)) for w in cap_weeks]

            fig.add_trace(go.Bar(
                x=cap_weeks, y=bar_vals,
                name=f"{company} consumed",
                marker_color=color, opacity=0.85,
            ))

            cap_vals  = build_cap_series(capacity, company, operation, cap_weeks, current_year)
            if any(v > 0 for v in cap_vals):
                cap_xs, cap_ys = _to_step_series(cap_weeks, cap_vals)
                fig.add_trace(go.Scatter(
                    x=cap_xs, y=cap_ys,
                    mode="lines",
                    name=f"{company} capacity",
                    line=dict(color=color, dash="dash", width=2.5),
                    showlegend=True,
                ))

    # ── Combined (stacked) view ───────────────────────────────────────────────
    else:
        fig.update_layout(barmode="stack")

        all_df    = active_rows(consumed, operation, current_week, current_year)
        all_weeks = sorted(all_df["Week"].unique().tolist())

        if not all_weeks:
            # No consumption data — still draw capacity lines over the next 10 weeks
            cap_weeks     = list(range(current_week, min_end_week + 1))
            combined_vals = [0.0] * len(cap_weeks)
            has_any_cap   = False
            for company in COMPANIES:
                cap_vals = build_cap_series(
                    capacity, company, operation, cap_weeks, current_year
                )
                if any(v > 0 for v in cap_vals):
                    has_any_cap = True
                    cap_xs, cap_ys = _to_step_series(cap_weeks, cap_vals)
                    fig.add_trace(go.Scatter(
                        x=cap_xs, y=cap_ys,
                        mode="lines",
                        name=f"{company} cap.",
                        line=dict(color=COLORS[company], dash="dash", width=2.0),
                        showlegend=True,
                    ))
                combined_vals = [c + v for c, v in zip(combined_vals, cap_vals)]
            if any(v > 0 for v in combined_vals):
                comb_xs, comb_ys = _to_step_series(cap_weeks, combined_vals)
                fig.add_trace(go.Scatter(
                    x=comb_xs, y=comb_ys,
                    mode="lines",
                    name="Combined cap.",
                    line=dict(color=COLORS["combined_line"], dash="dashdot", width=2.5),
                    showlegend=True,
                ))
            # True Capacity line
            true_cap_vals = build_cap_series(
                capacity, "True Cap", operation, cap_weeks, current_year
            )
            if any(v > 0 for v in true_cap_vals):
                tc_xs, tc_ys = _to_step_series(cap_weeks, true_cap_vals)
                fig.add_trace(go.Scatter(
                    x=tc_xs, y=tc_ys,
                    mode="lines",
                    name="True Capacity",
                    line=dict(color=COLORS["true_cap"], dash="dash", width=2.5),
                    showlegend=True,
                ))
            if not has_any_cap:
                fig.add_annotation(**_NO_DATA_ANNOTATION)
        else:
            for company in COMPANIES:
                color = COLORS[company]
                df    = active_rows(consumed, operation, current_week, current_year, company)
                if not df.empty and operation in df.columns:
                    agg = df.groupby("Week")[operation].sum()
                    wv  = agg.to_dict()
                else:
                    wv  = {}
                vals = [float(wv.get(w, 0)) for w in all_weeks]

                fig.add_trace(go.Bar(
                    x=all_weeks, y=vals,
                    name=company,
                    marker_color=color, opacity=0.85,
                ))

            # Build capacity step lines across the full week range, extended
            # to at least min_end_week so holiday dips and future weeks show.
            cap_weeks     = list(range(min(all_weeks), max(max(all_weeks), min_end_week) + 1))
            combined_vals = [0.0] * len(cap_weeks)
            for company in COMPANIES:
                cap_vals = build_cap_series(
                    capacity, company, operation, cap_weeks, current_year
                )
                if any(v > 0 for v in cap_vals):
                    cap_xs, cap_ys = _to_step_series(cap_weeks, cap_vals)
                    fig.add_trace(go.Scatter(
                        x=cap_xs, y=cap_ys,
                        mode="lines",
                        name=f"{company} cap.",
                        line=dict(color=COLORS[company], dash="dash", width=2.0),
                        showlegend=True,
                    ))
                combined_vals = [c + v for c, v in zip(combined_vals, cap_vals)]

            if any(v > 0 for v in combined_vals):
                comb_xs, comb_ys = _to_step_series(cap_weeks, combined_vals)
                fig.add_trace(go.Scatter(
                    x=comb_xs, y=comb_ys,
                    mode="lines",
                    name="Combined cap.",
                    line=dict(color=COLORS["combined_line"], dash="dashdot", width=2.5),
                    showlegend=True,
                ))
            # True Capacity line
            true_cap_vals = build_cap_series(
                capacity, "True Cap", operation, cap_weeks, current_year
            )
            if any(v > 0 for v in true_cap_vals):
                tc_xs, tc_ys = _to_step_series(cap_weeks, true_cap_vals)
                fig.add_trace(go.Scatter(
                    x=tc_xs, y=tc_ys,
                    mode="lines",
                    name="True Capacity",
                    line=dict(color=COLORS["true_cap"], dash="dash", width=2.5),
                    showlegend=True,
                ))

    fig.update_layout(
        title=dict(text=f"<b>{op_label}</b>", font=dict(size=14)),
        xaxis=dict(
            title="Manufacturing Week",
            tickmode="linear", tick0=1, dtick=1,
            showgrid=True, gridcolor="#eeeeee",
        ),
        yaxis=dict(
            title="Hours",
            showgrid=True, gridcolor="#eeeeee",
            rangemode="tozero",
        ),
        height=320,
        margin=dict(l=50, r=20, t=50, b=45),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=10),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be the first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

_favicon_path = Path(__file__).parent / "ProosPLogo.jpg"

st.set_page_config(
    page_title="Manufacturing Capacity Dashboard",
    page_icon=str(_favicon_path) if _favicon_path.exists() else "🏭",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, width='stretch')
    else:
        st.markdown("## 🏭 Capacity Dashboard")

    st.markdown("---")

    view = st.selectbox(
        "View",
        ["Proos Capacity", "MS Capacity", "Combined Capacity"],
        index=2,
        key="cap_view",
        help="Select which company's capacity to display, or view both stacked.",
    )

    st.markdown("---")
    st.markdown("**Data Sources**")
    st.caption(
        "Paste a Google Sheets *Publish to web → CSV* URL, "
        "or upload local CSV files.  \n"
        "*(File → Share → Publish to web → select sheet → CSV)*"
    )

    source_mode = st.radio(
        "Source",
        ["Google Sheet URL", "Upload CSV"],
        horizontal=True,
        label_visibility="collapsed",
        key="cap_source_mode",
    )

    consumed_url = capacity_url = None
    consumed_file = capacity_file = None
    consumed_op_urls = {}   # MS per-operation URL dict

    if source_mode == "Google Sheet URL":
        # ── Proos: single wide-format URL ─────────────────────────────────────
        st.markdown("**Proos Consumed Capacity**")
        _secret_consumed = st.secrets.get("capacity_consumed_url", "")
        consumed_url = st.text_input(
            "Proos Consumed URL",
            value=_secret_consumed,
            key="cap_consumed_url_proos",
            placeholder="https://docs.google.com/spreadsheets/…/pub?output=csv",
        )

        st.markdown("---")
        # ── MS: one URL per operation tab ─────────────────────────────────────
        st.markdown("**MS Consumed Capacity** *(per-operation tabs)*")
        st.caption(
            "Paste a *Publish to web → CSV* URL for each MS operation tab.  "
            "Leave blank to skip that operation."
        )
        _any_per_op = False
        for op in OPERATIONS:
            _secret_key = f"consumed_url_{op}"
            _default    = st.secrets.get(_secret_key, "")
            _url = st.text_input(
                OP_LABELS[op],
                value=_default,
                key=f"cap_consumed_url_{op}",
                placeholder="…/pub?gid=…&output=csv",
                label_visibility="visible",
            )
            consumed_op_urls[op] = _url
            if _url:
                _any_per_op = True

        st.markdown("---")
        st.markdown("**Capacity Limits**")
        _secret_limits = st.secrets.get("capacity_limits_url", "")
        capacity_url = st.text_input(
            "Capacity Limits URL",
            value=_secret_limits,
            key="cap_capacity_url",
            placeholder="https://docs.google.com/spreadsheets/…/pub?output=csv",
        )

        if (consumed_url or _any_per_op) and _secret_limits:
            st.caption("✅ URLs loaded from secrets.toml")
    else:
        consumed_file = st.file_uploader(
            "Consumed Capacity CSV", type="csv", key="cap_consumed_file"
        )
        capacity_file = st.file_uploader(
            "Capacity Limits CSV", type="csv", key="cap_capacity_file"
        )

    st.markdown("---")
    st.markdown("**📥 MS Consumed Data**")
    st.caption(
        "Upload a CSV with the standard consumed format "
        "(Company=MS, Year, Week, + operation columns). "
        "Replaces all MS rows and saves to GitHub automatically."
    )
    ms_upload = st.file_uploader(
        "MS Consumed CSV", type="csv", key="ms_data_upload",
        help="Must have columns: Company, Year, Week, and one or more operation columns.",
    )
    if ms_upload is not None:
        try:
            _uploaded_df = pd.read_csv(ms_upload)
            _missing = [c for c in ("Company", "Year", "Week") if c not in _uploaded_df.columns]
            if _missing:
                st.error(f"CSV is missing required columns: {', '.join(_missing)}")
            else:
                _csv_bytes = ms_upload.getvalue()
                _ok, _msg  = _save_ms_data_to_github(_csv_bytes)
                if _ok:
                    st.success(_msg)
                    st.cache_data.clear()
                    st.rerun()
                else:
                    # No GitHub credentials — save locally only
                    _MS_DATA_FILE.write_bytes(_csv_bytes)
                    st.warning(f"Saved locally only (this session).  \n{_msg}")
                    st.cache_data.clear()
                    st.rerun()
        except Exception as _e:
            st.error(f"Could not read CSV: {_e}")

    # Status line
    _has_github = bool(st.secrets.get("github_repo", ""))
    if _MS_DATA_FILE.exists():
        _mtime = datetime.fromtimestamp(_MS_DATA_FILE.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        st.caption(f"✅ MS data file present (updated {_mtime})")
        if st.button("🗑️ Remove MS data", key="ms_data_clear"):
            _MS_DATA_FILE.unlink(missing_ok=True)
            if _has_github:
                # Delete the file from GitHub too
                try:
                    _api = f"https://api.github.com/repos/{st.secrets['github_repo']}/contents/{_MS_GITHUB_PATH}"
                    _r   = _requests.get(_api, headers=_github_headers(), timeout=10)
                    if _r.status_code == 200:
                        _sha = _r.json().get("sha")
                        _requests.delete(_api, headers=_github_headers(), timeout=10,
                            json={"message": "Remove MS consumed data", "sha": _sha})
                except Exception:
                    pass
            st.cache_data.clear()
            st.rerun()
    elif not _has_github:
        st.caption("ℹ️ No MS data loaded. Upload a CSV above.")

    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    cw = current_iso_week()
    cy = current_iso_year()
    st.markdown("---")
    st.caption(f"Current ISO week: **{cw}** / {cy}")

# ─────────────────────────────────────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────────────────────────────────────

st.title("🏭 Manufacturing Capacity Dashboard")
st.caption(
    f"Displaying weeks **{cw}–52** ({cy}) where capacity is consumed.  "
    "Past weeks and future zero-consumption weeks are hidden automatically."
)

_script_dir     = os.path.dirname(os.path.abspath(__file__))
_local_consumed = os.path.join(_script_dir, "capacity_consumed_template.csv")
_local_limits   = os.path.join(_script_dir, "capacity_limits_template.csv")

_has_per_op_urls = any(v for v in consumed_op_urls.values())
has_url_data     = (bool(consumed_url) or _has_per_op_urls) and bool(capacity_url)
has_file_data    = (consumed_file is not None) and (capacity_file is not None)
has_local_data   = os.path.exists(_local_consumed) and os.path.exists(_local_limits)

if not has_url_data and not has_file_data and not has_local_data:
    st.info(
        "👈 Connect your data in the sidebar — paste Google Sheets CSV URLs or upload local files.",
        icon="ℹ️",
    )
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────

_job_detail: pd.DataFrame = pd.DataFrame()  # preserved for the detail expander

try:
    if has_url_data:
        frames = []

        # Proos: single wide-format URL
        if consumed_url:
            frames.append(load_csv_url(consumed_url))

        # MS: per-operation tabs → aggregate to wide
        if _has_per_op_urls:
            _job_detail = load_consumed_per_op_urls(consumed_op_urls)
            _job_detail["Hours Remaining"] = pd.to_numeric(
                _job_detail["Hours Remaining"], errors="coerce"
            ).fillna(0)
            if not _job_detail.empty:
                _agg = (
                    _job_detail
                    .groupby(["Company", "Year", "Week", "Operation"])["Hours Remaining"]
                    .sum()
                    .reset_index()
                )
                _wide = _agg.pivot_table(
                    index=["Company", "Year", "Week"],
                    columns="Operation",
                    values="Hours Remaining",
                    aggfunc="sum",
                    fill_value=0,
                ).reset_index()
                _wide.columns.name = None
                for op in OPERATIONS:
                    if op not in _wide.columns:
                        _wide[op] = 0.0
                frames.append(_wide[["Company", "Year", "Week"] + OPERATIONS])

        if frames:
            consumed = pd.concat(frames, ignore_index=True)
        else:
            consumed = pd.DataFrame(columns=["Company", "Year", "Week"] + OPERATIONS)

        capacity = load_csv_url(capacity_url)
    elif has_file_data:
        consumed = pd.read_csv(consumed_file)
        capacity = pd.read_csv(capacity_file)
    else:  # local CSV files in the same directory
        consumed = pd.read_csv(_local_consumed)
        capacity = pd.read_csv(_local_limits)
except Exception as exc:
    st.error(f"Failed to load data: {exc}")
    st.stop()

# ── Sum duplicate (Company, Year, Week) rows in consumed ─────────────────────
# Wide-format fallback paths: sum any split rows so charts are always correct.

_num_cols = [c for c in OPERATIONS if c in consumed.columns]
if _num_cols and not consumed.empty:
    _key_cols = [c for c in ("Company", "Year", "Week") if c in consumed.columns]
    _extra    = [c for c in consumed.columns if c not in _key_cols + _num_cols]
    consumed[_num_cols] = consumed[_num_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    consumed = (
        consumed
        .groupby(_key_cols, as_index=False)[_num_cols]
        .sum()
    )

# ── Validate columns ──────────────────────────────────────────────────────────

for col in ("Company", "Week"):
    if col not in consumed.columns:
        st.error(f"Consumed sheet is missing required column: **{col}**")
        st.stop()

if "Company" not in capacity.columns:
    st.error("Capacity Limits sheet is missing required column: **Company**")
    st.stop()

# ── Coerce types ──────────────────────────────────────────────────────────────

consumed = coerce_numerics(consumed, OPERATIONS + ["Week", "Year"])
capacity = coerce_numerics(capacity, OPERATIONS)
consumed["Week"] = consumed["Week"].astype(int)
if "Year" in consumed.columns:
    consumed["Year"] = consumed["Year"].astype(int)

# ── Overlay MS consumed data from file / GitHub ──────────────────────────────
# Only apply when MS is NOT being loaded live from per-operation tab URLs.
# If per-op URLs are active, the live data is already in `consumed` and the
# old uploaded file would overwrite it with stale data.

if not _has_per_op_urls:
    _ms_override = None

    # 1. Local file (fastest; written on upload and after GitHub save)
    if _MS_DATA_FILE.exists():
        try:
            _ms_override = pd.read_csv(_MS_DATA_FILE)
        except Exception:
            pass

    # 2. GitHub raw URL fallback (works on cloud after a fresh deploy)
    if _ms_override is None or _ms_override.empty:
        _ms_override = _load_ms_data_from_github()

    if _ms_override is not None and not _ms_override.empty:
        _ms_rows = _ms_override[_ms_override["Company"] == "MS"]
        if not _ms_rows.empty:
            consumed = pd.concat(
                [consumed[consumed["Company"] != "MS"], _ms_rows],
                ignore_index=True,
            )
            consumed = coerce_numerics(consumed, OPERATIONS + ["Week", "Year"])
            consumed["Week"] = consumed["Week"].astype(int)
            if "Year" in consumed.columns:
                consumed["Year"] = consumed["Year"].astype(int)

# ── Warn if a company is missing from the selected view ──────────────────────

if view == "Proos Capacity" and "Proos" not in consumed["Company"].values:
    st.warning("No Proos rows found in the consumed capacity sheet.")
elif view == "MS Capacity" and "MS" not in consumed["Company"].values:
    st.warning("No MS rows found in the consumed capacity sheet.")

# ── Render charts in a 2-column grid ─────────────────────────────────────────

_hidden_ops = (
    _PROOS_HIDDEN_OPS if view == "Proos Capacity"
    else _MS_HIDDEN_OPS if view == "MS Capacity"
    else _COMBINED_HIDDEN_OPS
)
_ops_to_show = [op for op in OPERATIONS if op not in _hidden_ops]

for i in range(0, len(_ops_to_show), 2):
    pair = _ops_to_show[i : i + 2]
    cols = st.columns(len(pair))
    for j, op in enumerate(pair):
        with cols[j]:
            st.plotly_chart(
                build_chart(consumed, capacity, op, view, cw, cy),
                width='stretch',
            )

# ── Raw data expanders ────────────────────────────────────────────────────────

if not _job_detail.empty:
    with st.expander("🔍 Job Detail — all open operations", expanded=False):
        _display_detail = _job_detail.copy()
        # Coerce mixed-type columns to string so Arrow serialization doesn't fail
        for _col in ("Job Number", "Part Number", "Source"):
            if _col in _display_detail.columns:
                _display_detail[_col] = _display_detail[_col].astype(str)
        # Show human-readable operation label
        _display_detail.insert(
            _display_detail.columns.get_loc("Operation") + 1,
            "Operation Label",
            _display_detail["Operation"].map(OP_LABELS).fillna(_display_detail["Operation"]),
        )
        # Active (current+future) rows only
        if "Year" in _display_detail.columns and "Week" in _display_detail.columns:
            _display_detail["Year"] = pd.to_numeric(_display_detail["Year"], errors="coerce")
            _display_detail["Week"] = pd.to_numeric(_display_detail["Week"], errors="coerce")
            _display_detail = _display_detail[
                (_display_detail["Year"] > cy)
                | ((_display_detail["Year"] == cy) & (_display_detail["Week"] >= cw))
            ]
        _display_detail = _display_detail.sort_values(
            ["Operation", "Company", "Year", "Week"]
        ).reset_index(drop=True)
        st.dataframe(_display_detail, width="stretch")

with st.expander("📋 Consumed Capacity — aggregated", expanded=False):
    st.dataframe(consumed, width='stretch')

with st.expander("📋 Capacity Limits — raw data", expanded=False):
    st.dataframe(capacity, width='stretch')
