import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, date
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Proos Commodity Pricing Dashboard",
    page_icon="📊",
    layout="wide",
)

# ── Password gate ─────────────────────────────────────────────────────────────
def _check_password():
    if st.session_state.get("authenticated"):
        return
    st.markdown("## \U0001f4ca Proos Commodity Pricing Dashboard")
    st.markdown("Please enter the password to continue.")
    pwd = st.text_input("Password", type="password", key="_pwd_input")
    if st.button("Login"):
        expected = st.secrets.get("password", "PROOS")
        if pwd == expected:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password. Please try again.")
    st.stop()

_check_password()

# ── File paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
CRU_FILE = BASE_DIR / "cru_prices.csv"
STAINLESS_FILE = BASE_DIR / "stainless_surcharges.csv"
FUTURES_FILE = BASE_DIR / "cru_futures.csv"
LOGO_FILE = BASE_DIR / "logo.png"

# ── Logo ──────────────────────────────────────────────────────────────────────
if LOGO_FILE.exists():
    st.sidebar.image(str(LOGO_FILE), use_container_width=True)


# ── Data seeders (first-run sample data) ─────────────────────────────────────
def _seed_cru():
    rng = np.random.default_rng(42)
    weeks = pd.date_range(end=pd.Timestamp.today(), periods=12, freq="W-FRI")
    steel = [round(980 + i * 8 + rng.integers(-5, 6), 2) for i in range(12)]
    galv = [round(s + 105 + rng.integers(-6, 7), 2) for s in steel]
    df = pd.DataFrame({
        "date": weeks,
        "steel_price": steel,
        "galvanized_price": galv,
        "avg_steel_price": None,
        "avg_galv_price": None,
        "avg_ss_price": None,
    })
    df.to_csv(CRU_FILE, index=False)
    return df


def _seed_stainless():
    """Create an empty stainless surcharge file with correct headers."""
    df = pd.DataFrame(
        columns=["month", "surcharge_304", "surcharge_316", "producer", "notes"]
    )
    df.to_csv(STAINLESS_FILE, index=False)
    return df


def _seed_futures():
    """Create an empty CRU HRC futures file with correct headers."""
    df = pd.DataFrame(columns=["month", "settle_price"])
    df.to_csv(FUTURES_FILE, index=False)
    return df


# ── Loaders ───────────────────────────────────────────────────────────────────
def load_cru():
    if CRU_FILE.exists():
        df = pd.read_csv(CRU_FILE, parse_dates=["date"])
        # Add new columns if missing (backward compat)
        for col in ["avg_steel_price", "avg_galv_price", "avg_ss_price"]:
            if col not in df.columns:
                df[col] = None
        return df.sort_values("date").reset_index(drop=True)
    return _seed_cru()


def load_stainless():
    if STAINLESS_FILE.exists():
        df = pd.read_csv(STAINLESS_FILE, parse_dates=["month"])
        return df.sort_values("month").reset_index(drop=True)
    return _seed_stainless()


def load_futures():
    """Load futures CSV and auto-prune to current month + 11 forward (12 rows max)."""
    if FUTURES_FILE.exists():
        df = pd.read_csv(FUTURES_FILE, parse_dates=["month"])
        df = df.sort_values("month").reset_index(drop=True)
    else:
        df = _seed_futures()
    current_month = pd.Timestamp.today().normalize().replace(day=1)
    max_month = current_month + pd.DateOffset(months=11)
    pruned = df[(df["month"] >= current_month) & (df["month"] <= max_month)].reset_index(drop=True)
    if len(pruned) != len(df):
        pruned.to_csv(FUTURES_FILE, index=False)
    return pruned


@st.cache_data(ttl=3600)
def get_live_prices(period: str):
    """Fetch Aluminum futures via yfinance. Nickel (NI=F) is delisted on Yahoo Finance."""
    tickers = {"Aluminum": "ALI=F"}
    out = {}
    for name, sym in tickers.items():
        try:
            hist = yf.Ticker(sym).history(period=period)
            if not hist.empty:
                out[name] = hist["Close"].dropna()
        except Exception:
            pass
    return out


# ── Prediction model ─────────────────────────────────────────────────────────
UNIT_DIVISORS = {"$/cwt": 100.0, "$/short ton": 2000.0, "$/metric ton": 2204.62}


def build_price_prediction(cru_df, ss_df, al_series, price_unit, forecast_months=6, futures_df=None):
    """
    Train Ridge regression models for actual prices paid ($/lb) per commodity.
    Targets: avg_steel_price, avg_galv_price, avg_ss_price.
    All prices are 1st-of-month snapshots; steel pricing updated monthly.
    """
    divisor = UNIT_DIVISORS[price_unit]
    df = cru_df.copy().sort_values("date").reset_index(drop=True)
    df["cru_steel_lb"] = df["steel_price"] / divisor
    df["cru_galv_lb"] = df["galvanized_price"] / divisor
    df["galv_premium"] = df["cru_galv_lb"] - df["cru_steel_lb"]
    df["t"] = (df["date"] - df["date"].min()).dt.days.astype(float)

    # Align aluminum futures monthly
    has_al = False
    if al_series is not None and len(al_series) > 0:
        al_copy = al_series.copy()
        if getattr(al_copy.index, "tz", None) is not None:
            al_copy.index = al_copy.index.tz_localize(None)
        al_df = al_copy.resample("MS").last().ffill().reset_index()
        al_df.columns = ["date", "al_price"]
        df = pd.merge_asof(
            df.sort_values("date"), al_df.sort_values("date"),
            on="date", direction="nearest", tolerance=pd.Timedelta("35D"),
        )
        has_al = df["al_price"].notna().sum() > len(df) * 0.5
        if has_al:
            df["al_price"] = df["al_price"].ffill().bfill()

    # Align SS 304 surcharge monthly
    has_ss_charge = False
    if ss_df is not None and not ss_df.empty:
        ss_copy = ss_df[["month", "surcharge_304"]].copy().sort_values("month")
        df = pd.merge_asof(
            df.sort_values("date"), ss_copy.sort_values("month"),
            left_on="date", right_on="month", direction="nearest",
            tolerance=pd.Timedelta("35D"),
        )
        has_ss_charge = df["surcharge_304"].notna().sum() > len(df) * 0.5
        if has_ss_charge:
            df["surcharge_304"] = df["surcharge_304"].ffill().bfill()

    config = [
        ("avg_steel_price", "HRC Sheet Steel", "#1f77b4",
         ["t", "cru_steel_lb", "galv_premium"] + (["al_price"] if has_al else [])),
        ("avg_galv_price", "Galvanized Sheet", "#ff7f0e",
         ["t", "cru_galv_lb", "cru_steel_lb"]),
        ("avg_ss_price", "304 Stainless", "#d62728",
         ["t"] + (["surcharge_304"] if has_ss_charge else [])),
    ]

    all_feat_cols = ["t", "cru_steel_lb", "cru_galv_lb", "galv_premium"]
    if has_al:
        all_feat_cols.append("al_price")
    if has_ss_charge:
        all_feat_cols.append("surcharge_304")

    last_t = float(df["t"].max())
    last_cru_steel = float(df["cru_steel_lb"].iloc[-1])
    last_cru_galv = float(df["cru_galv_lb"].iloc[-1])
    last_galv_premium = float(df["galv_premium"].tail(3).mean())
    last_al = float(df["al_price"].iloc[-1]) if has_al else None
    last_ss_charge = float(df["surcharge_304"].iloc[-1]) if has_ss_charge else None

    future_dates = pd.date_range(
        start=df["date"].max() + pd.DateOffset(months=1),
        periods=forecast_months, freq="MS",
    )

    # Build CME HRC futures lookup {delivery_month_timestamp: settle_price in $/lb}
    futures_lookup = {}
    if futures_df is not None and not futures_df.empty:
        for _, frow in futures_df.iterrows():
            if pd.notna(frow["month"]) and pd.notna(frow["settle_price"]):
                key = pd.Timestamp(frow["month"]).replace(day=1).normalize()
                futures_lookup[key] = float(frow["settle_price"]) / divisor

    results = {}
    for target_col, label, color, features in config:
        sub = df[all_feat_cols + [target_col, "date"]].copy()
        sub[target_col] = pd.to_numeric(sub[target_col], errors="coerce")
        sub = sub.dropna(subset=[target_col]).reset_index(drop=True)

        if len(sub) < 4:
            results[target_col] = None
            continue

        X = sub[features].values
        y = sub[target_col].astype(float).values
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(X, y)
        y_pred_hist = model.predict(X)
        rmse = float(np.sqrt(np.mean((y - y_pred_hist) ** 2)))
        ci_half = 1.96 * rmse

        future_rows = []
        futures_months_used = 0
        for i, fd in enumerate(future_dates, 1):
            fd_key = pd.Timestamp(fd).normalize()
            fhrc = futures_lookup.get(fd_key)
            row = {"date": fd, "t": last_t + i * 30.44}
            if "cru_steel_lb" in features:
                row["cru_steel_lb"] = fhrc if fhrc is not None else last_cru_steel
                if fhrc is not None:
                    futures_months_used += 1
            if "cru_galv_lb" in features:
                # Derive forward galv from HRC futures + last known galv premium
                row["cru_galv_lb"] = (fhrc + last_galv_premium) if fhrc is not None else last_cru_galv
            if "galv_premium" in features: row["galv_premium"] = last_galv_premium
            if "al_price" in features: row["al_price"] = last_al
            if "surcharge_304" in features: row["surcharge_304"] = last_ss_charge
            future_rows.append(row)

        future_df = pd.DataFrame(future_rows)
        y_future = model.predict(future_df[features].values)
        future_df["pred"] = y_future
        future_df["lower"] = y_future - ci_half
        future_df["upper"] = y_future + ci_half

        results[target_col] = {
            "label": label,
            "color": color,
            "features": features,
            "history_dates": sub["date"].values,
            "history_actual": y,
            "y_pred_hist": y_pred_hist,
            "forecast": future_df[["date", "pred", "lower", "upper"]],
            "rmse": rmse,
            "n_train": len(sub),
            "has_al": has_al,
            "has_ss_charge": has_ss_charge,
            "futures_months_used": futures_months_used,
        }

    return results


# ── Load data ─────────────────────────────────────────────────────────────────
cru_df = load_cru()
ss_df = load_stainless()
futures_df = load_futures()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    price_unit = "$/short ton"
    st.markdown("**CRU Price Unit:** $/short ton")
    live_period = st.selectbox(
        "Live Market History",
        ["1mo", "3mo", "6mo", "1y", "2y"],
        index=2,
    )
    st.markdown("---")
    st.markdown("**Data Sources**")
    st.markdown(
        "- \U0001f535 CRU Americas Weekly Index *(manual)*\n"
        "- \U0001f7e1 CME HRC Futures Settle *(manual)*\n"
        "- \U0001f7e2 NAS / ATI Alloy Surcharges *(manual)*\n"
        "- \U0001f534 Yahoo Finance \u2014 Aluminum & Nickel *(live)*"
    )
    if st.button("🔄 Refresh Live Data"):
        st.cache_data.clear()
        st.rerun()

# ── Live market data ──────────────────────────────────────────────────────────
live_data = get_live_prices(period=live_period)

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("📊 Proos Commodity Pricing Dashboard")
st.caption(f"Last refreshed: {datetime.now().strftime('%B %d, %Y  %I:%M %p')}")

tab_overview, tab_charts, tab_entry, tab_predict = st.tabs(
    ["📈 Overview", "📉 Historical Charts", "✏️ Data Entry", "🔮 Price Prediction"]
)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.subheader("Latest Prices")
    c1, c2, c3, c4, c5 = st.columns(5)

    # HRC Steel (CRU)
    if not cru_df.empty:
        v = cru_df["steel_price"].iloc[-1]
        d = (v - cru_df["steel_price"].iloc[-2]) if len(cru_df) > 1 else 0
        c1.metric(f"HRC Steel ({price_unit})", f"${v:,.2f}", f"{d:+.2f} WoW")
    else:
        c1.metric(f"HRC Steel ({price_unit})", "—")

    # Galvanized (CRU)
    if not cru_df.empty:
        v = cru_df["galvanized_price"].iloc[-1]
        d = (v - cru_df["galvanized_price"].iloc[-2]) if len(cru_df) > 1 else 0
        c2.metric(f"Galvanized ({price_unit})", f"${v:,.2f}", f"{d:+.2f} WoW")
    else:
        c2.metric(f"Galvanized ({price_unit})", "—")

    # Stainless 304 surcharge
    if not ss_df.empty:
        v = ss_df["surcharge_304"].iloc[-1]
        d = (v - ss_df["surcharge_304"].iloc[-2]) if len(ss_df) > 1 else 0
        c3.metric("304 SS Surcharge ($/lb)", f"${v:.4f}", f"{d:+.4f} MoM")
    else:
        c3.metric("304 SS Surcharge ($/lb)", "—")

    # Stainless 316 surcharge
    if not ss_df.empty:
        v = ss_df["surcharge_316"].iloc[-1]
        d = (v - ss_df["surcharge_316"].iloc[-2]) if len(ss_df) > 1 else 0
        c4.metric("316 SS Surcharge ($/lb)", f"${v:.4f}", f"{d:+.4f} MoM")
    else:
        c4.metric("316 SS Surcharge ($/lb)", "—")

    # Aluminum (live)
    al_series = live_data.get("Aluminum")
    if al_series is not None and len(al_series) > 1:
        v = float(al_series.iloc[-1])
        p = ((v - float(al_series.iloc[-2])) / float(al_series.iloc[-2])) * 100
        c5.metric("Aluminum (live, $/lb)", f"${v:,.4f}", f"{p:+.2f}% DoD")
    else:
        c5.metric("Aluminum (live, $/lb)", "No data")

    st.markdown("---")
    c6, c7, _ = st.columns(3)

    # Galvanized premium
    if not cru_df.empty and len(cru_df) > 1:
        prem = cru_df["galvanized_price"].iloc[-1] - cru_df["steel_price"].iloc[-1]
        c6.metric(f"Galv. Premium over HRC ({price_unit})", f"${prem:,.2f}")

    # Nickel — note: NI=F is delisted on Yahoo Finance; use LME or manual entry
    c7.metric(
        "Nickel — SS Cost Indicator",
        "See lme.com",
        help="LME Nickel is not available via Yahoo Finance. Check lme.com for daily prices.",
    )

    st.markdown("---")
    st.subheader("Recent CRU Entries")
    if not cru_df.empty:
        disp = cru_df.copy().iloc[::-1]
        disp["date"] = disp["date"].dt.strftime("%Y-%m-%d")
        disp.columns = [
            "Date",
            f"HRC Steel ({price_unit})",
            f"Galvanized ({price_unit})",
            "Avg Steel ($/lb)",
            "Avg Galv ($/lb)",
            "Avg SS ($/lb)",
        ]
        st.dataframe(disp, width="stretch", hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — HISTORICAL CHARTS
# ═════════════════════════════════════════════════════════════════════════════
with tab_charts:
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### CRU Weekly — HRC Steel & Galvanized")
        if len(cru_df) > 1:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=cru_df["date"],
                    y=cru_df["steel_price"],
                    name="HRC Steel",
                    line=dict(color="#1f77b4", width=2),
                    mode="lines+markers",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=cru_df["date"],
                    y=cru_df["galvanized_price"],
                    name="Galvanized",
                    line=dict(color="#ff7f0e", width=2),
                    mode="lines+markers",
                )
            )
            fig.update_layout(
                yaxis_title=price_unit,
                xaxis_title="Week Ending",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                height=360,
                margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Add at least 2 CRU entries to display this chart.")

    with col_r:
        st.markdown("#### Stainless Alloy Surcharges (Monthly)")
        if len(ss_df) > 1:
            fig2 = go.Figure()
            fig2.add_trace(
                go.Scatter(
                    x=ss_df["month"],
                    y=ss_df["surcharge_304"],
                    name="Grade 304",
                    line=dict(color="#2ca02c", width=2),
                    mode="lines+markers",
                )
            )
            fig2.add_trace(
                go.Scatter(
                    x=ss_df["month"],
                    y=ss_df["surcharge_316"],
                    name="Grade 316",
                    line=dict(color="#d62728", width=2),
                    mode="lines+markers",
                )
            )
            fig2.update_layout(
                yaxis_title="$/lb",
                xaxis_title="Month",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                height=360,
                margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig2, width="stretch")
        else:
            st.info("Add at least 2 stainless surcharge entries to display this chart.")

    # Live market indicator charts
    st.markdown("#### Live Market Indicators (Yahoo Finance)")
    st.caption("ℹ️ Nickel (NI=F) is no longer listed on Yahoo Finance. Check [lme.com](https://www.lme.com) for LME Nickel prices.")
    if live_data:
        n = len(live_data)
        fig3 = make_subplots(
            rows=1,
            cols=n,
            subplot_titles=[f"{k} ($/lb)" for k in live_data],
        )
        colors = ["#1f77b4", "#d62728"]
        for i, (name, series) in enumerate(live_data.items(), start=1):
            fig3.add_trace(
                go.Scatter(
                    x=series.index,
                    y=series.values,
                    name=name,
                    mode="lines",
                    line=dict(color=colors[i - 1], width=1.5),
                ),
                row=1,
                col=i,
            )
        fig3.update_layout(
            height=300,
            showlegend=False,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig3, width="stretch")
    else:
        st.warning(
            "Live market data unavailable. Yahoo Finance may be rate-limiting or "
            "the Aluminum futures ticker (ALI=F) may have changed."
        )

    # Galvanized premium chart
    st.markdown("#### Galvanized Premium Over HRC Steel")
    if len(cru_df) > 1:
        chart_df = cru_df.copy()
        chart_df["premium"] = chart_df["galvanized_price"] - chart_df["steel_price"]
        fig4 = px.area(
            chart_df,
            x="date",
            y="premium",
            labels={"premium": f"Premium ({price_unit})", "date": "Week Ending"},
            color_discrete_sequence=["#9467bd"],
        )
        fig4.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig4, width="stretch")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — DATA ENTRY
# ═════════════════════════════════════════════════════════════════════════════
with tab_entry:
    col_entry_l, col_entry_r = st.columns(2)

    # ── Monthly CRU Entry ─────────────────────────────────────────────────────
    with col_entry_l:
        st.subheader("Monthly CRU Entry")
        st.caption("Prices are a snapshot taken on the 1st of the month.")
        with st.form("cru_form", clear_on_submit=True):
            entry_date = st.date_input("Date (1st of month)", value=date.today().replace(day=1))
            steel_val = st.number_input(
                f"HRC Steel ({price_unit})", min_value=0.0, step=0.01, format="%.2f"
            )
            galv_val = st.number_input(
                f"Galvanized ({price_unit})", min_value=0.0, step=0.01, format="%.2f"
            )
            st.markdown("**Actual Prices Paid ($/lb)**")
            avg_steel = st.number_input(
                "Avg Steel Price ($/lb)", min_value=0.0, step=0.0001, format="%.4f",
                help="Actual avg $/lb paid for standard CRU sheet steel this month"
            )
            avg_galv = st.number_input(
                "Avg Galv Price ($/lb)", min_value=0.0, step=0.0001, format="%.4f",
                help="Actual avg $/lb paid for galvanized sheet steel this month"
            )
            avg_ss = st.number_input(
                "Avg SS Price ($/lb)", min_value=0.0, step=0.0001, format="%.4f",
                help="Actual avg $/lb paid for 304 stainless steel this month"
            )
            if st.form_submit_button("➕ Add CRU Entry", type="primary"):
                new_row = pd.DataFrame(
                    {
                        "date": [pd.to_datetime(entry_date)],
                        "steel_price": [steel_val],
                        "galvanized_price": [galv_val],
                        "avg_steel_price": [avg_steel if avg_steel > 0 else None],
                        "avg_galv_price": [avg_galv if avg_galv > 0 else None],
                        "avg_ss_price": [avg_ss if avg_ss > 0 else None],
                    }
                )
                updated = cru_df[cru_df["date"] != pd.to_datetime(entry_date)]
                updated = (
                    pd.concat([updated, new_row])
                    .sort_values("date")
                    .reset_index(drop=True)
                )
                updated.to_csv(CRU_FILE, index=False)
                st.success(f"Added CRU entry for {entry_date.strftime('%B %d, %Y')}")
                st.rerun()

        st.markdown("**All CRU Entries**")
        if not cru_df.empty:
            disp_cru = cru_df.copy().iloc[::-1]
            disp_cru["date"] = disp_cru["date"].dt.strftime("%Y-%m-%d")
            disp_cru.columns = [
                "Date",
                f"HRC Steel ({price_unit})",
                f"Galvanized ({price_unit})",
                "Avg Steel ($/lb)",
                "Avg Galv ($/lb)",
                "Avg SS ($/lb)",
            ]
            st.dataframe(disp_cru, width="stretch", hide_index=True)

            if st.button("🗑️ Delete Most Recent CRU Entry", key="del_cru"):
                cru_df.iloc[:-1].to_csv(CRU_FILE, index=False)
                st.warning("Most recent CRU entry deleted.")
                st.rerun()

    # ── Monthly Stainless Surcharge Entry ─────────────────────────────────────
    with col_entry_r:
        st.subheader("Monthly Stainless Alloy Surcharge")
        with st.form("ss_form", clear_on_submit=True):
            ss_month = st.date_input(
                "Month",
                value=date.today().replace(day=1),
                help="Select any date within the target month",
            )
            val_304 = st.number_input(
                "Grade 304 Surcharge ($/lb)",
                min_value=0.0,
                step=0.0001,
                format="%.4f",
            )
            val_316 = st.number_input(
                "Grade 316 Surcharge ($/lb)",
                min_value=0.0,
                step=0.0001,
                format="%.4f",
            )
            producer = st.selectbox("Source", ["NAS", "ATI", "Outokumpu", "Other"])
            notes = st.text_input("Notes (optional)")
            if st.form_submit_button("➕ Add Stainless Entry", type="primary"):
                month_key = pd.to_datetime(ss_month.replace(day=1))
                new_ss = pd.DataFrame(
                    {
                        "month": [month_key],
                        "surcharge_304": [val_304],
                        "surcharge_316": [val_316],
                        "producer": [producer],
                        "notes": [notes],
                    }
                )
                updated_ss = ss_df[ss_df["month"] != month_key]
                updated_ss = (
                    pd.concat([updated_ss, new_ss])
                    .sort_values("month")
                    .reset_index(drop=True)
                )
                updated_ss.to_csv(STAINLESS_FILE, index=False)
                st.success(f"Added stainless entry for {month_key.strftime('%B %Y')}")
                st.rerun()

        st.markdown("**All Stainless Entries**")
        if not ss_df.empty:
            disp_ss = ss_df.copy().iloc[::-1]
            disp_ss["month"] = disp_ss["month"].dt.strftime("%B %Y")
            disp_ss.columns = ["Month", "304 ($/lb)", "316 ($/lb)", "Producer", "Notes"]
            st.dataframe(disp_ss, width="stretch", hide_index=True)

            if st.button("🗑️ Delete Most Recent Stainless Entry", key="del_ss"):
                ss_df.iloc[:-1].to_csv(STAINLESS_FILE, index=False)
                st.warning("Most recent stainless entry deleted.")
                st.rerun()

    # ── CRU HRC Futures Settle Prices ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("CRU HRC Steel Futures Settle Prices")
    st.caption(
        "U.S. Midwest Domestic Hot-Rolled Coil Steel (CRU) Index Futures — CME. "
        "Upload a CSV with your settle prices for the current month and up to 11 forward months. "
        "Months before the current month are automatically removed. "
        "These replace the flat CRU extrapolation in the Price Prediction model."
    )

    col_fut_l, col_fut_r = st.columns(2)

    with col_fut_l:
        st.markdown("**Upload Futures CSV**")

        # Template download — pre-filled with current + 11 forward months
        _tpl_months = pd.date_range(
            start=pd.Timestamp.today().normalize().replace(day=1),
            periods=12, freq="MS",
        )
        _tpl_df = pd.DataFrame({
            "month": _tpl_months.strftime("%Y-%m-%d"),
            "settle_price": [""] * 12,
        })
        st.download_button(
            "⬇️ Download Template CSV",
            _tpl_df.to_csv(index=False),
            file_name="cru_futures_template.csv",
            mime="text/csv",
            help="Pre-filled with the correct 12 delivery months. Fill in settle_price and upload.",
        )

        uploaded_futures = st.file_uploader(
            "Upload filled CSV",
            type=["csv"],
            key="futures_upload",
            help="Required columns: 'month' (YYYY-MM-DD) and 'settle_price' ($/short ton)",
        )
        if uploaded_futures is not None:
            try:
                up_df = pd.read_csv(uploaded_futures)
                # Flexible column name matching
                up_df.columns = [c.strip().lower().replace(" ", "_") for c in up_df.columns]
                if "month" not in up_df.columns or "settle_price" not in up_df.columns:
                    st.error("CSV must have columns named 'month' and 'settle_price'.")
                else:
                    up_df["month"] = pd.to_datetime(up_df["month"], infer_datetime_format=True)
                    up_df["settle_price"] = pd.to_numeric(up_df["settle_price"], errors="coerce")
                    up_df = up_df.dropna(subset=["month", "settle_price"])[["month", "settle_price"]]
                    # Merge: uploaded rows override existing rows for the same month
                    existing = futures_df.copy()
                    merged = pd.concat([existing, up_df]).drop_duplicates(subset="month", keep="last")
                    current_month = pd.Timestamp.today().normalize().replace(day=1)
                    max_month = current_month + pd.DateOffset(months=11)
                    merged = merged[
                        (merged["month"] >= current_month) & (merged["month"] <= max_month)
                    ].sort_values("month").reset_index(drop=True)
                    merged.to_csv(FUTURES_FILE, index=False)
                    st.success(f"Imported {len(merged)} futures rows. Old months auto-removed.")
                    st.rerun()
            except Exception as _e:
                st.error(f"Failed to parse CSV: {_e}")

        with st.expander("✏️ Add or correct a single month"):
            with st.form("futures_form", clear_on_submit=True):
                fut_month = st.date_input(
                    "Delivery Month",
                    value=date.today().replace(day=1),
                    help="Select any date in the target delivery month",
                )
                fut_settle = st.number_input(
                    "Settle Price ($/short ton)",
                    min_value=0.0,
                    step=0.01,
                    format="%.2f",
                )
                if st.form_submit_button("➕ Save Entry", type="primary"):
                    month_key = pd.to_datetime(fut_month.replace(day=1))
                    new_fut = pd.DataFrame({"month": [month_key], "settle_price": [fut_settle]})
                    updated_fut = futures_df[futures_df["month"] != month_key]
                    updated_fut = (
                        pd.concat([updated_fut, new_fut])
                        .sort_values("month")
                        .reset_index(drop=True)
                    )
                    updated_fut.to_csv(FUTURES_FILE, index=False)
                    st.success(f"Saved {month_key.strftime('%B %Y')}: ${fut_settle:,.2f}/short ton")
                    st.rerun()

    with col_fut_r:
        st.markdown("**Current Futures Data** (current month + 11 forward, auto-pruned)")
        if not futures_df.empty:
            disp_fut = futures_df.copy()
            disp_fut["month"] = disp_fut["month"].dt.strftime("%B %Y")
            disp_fut.columns = ["Delivery Month", "Settle Price ($/short ton)"]
            st.dataframe(disp_fut, hide_index=True, use_container_width=True)
            if st.button("🗑️ Clear All Futures Data", key="del_fut"):
                _empty_fut = pd.DataFrame(columns=["month", "settle_price"])
                _empty_fut.to_csv(FUTURES_FILE, index=False)
                st.warning("All futures data cleared.")
                st.rerun()
        else:
            st.info("No futures data yet. Download the template, fill in settle prices, and upload.")
            disp_months = pd.date_range(
                start=pd.Timestamp.today().normalize().replace(day=1),
                periods=12, freq="MS",
            ).strftime("%B %Y").tolist()
            st.markdown("**Expected months:** " + ", ".join(disp_months))


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — PRICE PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.subheader("🔮 Actual Price Prediction ($/lb)")
    st.caption(
        "Ridge regression trained on your actual prices paid (1st-of-month snapshots). "
        "Each commodity has its own model. "
        "Confidence band = ±1.96 × in-sample RMSE (approximate 95% prediction interval)."
    )

    has_actual = cru_df[["avg_steel_price", "avg_galv_price", "avg_ss_price"]].notna().any().any()
    if not has_actual or len(cru_df) < 4:
        st.warning(
            "Need at least **4 months of actual price data** (Avg Steel, Avg Galv, Avg SS) "
            "in the ✏️ Data Entry tab to build the prediction models."
        )
    else:
        forecast_months = st.slider("Forecast Horizon (months)", min_value=2, max_value=12, value=6, step=1)

        pred_results = build_price_prediction(
            cru_df, ss_df, live_data.get("Aluminum"), price_unit, forecast_months,
            futures_df=futures_df,
        )

        sub_steel, sub_galv, sub_ss = st.tabs(
            ["HRC Sheet Steel", "Galvanized Sheet", "304 Stainless"]
        )

        def render_prediction_tab(container, result, commodity_label):
            with container:
                if result is None:
                    st.info(f"Not enough actual price data for {commodity_label} to build a model.")
                    return

                hist_dates = pd.to_datetime(result["history_dates"])
                hist_actual = result["history_actual"]
                fc = result["forecast"]
                color = result["color"]

                latest = float(hist_actual[-1])
                next_pred = float(fc["pred"].iloc[0])

                m1, m2, m3 = st.columns(3)
                m1.metric("Latest Actual ($/lb)", f"${latest:.4f}")
                m2.metric("Next Month Forecast ($/lb)", f"${next_pred:.4f}", f"{next_pred - latest:+.4f}")
                m3.metric("Model RMSE ($/lb)", f"${result['rmse']:.4f}",
                          f"{result['n_train']} training months", delta_color="off")

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=hist_dates, y=hist_actual,
                    mode="lines+markers", name="Actual Price Paid ($/lb)",
                    line=dict(color=color, width=2), marker=dict(size=7),
                ))
                fig.add_trace(go.Scatter(
                    x=hist_dates, y=result["y_pred_hist"],
                    mode="lines", name="Model Fit",
                    line=dict(color="gray", width=1.5, dash="dash"),
                ))
                fig.add_trace(go.Scatter(
                    x=pd.concat([fc["date"], fc["date"].iloc[::-1]]),
                    y=pd.concat([fc["upper"], fc["lower"].iloc[::-1]]),
                    fill="toself", fillcolor="rgba(44,160,44,0.15)",
                    line=dict(color="rgba(0,0,0,0)"), name="95% Confidence Band",
                ))
                fig.add_trace(go.Scatter(
                    x=fc["date"], y=fc["pred"],
                    mode="lines+markers", name="Forecast ($/lb)",
                    line=dict(color="#2ca02c", width=2, dash="dot"),
                    marker=dict(size=8, symbol="diamond"),
                ))
                fig.add_shape(
                    type="line",
                    x0=hist_dates.max().isoformat(), x1=hist_dates.max().isoformat(),
                    y0=0, y1=1, xref="x", yref="paper",
                    line=dict(dash="dot", color="gray", width=1.5),
                )
                fig.add_annotation(
                    x=hist_dates.max().isoformat(), y=1,
                    xref="x", yref="paper", text="Latest",
                    showarrow=False, xanchor="left", yanchor="top",
                    font=dict(color="gray", size=11),
                )
                fig.update_layout(
                    yaxis_title="$/lb", xaxis_title="Month", height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    margin=dict(l=0, r=0, t=30, b=0),
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("**Forecast Table**")
                fc_disp = fc.copy()
                fc_disp["date"] = fc_disp["date"].dt.strftime("%Y-%m-%d")
                fc_disp.columns = ["Month", "Forecast ($/lb)", "Lower 95% CI", "Upper 95% CI"]
                st.dataframe(fc_disp.round(4), hide_index=True, use_container_width=True)

                with st.expander("ℹ️ Model Details"):
                    st.markdown(f"""
**Algorithm:** Ridge Regression (L2 regularization, α=1.0) with StandardScaler  
**Target:** {commodity_label} actual price paid ($/lb)  
**Features used:** {', '.join(result['features'])}  
**Aluminum in steel model:** {'Yes — matched from Yahoo Finance ALI=F (monthly)' if result['has_al'] else 'No — insufficient overlap with CRU dates'}  
**SS 304 surcharge in stainless model:** {'Yes — from your Stainless Surcharge entries (1st-of-month snapshot; 304 grade only used in regression, 316 visible in Historical Charts)' if result['has_ss_charge'] else 'No — enter surcharge data in the ✏️ Data Entry tab'}  
**Training samples:** {result['n_train']} months  
**In-sample RMSE:** ${result['rmse']:.4f}/lb  
**Data cadence:** Prices are 1st-of-month snapshots representing market conditions at that point in time. Steel pricing is updated monthly.  
**CRU HRC Futures:** {str(result['futures_months_used']) + ' of ' + str(len(result['forecast'])) + ' forecast months use actual CME settle prices (remaining months use last known CRU)' if result['futures_months_used'] > 0 else 'None entered — add CME HRC futures settle prices in ✏️ Data Entry to replace flat CRU extrapolation'}  
**Forecast assumptions:** CRU index uses CME HRC futures settle prices where available, otherwise held at last known value; galvanized premium and surcharge held at last known values; time trend projected forward.  
**CI method:** ±1.96 × RMSE (assumes residuals are approximately normal).
                    """)

        render_prediction_tab(sub_steel, pred_results.get("avg_steel_price"), "HRC Sheet Steel")
        render_prediction_tab(sub_galv, pred_results.get("avg_galv_price"), "Galvanized Sheet")
        render_prediction_tab(sub_ss, pred_results.get("avg_ss_price"), "304 Stainless")

