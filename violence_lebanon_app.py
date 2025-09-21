# app.py
import calendar
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path
import os

st.set_page_config(page_title="Lebanon Violence Explorer", layout="wide")

@st.cache_data
def load_data(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    req = {"refPeriod", "Month", "Events", "Fatalities"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # clean months
    df["Month"] = (
        df["Month"].astype(str).str.strip().str.title().replace({"Sept": "September"})
    )
    month_map = {m: i for i, m in enumerate(calendar.month_name) if m}
    df["month_num"] = df["Month"].map(month_map)

    # types
    df["refPeriod"]   = pd.to_numeric(df["refPeriod"], errors="coerce")
    df["Events"]      = pd.to_numeric(df["Events"], errors="coerce")
    df["Fatalities"]  = pd.to_numeric(df["Fatalities"], errors="coerce")

    df = df.dropna(subset=["refPeriod", "month_num"])
    df["date"] = pd.to_datetime(
        dict(year=df["refPeriod"].astype(int), month=df["month_num"].astype(int), day=1)
    )

    # vectorized severity
    sev = np.divide(
        df["Fatalities"],
        df["Events"],
        out=np.zeros_like(df["Fatalities"], dtype=float),
        where=df["Events"].to_numpy() != 0,
    )
    df["Severity"] = np.nan_to_num(sev, nan=0.0, posinf=0.0, neginf=0.0)
    return df

# --- load data (relative to this file), with a friendly fallback on Cloud
DATA_PATH = Path(__file__).with_name("violenceleb_rawdata.csv")
try:
    if os.path.exists(DATA_PATH):
        df = load_data(DATA_PATH)
    else:
        st.warning(f"CSV not found at: {DATA_PATH}. Upload the file below or fix the name.")
        uploaded = st.file_uploader("Upload violenceleb_rawdata.csv", type="csv")
        if uploaded is None:
            st.stop()
        df = load_data(uploaded)
except Exception as e:
    st.error(f"❌ Failed to load data: {e}")
    st.stop()

# ===================== UI =====================

st.title("Lebanon Violence Explorer")
st.caption("ACLED/CODEC extract with monthly aggregation: refPeriod (year), Month, Events, Fatalities.")

# sidebar
years = sorted(df["refPeriod"].dropna().astype(int).unique())
year_range = st.sidebar.slider("Year range", min(years), max(years), (min(years), max(years)))
months = [m for m in calendar.month_name if m]  # Jan..Dec
sel_months = st.sidebar.multiselect("Months", months, default=months)
metric = st.sidebar.radio("Metric", ["Events", "Fatalities", "Severity"], horizontal=True)
smooth = st.sidebar.checkbox("3-month moving average (time series)", value=True)

# filter
fdf = df[df["refPeriod"].between(*year_range) & df["Month"].isin(sel_months)].copy()

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Years", f"{year_range[0]}–{year_range[1]}")
c2.metric("Months", str(len(sel_months)))
c3.metric("Events (sum)", f"{int(fdf['Events'].sum()):,}")
c4.metric("Fatalities (sum)", f"{int(fdf['Fatalities'].sum()):,}")

# Chart 1 — trend
st.subheader("Trend over time")
series = (fdf.sort_values("date")
            .groupby("date", as_index=False)
            .agg(Events=("Events", "sum"),
                 Fatalities=("Fatalities", "sum"),
                 Severity=("Severity", "mean")))
y = metric
if smooth and not series.empty:
    series[y] = series[y].rolling(3, min_periods=1).mean()

fig1 = px.line(series, x="date", y=y)
fig1.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Month", yaxis_title=y)
st.plotly_chart(fig1, use_container_width=True)

with st.expander("Notes"):
    st.write(
        "- Time series to see turning points.\n"
        "- Compare Events vs Fatalities vs Severity to separate frequency from deadliness."
    )

# Chart 2 — monthly pattern (heatmap)
st.subheader("Monthly pattern by year")
pivot = (fdf.groupby(["refPeriod", "Month", "month_num"], as_index=False)
            .agg(Events=("Events", "sum"),
                 Fatalities=("Fatalities", "sum"),
                 Severity=("Severity", "mean"))
            .sort_values(["refPeriod", "month_num"]))

heat = (pivot.pivot_table(index="refPeriod", columns="Month", values=metric, aggfunc="mean")
              .reindex(columns=months, fill_value=np.nan))

fig2 = px.imshow(heat, aspect="auto", origin="lower",
                 labels=dict(x="Month", y="Year", color=metric))
fig2.update_layout(margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig2, use_container_width=True)

with st.expander("Notes"):
    st.write(
        "- Heatmap to spot seasonal spikes.\n"
        "- Check if Severity clusters in particular months."
    )

# download
st.download_button(
    "Download filtered CSV",
    data=fdf[["refPeriod", "Month", "Events", "Fatalities", "Severity", "date"]].to_csv(index=False),
    file_name="lebanon_violence_filtered.csv",
    mime="text/csv",
)
