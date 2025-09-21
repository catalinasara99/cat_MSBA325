import calendar
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Lebanon Violence Explorer", layout="wide")

# ----------------------------
# Data loading & cleaning
# ----------------------------
@st.cache_data
def load_data(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    required = {"refPeriod", "Month", "Events", "Fatalities"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Normalize months (e.g., "Sept" -> "September")
    df["Month"] = (
        df["Month"].astype(str).str.strip().str.title().replace({"Sept": "September"})
    )
    month_map = {m: i for i, m in enumerate(calendar.month_name) if m}
    df["month_num"] = df["Month"].map(month_map)

    # Types
    df["refPeriod"] = pd.to_numeric(df["refPeriod"], errors="coerce")
    df["Events"] = pd.to_numeric(df["Events"], errors="coerce")
    df["Fatalities"] = pd.to_numeric(df["Fatalities"], errors="coerce")

    # Keep valid rows
    df = df.dropna(subset=["refPeriod", "month_num"])
    df["refPeriod"] = df["refPeriod"].astype(int)
    df["month_num"] = df["month_num"].astype(int)

    # Monthly date
    df["date"] = pd.to_datetime(
        dict(year=df["refPeriod"], month=df["month_num"], day=1)
    )
    return df


# ----------------------------
# Load data: local file or upload
# ----------------------------
st.sidebar.header("Data")
default_path = Path(__file__).with_name("violenceleb_rawdata.csv")
uploaded = st.sidebar.file_uploader(
    "Upload CSV (columns: refPeriod, Month, Events, Fatalities)", type=["csv"]
)

if uploaded is not None:
    df = load_data(uploaded)
elif default_path.exists():
    df = load_data(default_path)
else:
    st.error("No data found. Please upload a CSV with: refPeriod, Month, Events, Fatalities.")
    st.stop()

# ----------------------------
# Title & caption
# ----------------------------
st.title("Lebanon Violence Explorer")
st.caption("Monthly aggregation from ACLED/CODEC extract: refPeriod (year), Month, Events, Fatalities.")

# ----------------------------
# INTERACTIVITY (exactly two controls)
# ----------------------------
st.sidebar.header("Controls")

years = sorted(df["refPeriod"].unique())
year_range = st.sidebar.slider(
    "Year range",
    min_value=int(min(years)),
    max_value=int(max(years)),
    value=(int(min(years)), int(max(years))),
    step=1,
)

metric = st.sidebar.radio(
    "Metric",
    options=["Events", "Fatalities", "Severity"],
    index=0,
    help="Choose what the charts display."
)

# ----------------------------
# Filter by year range
# ----------------------------
fdf = df.loc[df["refPeriod"].between(*year_range)].copy()
if fdf.empty:
    st.warning("No data in the selected range.")
    st.stop()

# ----------------------------
# Helpers & KPIs
# ----------------------------
def _safe_severity(events: pd.Series, fatals: pd.Series) -> pd.Series:
    sev = fatals / events
    return sev.mask(events == 0)  # NaN where Events==0

totals = fdf.agg({"Events": "sum", "Fatalities": "sum"})
overall_sev = _safe_severity(pd.Series([totals["Events"]]), pd.Series([totals["Fatalities"]])).iloc[0]

c1, c2, c3 = st.columns(3)
c1.metric("Years", f"{year_range[0]}–{year_range[1]}")
c2.metric("Events (sum)", f"{int(totals['Events']):,}")
c3.metric(
    "Fatalities (sum) | Sev",
    f"{int(totals['Fatalities']):,}  |  " + (f"{overall_sev:.2f}" if pd.notna(overall_sev) else "—")
)

st.divider()

# ----------------------------
# Chart 1 — Trend over time (monthly)
# ----------------------------
st.subheader(f"Trend over time ({metric})")

series = (
    fdf.sort_values("date")
       .groupby("date", as_index=False)
       .agg(Events=("Events", "sum"), Fatalities=("Fatalities", "sum"))
)
series["Severity"] = _safe_severity(series["Events"], series["Fatalities"])

fig1 = px.line(
    series,
    x="date",
    y=metric,
    markers=True,
    labels={"date": "Month", metric: metric},
)
fig1.update_layout(margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig1, use_container_width=True)

with st.expander("📌 Key Insights (static)"):
    st.markdown(
        """
- **Late-year concentration**: Within-year distributions are often right-tailed; **Oct–Dec** can dominate totals. Annual comparisons may hide this clustering.
- **Frequency vs. impact**: **Events** and **Fatalities** can **decouple**. High-event months may have low fatality ratios; low-event months can show high **Severity**. Read them together to avoid base-rate errors.
- **Severity definition**: Computed as `sum(Fatalities) / sum(Events)` per period. Months with `Events = 0` are **undefined (NaN)**, not zero—this avoids understating risk in sparse periods.
- **Seasonality vs. shocks**: The heatmap distinguishes repeated seasonal patterns from one-off shocks that drive specific months.
- **Window sensitivity**: Changing the **Year range** can shift conclusions (e.g., late-2023 spikes dominate short windows). Always state the analysis window.
        """
    )

# ----------------------------
# Chart 2 — Monthly pattern by year (heatmap)
# ----------------------------
st.subheader(f"Monthly pattern by year ({metric})")

ym = (
    fdf.groupby(["refPeriod", "Month", "month_num"], as_index=False)
       .agg(Events=("Events", "sum"), Fatalities=("Fatalities", "sum"))
)
ym["Severity"] = _safe_severity(ym["Events"], ym["Fatalities"])

months_jan_dec = [m for m in calendar.month_name if m]  # Jan..Dec
heat = (
    ym.pivot(index="refPeriod", columns="Month", values=metric)
      .reindex(columns=months_jan_dec)
      .sort_index()
)

fig2 = px.imshow(
    heat,
    aspect="auto",
    origin="lower",
    labels=dict(x="Month", y="Year", color=metric),
    color_continuous_scale="YlOrRd"  # warm heatmap: yellow→orange→red
)
fig2.update_layout(margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig2, use_container_width=True)

with st.expander("📝 Methods & Choices"):
    st.markdown(
        """
**Metric calculation**
- **Events/Fatalities** are **summed** per period.
- **Severity** computed **after** aggregation: `sum(Fatalities) / sum(Events)`.
- Periods with `Events = 0` → **Severity = NaN**.

**Design choices**
- Exactly two interactions: **Year range** and **Metric**.
- Line chart for temporal dynamics; YlOrRd heatmap for intuitive **“hotter = higher”** seasonal patterns.
        """
    )

# ----------------------------
# Data preview (no download)
# ----------------------------
st.divider()
st.subheader("Filtered data (monthly)")
preview = fdf[["refPeriod", "Month", "Events", "Fatalities", "date", "month_num"]].copy()
preview["Severity"] = _safe_severity(preview["Events"], preview["Fatalities"])
st.dataframe(preview.sort_values(["refPeriod", "month_num"]).drop(columns=["month_num"]),
             use_container_width=True)
