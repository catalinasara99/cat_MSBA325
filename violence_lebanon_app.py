import calendar
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Violence in Lebanon 2016-2024", layout="wide")

@st.cache_data
def load_data(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    df["Month"] = df["Month"].astype(str).str.strip().str.title().replace({"Sept": "September"})
    month_map = {m: i for i, m in enumerate(calendar.month_name) if m}
    df["month_num"] = df["Month"].map(month_map)

    df["refPeriod"] = pd.to_numeric(df["refPeriod"], errors="coerce")
    df["Events"] = pd.to_numeric(df["Events"], errors="coerce")
    df["Fatalities"] = pd.to_numeric(df["Fatalities"], errors="coerce")

    df = df.dropna(subset=["refPeriod", "month_num"])
    df["refPeriod"] = df["refPeriod"].astype(int)
    df["month_num"] = df["month_num"].astype(int)
    df["date"] = pd.to_datetime(dict(year=df["refPeriod"], month=df["month_num"], day=1))
    return df

def severity(events, fatals):
    sev = fatals / events
    return sev.mask(events == 0)

DATA_PATH = Path(__file__).with_name("violenceleb_rawdata.csv")
if not DATA_PATH.exists():
    st.error("Missing file: violenceleb_rawdata.csv")
    st.stop()

df = load_data(DATA_PATH)

st.title("Violence in Lebanon 2016-2024")
st.markdown(
    """
### About the data  
This dashboard uses ACLED data on conflict in Lebanon (2016–2024), showing monthly reported events and fatalities.  
"""
)


# Sidebar controls for interactivty
years = sorted(df["refPeriod"].unique())
year_range = st.sidebar.slider("Year range", int(min(years)), int(max(years)), (int(min(years)), int(max(years))))
months_all = [m for m in calendar.month_name if m]
month_filter = st.sidebar.multiselect("Months", months_all, default=months_all)
metric = st.sidebar.radio("Metric", ["Events", "Fatalities", "Severity"], index=0)

# Filtering data
fdf = df[(df["refPeriod"].between(*year_range)) & (df["Month"].isin(month_filter))].copy()
fdf["Severity"] = severity(fdf["Events"], fdf["Fatalities"])

#summary section to understand the numbers based on filters selected 
totals = fdf.agg({"Events": "sum", "Fatalities": "sum"})
sev_total = severity(pd.Series([totals["Events"]]), pd.Series([totals["Fatalities"]])).iloc[0]
c1, c2, c3 = st.columns(3)
c1.metric("Years", f"{year_range[0]}–{year_range[1]}")
c2.metric("Events", f"{int(totals['Events']):,}")
c3.metric("Fatalities | Sev", f"{int(totals['Fatalities']):,} | {sev_total:.2f}" if pd.notna(sev_total) else "—")

st.divider()

#First Visual - Line chart 
st.subheader(f"Trend of {metric} Over Time")
series = fdf.groupby("date", as_index=False).agg(Events=("Events", "sum"), Fatalities=("Fatalities", "sum"))
series["Severity"] = severity(series["Events"], series["Fatalities"])
fig1 = px.line(series, x="date", y=metric, markers=True, labels={"date": "Month", metric: metric})
st.plotly_chart(fig1, use_container_width=True)


#Second Visual - Heat Map
st.subheader(f"Monthly Distribution of {metric} by Year")
ym = fdf.groupby(["refPeriod", "Month", "month_num"], as_index=False).agg(Events=("Events", "sum"), Fatalities=("Fatalities", "sum"))
ym["Severity"] = severity(ym["Events"], ym["Fatalities"])
months_jan_dec = [m for m in calendar.month_name if m]
heat = ym.pivot(index="refPeriod", columns="Month", values=metric).reindex(columns=months_jan_dec).sort_index()
fig2 = px.imshow(heat, aspect="auto", origin="lower", labels=dict(x="Month", y="Year", color=metric), color_continuous_scale="YlOrRd")
st.plotly_chart(fig2, use_container_width=True)

with st.expander("Key Insights"):
    st.markdown(
        """
- The number of events isn’t steady across the year. **Late 2023**, especially in **December**, shows the highest recorded incidents **(830 events)**, which drove totals up sharply.  
- Events and fatalities don’t always move together. In **July 2017**, only **43 events** were recorded. However, they caused **183 fatalities**, showing that impact can outweigh frequency.  
- Other peaks also appear in earlier years, such as **March 2016** and the **summer of 2017**, which likely connect to episodes of conflict and unrest in Lebanon that were happening at the time.   
        """
    )

