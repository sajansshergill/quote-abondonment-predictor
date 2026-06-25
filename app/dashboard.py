"""
dashboard.py
------------
Streamlit dashboard: funnel health + abandonment risk + intervention ROI.

Run: streamlit run app/dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Funnel Analytics",
    page_icon="📊",
    layout="wide",
)

st.title("🚗 Insurance Quote Funnel Analytics")
st.caption("Jerry.ai-style growth analytics · Built by Sajan Shergill")

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "funnel_events.csv"
SCORED_PATH = ROOT / "data" / "scored_sessions.csv"

STEP_ORDER = ["zip", "vehicle", "driver", "quotes", "bind"]
AVG_COMMISSION = 45

# ── Load data ─────────────────────────────────────────────────────────────────
def ensure_funnel_data() -> None:
    if DATA_PATH.exists():
        return

    st.info("Generating demo funnel dataset for first launch. This takes a few seconds.")
    sys.path.insert(0, str(ROOT))
    from data.simulate_funnel import main as simulate_funnel

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    simulate_funnel()

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_scored():
    if SCORED_PATH.exists():
        return pd.read_csv(SCORED_PATH)
    return None

ensure_funnel_data()
df = load_data()
scored = load_scored()

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.header("Filters")
channels = st.sidebar.multiselect("Channel", df["channel"].unique().tolist(), default=df["channel"].unique().tolist())
devices  = st.sidebar.multiselect("Device",  df["device"].unique().tolist(),  default=df["device"].unique().tolist())

filtered = df[df["channel"].isin(channels) & df["device"].isin(devices)]

# ── KPI row ───────────────────────────────────────────────────────────────────
total_sessions = filtered["session_id"].nunique()
bound = filtered[(filtered["step"] == "bind") & (filtered["dropped"] == False)]["session_id"].nunique()
bind_rate = bound / total_sessions if total_sessions else 0
lost_rev   = (total_sessions - bound) * AVG_COMMISSION

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Sessions",    f"{total_sessions:,}")
k2.metric("Bound Policies",    f"{bound:,}")
k3.metric("Bind Rate",         f"{bind_rate:.2%}")
k4.metric("Est. Lost Revenue", f"${lost_rev:,.0f}", delta=f"-${lost_rev:,.0f}", delta_color="inverse")

st.divider()

# ── Funnel waterfall ──────────────────────────────────────────────────────────
st.subheader("Funnel: Step-Level Conversion")

funnel_data = (
    filtered.groupby("step")
    .agg(reached=("session_id", "count"), dropped=("dropped", "sum"))
    .reindex(STEP_ORDER)
    .fillna(0)
    .assign(pass_through=lambda x: x["reached"] - x["dropped"])
    .assign(drop_rate=lambda x: (x["dropped"] / x["reached"] * 100).round(1))
)

fig_funnel = go.Figure(go.Funnel(
    y=STEP_ORDER,
    x=funnel_data["reached"].values,
    textinfo="value+percent initial",
    marker=dict(color=["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]),
))
fig_funnel.update_layout(height=380, margin=dict(l=20, r=20, t=20, b=20))
st.plotly_chart(fig_funnel, width="stretch")

# Drop rate table
st.dataframe(
    funnel_data[["reached", "dropped", "drop_rate"]]
    .rename(columns={"reached": "Sessions Reached", "dropped": "Dropped", "drop_rate": "Drop Rate %"}),
    width="stretch",
)

st.divider()

# ── Drop rate heatmap: channel × device ───────────────────────────────────────
st.subheader("Abandonment Heatmap: Channel × Device (Quotes Step)")

heatmap_df = (
    filtered[filtered["step"] == "quotes"]
    .groupby(["channel", "device"])["dropped"]
    .mean()
    .unstack("device")
    .round(3) * 100
)

fig_heat = px.imshow(
    heatmap_df,
    text_auto=".1f",
    color_continuous_scale="Reds",
    labels=dict(color="Drop Rate %"),
    title="Drop Rate % at Quotes Step",
    height=350,
)
st.plotly_chart(fig_heat, width="stretch")

st.divider()

# ── Abandonment model score distribution ─────────────────────────────────────
if scored is not None:
    st.subheader("Abandonment Risk: Model Score Distribution")

    threshold = st.slider("Risk threshold (flag sessions above this)", 0.3, 0.9, 0.65, 0.05)
    high_risk = scored[scored["abandon_prob"] >= threshold]
    st.caption(f"**{len(high_risk):,}** sessions above threshold ({threshold}) — estimated intervention value: **${len(high_risk) * AVG_COMMISSION * 0.12:,.0f}**")

    fig_hist = px.histogram(
        scored, x="abandon_prob", nbins=40,
        color_discrete_sequence=["#4C72B0"],
        labels={"abandon_prob": "P(Abandon)"},
        title="Distribution of Abandonment Probability Scores",
        height=320,
    )
    fig_hist.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text=f"Threshold: {threshold}")
    st.plotly_chart(fig_hist, width="stretch")
    st.divider()

# ── Intervention ROI panel ────────────────────────────────────────────────────
st.subheader("Intervention ROI Estimator")

def cohort_row(intervention: str, cohort_size: int, lift_rate: float) -> dict:
    new_conversions = round(cohort_size * lift_rate)
    return {
        "intervention": intervention,
        "cohort_size": cohort_size,
        "lift_rate": lift_rate,
        "new_conversions": new_conversions,
        "revenue_lift_usd": round(new_conversions * AVG_COMMISSION),
    }

quotes_drop = filtered[(filtered["step"] == "quotes") & (filtered["dropped"] == True)]
reengage = filtered[
    (filtered["device"] == "mobile")
    & (filtered["step"].isin(["driver", "quotes"]))
    & (filtered["dropped"] == True)
]

cohorts = pd.DataFrame(
    [
        cohort_row(
            "Early Price Reveal",
            len(quotes_drop[quotes_drop["cheapest_quote_usd"] < 120]),
            0.15,
        ),
        cohort_row(
            "Quote Count Cap",
            len(quotes_drop[quotes_drop["quote_count"] >= 5]),
            0.12,
        ),
        cohort_row("Re-engagement Nudge", len(reengage), 0.08),
    ]
).sort_values("revenue_lift_usd", ascending=False)

c1, c2 = st.columns([2, 1])
with c1:
    fig_roi = px.bar(
        cohorts, x="intervention", y="revenue_lift_usd",
        color="intervention",
        text=cohorts["revenue_lift_usd"].apply(lambda x: f"${x:,.0f}"),
        labels={"revenue_lift_usd": "Revenue Lift (USD)", "intervention": ""},
        title="Projected Revenue Lift by Intervention (this dataset)",
        height=380,
        color_discrete_sequence=["#2196F3", "#4CAF50", "#FF9800"],
    )
    fig_roi.update_traces(textposition="outside")
    fig_roi.update_layout(showlegend=False)
    st.plotly_chart(fig_roi, width="stretch")

with c2:
    st.markdown("**Cohort Sizes**")
    st.dataframe(
        cohorts[["intervention", "cohort_size", "lift_rate", "new_conversions", "revenue_lift_usd"]]
        .rename(columns={
            "intervention": "Intervention",
            "cohort_size": "Cohort",
            "lift_rate": "Lift",
            "new_conversions": "New Binds",
            "revenue_lift_usd": "Revenue $",
        }),
        width="stretch",
        hide_index=True,
    )
    st.markdown("---")
    total_lift = cohorts["revenue_lift_usd"].sum()
    st.metric("Total Projected Lift", f"${total_lift:,.0f}")
    st.caption(f"At ${AVG_COMMISSION} avg commission · Assumes zero cohort overlap")