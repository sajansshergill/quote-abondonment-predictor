# %% [markdown]
# # 04 — Intervention Simulator
# Estimate revenue lift from three targeted interventions.
# Based on cohort definitions in `sql/intervention_cohorts.sql`.

# %%
import duckdb
import pandas as pd
import numpy as np
import os
from pathlib import Path

def find_project_root() -> Path:
    start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    for path in (start, *start.parents):
        if (path / "requirements.txt").exists() and (path / "data").exists():
            return path
    raise RuntimeError("Could not locate project root.")

ROOT = find_project_root()
MPLCONFIGDIR = ROOT / ".matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

sns.set_theme(style="whitegrid")

DATA_PATH = ROOT / "data" / "funnel_events.csv"
SQL_PATH = ROOT / "sql" / "intervention_cohorts.sql"
ASSETS_DIR = ROOT / "app" / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

AVG_POLICY_COMMISSION = 45   # USD per bound policy
DAILY_SESSIONS        = 10_000

# ── Load data ─────────────────────────────────────────────────────────────────
# %%
con = duckdb.connect()
con.execute(f"CREATE VIEW funnel_events AS SELECT * FROM read_csv_auto('{DATA_PATH}')")

# %% [markdown]
# ## 1. Baseline Conversion Metrics

# %%
baseline = con.execute("""
    WITH reached AS (
        SELECT COUNT(DISTINCT session_id) AS total_sessions FROM funnel_events
    ),
    bound AS (
        SELECT COUNT(DISTINCT session_id) AS bound_sessions
        FROM funnel_events
        WHERE step = 'bind' AND dropped = FALSE
    )
    SELECT
        total_sessions,
        bound_sessions,
        ROUND(bound_sessions * 100.0 / total_sessions, 2) AS bind_rate_pct,
        bound_sessions * 45 AS total_revenue_usd
    FROM reached, bound
""").df()

print(baseline.T)
baseline_bind_rate = baseline["bind_rate_pct"].iloc[0] / 100

# %% [markdown]
# ## 2. Intervention Cohort Sizing

# %%
# Run intervention_cohorts.sql to get cohort sizes
with open(SQL_PATH, encoding="utf-8") as f:
    sql = f.read()

# Execute the final SELECT (sizing query) separately
sizing_sql = """
WITH cohort_early_price_reveal AS (
    SELECT session_id, 'early_price_reveal' AS intervention, 0.15 AS assumed_lift_rate
    FROM funnel_events
    WHERE step = 'quotes' AND dropped = TRUE AND cheapest_quote_usd < 120
),
cohort_quote_cap AS (
    SELECT session_id, 'quote_count_cap' AS intervention, 0.12 AS assumed_lift_rate
    FROM funnel_events
    WHERE step = 'quotes' AND dropped = TRUE AND quote_count >= 5
),
cohort_reengage_nudge AS (
    SELECT session_id, 'reengage_nudge' AS intervention, 0.08 AS assumed_lift_rate
    FROM funnel_events
    WHERE device = 'mobile' AND step IN ('driver', 'quotes') AND dropped = TRUE
),
all_cohorts AS (
    SELECT * FROM cohort_early_price_reveal
    UNION ALL SELECT * FROM cohort_quote_cap
    UNION ALL SELECT * FROM cohort_reengage_nudge
)
SELECT
    intervention,
    COUNT(*) AS cohort_size,
    MAX(assumed_lift_rate) AS lift_rate,
    ROUND(COUNT(*) * MAX(assumed_lift_rate)) AS projected_new_conversions,
    ROUND(COUNT(*) * MAX(assumed_lift_rate) * 45, 0) AS projected_revenue_lift_usd
FROM all_cohorts
GROUP BY intervention
ORDER BY projected_revenue_lift_usd DESC
"""

cohorts = con.execute(sizing_sql).df()
print(cohorts)

# %% [markdown]
# ## 3. Revenue Lift Visualization

# %%
INTERVENTION_LABELS = {
    "early_price_reveal": "Early Price Reveal\n(show quote at driver step)",
    "quote_count_cap":    "Quote Count Cap\n(show max 3 quotes)",
    "reengage_nudge":     "Re-engagement Nudge\n(email mobile abandoners)",
}
cohorts["label"] = cohorts["intervention"].map(INTERVENTION_LABELS)

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(
    cohorts["label"],
    cohorts["projected_revenue_lift_usd"],
    color=["#2196F3", "#4CAF50", "#FF9800"],
    height=0.5,
)
ax.bar_label(bars, labels=[f"${v:,.0f}" for v in cohorts["projected_revenue_lift_usd"]], padding=5)
ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.set_xlabel("Projected Revenue Lift (USD, this dataset)")
ax.set_title("Estimated Revenue Lift by Intervention", fontsize=13)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(ASSETS_DIR / "intervention_lift.png", dpi=150)
plt.close()

# %% [markdown]
# ## 4. Sensitivity Analysis — Lift Rate vs. Revenue

# %%
lift_range = np.arange(0.02, 0.30, 0.02)

fig, ax = plt.subplots(figsize=(9, 5))
for _, row in cohorts.iterrows():
    revenues = row["cohort_size"] * lift_range * AVG_POLICY_COMMISSION
    ax.plot(lift_range * 100, revenues, marker="o", markersize=4, label=row["intervention"])

ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.xaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_xlabel("Assumed Lift Rate (%)")
ax.set_ylabel("Revenue Lift (USD)")
ax.set_title("Sensitivity: Revenue Lift vs. Assumed Lift Rate", fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig(ASSETS_DIR / "sensitivity.png", dpi=150)
plt.close()

# %% [markdown]
# ## 5. Extrapolate to Production Scale

# %%
# Scale cohort sizes to daily production volume
dataset_sessions = con.execute("SELECT COUNT(DISTINCT session_id) FROM funnel_events").fetchone()[0]
scale_factor = DAILY_SESSIONS / dataset_sessions

cohorts["daily_cohort_size"]      = (cohorts["cohort_size"] * scale_factor).round()
cohorts["daily_new_conversions"]  = (cohorts["projected_new_conversions"] * scale_factor).round()
cohorts["daily_revenue_lift_usd"] = (cohorts["projected_revenue_lift_usd"] * scale_factor).round()
cohorts["monthly_revenue_lift_usd"] = cohorts["daily_revenue_lift_usd"] * 30

print("\n── Production-scale projections (10K daily sessions) ──")
print(
    cohorts[[
        "intervention", "daily_cohort_size",
        "daily_new_conversions", "monthly_revenue_lift_usd"
    ]].to_string(index=False)
)

# %% [markdown]
# ## Key Takeaways
#
# | Intervention | Monthly Lift (est.) | Effort |
# |---|---|---|
# | Early Price Reveal | ~$XXK | Medium — requires product change |
# | Quote Count Cap | ~$XXK | Low — UI config change |
# | Re-engagement Nudge | ~$XXK | Low — email trigger |
#
# **Recommendation**: Ship quote count cap first (lowest effort, immediate).
# Run early price reveal as an A/B test alongside it.
# Re-engagement nudge requires email infra but has the lowest per-session cost.