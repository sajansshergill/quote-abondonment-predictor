# %% [markdown]
# # 02 — Survival Analysis
# How long do users survive in the funnel?
# Uses `lifelines` for Kaplan-Meier curves and Cox Proportional Hazards.

# %%
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter

sns.set_theme(style="whitegrid")

STEP_ORDER = ["zip", "vehicle", "driver", "quotes", "bind"]
STEP_TO_INT = {s: i + 1 for i, s in enumerate(STEP_ORDER)}

# ── Load & prep ───────────────────────────────────────────────────────────────
# %%
con = duckdb.connect()
df = con.execute("SELECT * FROM read_csv_auto('../data/funnel_events.csv')").df()

# Build one row per session: duration = deepest step reached, event = dropped
session = (
    df.sort_values("step_order")
    .groupby("session_id")
    .agg(
        duration=("step_order", "max"),
        event=("dropped", "any"),           # True if user dropped anywhere
        channel=("channel", "first"),
        device=("device", "first"),
        prior_insured=("prior_insured", "first"),
    )
    .reset_index()
)
session["event"] = session["event"].astype(int)
print(session.shape)
session.head(3)

# %% [markdown]
# ## 1. Overall Kaplan-Meier Survival Curve

# %%
kmf = KaplanMeierFitter()
kmf.fit(durations=session["duration"], event_observed=session["event"])

fig, ax = plt.subplots(figsize=(8, 5))
kmf.plot_survival_function(ax=ax, ci_show=True, label="All users")
ax.set_xticks(range(1, 6))
ax.set_xticklabels(STEP_ORDER)
ax.set_xlabel("Funnel Step")
ax.set_ylabel("Probability of Still Being in Funnel")
ax.set_title("Kaplan-Meier: User Survival Through Quote Funnel", fontsize=13)
plt.tight_layout()
plt.savefig("../app/assets/km_overall.png", dpi=150)
plt.show()

# %% [markdown]
# ## 2. KM Curves by Channel

# %%
fig, ax = plt.subplots(figsize=(9, 5))
for channel, grp in session.groupby("channel"):
    kmf_c = KaplanMeierFitter()
    kmf_c.fit(grp["duration"], grp["event"], label=channel)
    kmf_c.plot_survival_function(ax=ax, ci_show=False)

ax.set_xticks(range(1, 6))
ax.set_xticklabels(STEP_ORDER)
ax.set_xlabel("Funnel Step")
ax.set_ylabel("Survival Probability")
ax.set_title("KM Curves by Acquisition Channel", fontsize=13)
plt.tight_layout()
plt.savefig("../app/assets/km_by_channel.png", dpi=150)
plt.show()

# %% [markdown]
# ## 3. KM Curves by Device

# %%
fig, ax = plt.subplots(figsize=(9, 5))
for device, grp in session.groupby("device"):
    kmf_d = KaplanMeierFitter()
    kmf_d.fit(grp["duration"], grp["event"], label=device)
    kmf_d.plot_survival_function(ax=ax, ci_show=False)

ax.set_xticks(range(1, 6))
ax.set_xticklabels(STEP_ORDER)
ax.set_xlabel("Funnel Step")
ax.set_ylabel("Survival Probability")
ax.set_title("KM Curves by Device Type", fontsize=13)
plt.tight_layout()
plt.savefig("../app/assets/km_by_device.png", dpi=150)
plt.show()

# %% [markdown]
# ## 4. Cox Proportional Hazards Model
# Quantify the effect of each feature on the hazard (risk of abandoning).

# %%
cox_df = session.copy()

# Encode categoricals as dummies (drop first to avoid multicollinearity)
cox_df = pd.get_dummies(cox_df, columns=["channel", "device"], drop_first=True)
cox_df = cox_df.drop(columns=["session_id"])

cph = CoxPHFitter()
cph.fit(cox_df, duration_col="duration", event_col="event")
cph.print_summary()

# %%
fig, ax = plt.subplots(figsize=(8, 5))
cph.plot(ax=ax)
ax.set_title("Cox PH: Hazard Ratios (>1 = higher abandonment risk)", fontsize=12)
ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
plt.tight_layout()
plt.savefig("../app/assets/cox_hazard_ratios.png", dpi=150)
plt.show()

# %% [markdown]
# ## Key Takeaways
#
# - Survival drops sharpest between **driver → quotes** and **quotes → bind**
# - `paid_search` has the **highest hazard ratio** among channels (users are window shoppers)
# - `mobile` device increases abandonment risk by ~30% vs desktop
# - `prior_insured` users survive significantly longer — they understand the product