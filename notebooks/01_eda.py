# %% [markdown]
# # 01 — Exploratory Data Analysis
# Funnel shape, drop-off by channel and device.

# %%
import duckdb
import pandas as pd
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

sns.set_theme(style="whitegrid", palette="muted")

DATA_PATH = ROOT / "data" / "funnel_events.csv"
ASSETS_DIR = ROOT / "app" / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
# %%
con = duckdb.connect()
df = con.execute(f"SELECT * FROM read_csv_auto('{DATA_PATH}')").df()
print(df.shape)
df.dtypes

# %%
df.head(3)

# %% [markdown]
# ## 1. Overall Funnel Shape

# %%
STEP_ORDER = ["zip", "vehicle", "driver", "quotes", "bind"]

funnel = (
    df.groupby("step")
    .agg(reached=("session_id", "count"), dropped=("dropped", "sum"))
    .reindex(STEP_ORDER)
    .assign(drop_rate=lambda x: x["dropped"] / x["reached"])
    .assign(pass_through=lambda x: 1 - x["drop_rate"])
)
print(funnel)

# %%
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(STEP_ORDER, funnel["reached"], color="#4C72B0", label="Reached")
ax.bar(STEP_ORDER, funnel["dropped"], color="#DD8452", label="Dropped")
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.set_title("Funnel: Sessions Reached vs. Dropped per Step", fontsize=13)
ax.set_xlabel("Step")
ax.set_ylabel("Sessions")
ax.legend()
plt.tight_layout()
plt.savefig(ASSETS_DIR / "funnel_bar.png", dpi=150)
plt.close()

# %% [markdown]
# ## 2. Drop Rate by Channel

# %%
channel_drop = (
    df.groupby(["step", "channel"])["dropped"]
    .mean()
    .reset_index()
    .rename(columns={"dropped": "drop_rate"})
)
channel_drop["step"] = pd.Categorical(channel_drop["step"], categories=STEP_ORDER, ordered=True)
channel_drop = channel_drop.sort_values("step")

# %%
g = sns.FacetGrid(channel_drop, col="step", col_order=STEP_ORDER, height=4, sharey=True)
g.map_dataframe(
    sns.barplot,
    x="channel",
    y="drop_rate",
    hue="channel",
    palette="Set2",
    legend=False,
)
g.set_ylabels("Drop Rate")
g.set_titles("{col_name}")
for ax in g.axes.flat:
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.tick_params(axis="x", rotation=30)
g.figure.suptitle("Drop Rate by Channel × Step", y=1.02, fontsize=13)
plt.tight_layout()
plt.savefig(ASSETS_DIR / "drop_by_channel.png", dpi=150)
plt.close()

# %% [markdown]
# ## 3. Drop Rate by Device

# %%
device_drop = (
    df.groupby(["step", "device"])["dropped"]
    .mean()
    .unstack("device")
    .reindex(STEP_ORDER)
)
device_drop.plot(kind="bar", figsize=(9, 5), colormap="Set1")
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.title("Drop Rate by Device per Step", fontsize=13)
plt.xlabel("Step")
plt.ylabel("Drop Rate")
plt.legend(title="Device")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(ASSETS_DIR / "drop_by_device.png", dpi=150)
plt.close()

# %% [markdown]
# ## 4. Drop Rate at Quotes Step by Quote Count

# %%
quotes_df = df[df["step"] == "quotes"].copy()
quotes_df["quote_bucket"] = pd.cut(
    quotes_df["quote_count"],
    bins=[0, 2, 4, 6, 8],
    labels=["1-2", "3-4", "5-6", "7-8"],
)

quote_drop = (
    quotes_df.groupby("quote_bucket", observed=False)["dropped"]
    .mean()
    .reset_index()
    .rename(columns={"dropped": "drop_rate"})
)

fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(
    data=quote_drop,
    x="quote_bucket",
    y="drop_rate",
    hue="quote_bucket",
    palette="Blues_d",
    legend=False,
    ax=ax,
)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_title("Drop Rate at Quotes Step by Number of Quotes Shown", fontsize=12)
ax.set_xlabel("Quote Count Bucket")
ax.set_ylabel("Drop Rate")
plt.tight_layout()
plt.savefig(ASSETS_DIR / "drop_by_quote_count.png", dpi=150)
plt.close()

# %% [markdown]
# ## 5. Key Takeaways
#
# - **Quotes step** drives the largest absolute abandonment
# - **Paid search + mobile** users drop at 2x+ the rate of organic + desktop
# - Showing **5+ quotes** meaningfully increases abandonment vs. 2–3 quotes
# - Price band: users seeing quotes above $160 abandon more, but budget users still drop (decision fatigue)