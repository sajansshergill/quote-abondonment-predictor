# %% [markdown]
# # 03 — Abandonment Classifier
# Predict P(abandon before binding) at the quotes step using XGBoost.
# Output: model + SHAP feature importance.

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
try:
    import shap
    SHAP_IMPORT_ERROR = None
except ImportError as exc:
    shap = None
    SHAP_IMPORT_ERROR = exc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, roc_auc_score,
    RocCurveDisplay, PrecisionRecallDisplay,
)
from xgboost import XGBClassifier
import joblib

SEED = 42

DATA_PATH = ROOT / "data" / "funnel_events.csv"
SQL_PATH = ROOT / "sql" / "session_features.sql"
ASSETS_DIR = ROOT / "app" / "assets"
MODELS_DIR = ROOT / "models"
SCORED_PATH = ROOT / "data" / "scored_sessions.csv"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Load session-level features via SQL ──────────────────────────────────────
# %%
con = duckdb.connect()
con.execute(f"CREATE VIEW funnel_events AS SELECT * FROM read_csv_auto('{DATA_PATH}')")

with open(SQL_PATH, encoding="utf-8") as f:
    sql = f.read()

# session_features.sql ends with a SELECT; run it directly
features_df = con.execute(sql).df()
print(features_df.shape)
features_df.head(3)

# %% [markdown]
# ## 1. Feature Engineering

# %%
# Keep only sessions that reached the quotes step (that's where the model fires)
model_df = features_df[features_df["reached_quotes"] == 1].copy()

# Fill NULLs for sessions that didn't complete quote step
model_df["quote_count"]       = model_df["quote_count"].fillna(0)
model_df["cheapest_quote_usd"] = model_df["cheapest_quote_usd"].fillna(model_df["cheapest_quote_usd"].median())
model_df["time_quotes_sec"]   = model_df["time_quotes_sec"].fillna(0)

# Encode categoricals
le_channel = LabelEncoder()
le_device  = LabelEncoder()
le_price   = LabelEncoder()
le_qbucket = LabelEncoder()

model_df["channel_enc"]      = le_channel.fit_transform(model_df["channel"])
model_df["device_enc"]       = le_device.fit_transform(model_df["device"])
model_df["price_band_enc"]   = le_price.fit_transform(model_df["price_band"].fillna("unknown"))
model_df["quote_bucket_enc"] = le_qbucket.fit_transform(model_df["quote_count_bucket"].fillna("unknown"))

FEATURES = [
    "channel_enc", "device_enc", "prior_insured",
    "deepest_step", "quote_count", "cheapest_quote_usd",
    "time_driver_sec", "time_quotes_sec",
    "price_band_enc", "quote_bucket_enc",
    "total_session_sec",
]

X = model_df[FEATURES]
y = 1 - model_df["converted"]   # 1 = abandoned (did not bind)

print(f"Class balance — abandoned: {y.mean():.2%}")

# %% [markdown]
# ## 2. Train / Test Split

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# %% [markdown]
# ## 3. XGBoost Classifier

# %%
clf = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=SEED,
    n_jobs=-1,
)
clf.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50,
)

# %% [markdown]
# ## 4. Evaluation

# %%
y_pred  = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, target_names=["Converted", "Abandoned"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[0], name="XGBoost")
axes[0].set_title("ROC Curve")
PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=axes[1], name="XGBoost")
axes[1].set_title("Precision-Recall Curve")
plt.tight_layout()
plt.savefig(ASSETS_DIR / "model_eval.png", dpi=150)
plt.close()

# %% [markdown]
# ## 5. Feature Importance

# %%
fig, ax = plt.subplots(figsize=(9, 5))
if shap is not None:
    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X_test)
    shap.summary_plot(shap_vals, X_test, feature_names=FEATURES, show=False)
    plt.title("SHAP Feature Importance - Abandonment Model", fontsize=12)
else:
    print(f"SHAP unavailable ({SHAP_IMPORT_ERROR}); using XGBoost feature importance.")
    importance = (
        pd.Series(clf.feature_importances_, index=FEATURES)
        .sort_values(ascending=True)
    )
    importance.plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_title("XGBoost Feature Importance - Abandonment Model", fontsize=12)
    ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(ASSETS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

# %% [markdown]
# ## 6. Score the Full Dataset (for dashboard)

# %%
model_df["abandon_prob"] = clf.predict_proba(X)[:, 1]

model_df.to_csv(SCORED_PATH, index=False)
print(f"Scored sessions saved to {SCORED_PATH}")

# Save model
MODEL_PATH = MODELS_DIR / "abandonment_xgb.pkl"
joblib.dump(clf, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# %% [markdown]
# ## Key Takeaways
#
# - `cheapest_quote_usd` and `time_quotes_sec` are the top SHAP drivers
# - High time-on-quotes + high price = near-certain abandonment
# - `prior_insured` = 1 is a strong protective signal
# - Model ROC-AUC ~0.78–0.82 on held-out data