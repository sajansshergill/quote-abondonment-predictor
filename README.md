# Insurance Quote Abandonment Predictor + Intervention Simulator

> A growth analytics project modeled on real-world insurance funnel optimization — identifying where users drop out of the quote flow and simulating the revenue lift from targeted interventions.

---

## Motivation

Car insurance platforms live and die by funnel conversion. A user who starts a quote but never binds a policy is a lost customer — and understanding *why* they left, and *when*, is the difference between a 3% and a 7% bind rate.

This project answers two questions:
1. **Where** in the quote funnel are users most likely to abandon — and what signals predict it?
2. **What intervention** (earlier price reveal, coverage simplification, re-engagement nudge) produces the highest expected lift in conversions?

---

## Project Architecture

```
quote-abondonment-predictor/
│
├── data/
│   └── simulate_funnel.py          # Synthetic funnel event generator
│
├── sql/
│   ├── funnel_drop_rates.sql       # Step-level drop-off aggregation
│   ├── session_features.sql        # Feature engineering per session
│   └── intervention_cohorts.sql   # Cohort splits for A/B simulation
│
├── notebooks/
│   ├── 01_eda.ipynb                # Funnel shape, drop-off by channel/device
│   ├── 02_survival_analysis.ipynb  # Time-to-abandon at each step (lifelines)
│   ├── 03_abandonment_model.ipynb  # XGBoost classifier: P(abandon) at step N
│   └── 04_intervention_sim.ipynb  # Lift simulation under intervention scenarios
│
├── app/
│   └── dashboard.py               # Streamlit dashboard: funnel health + ROI
│
├── requirements.txt
└── README.md
```

---

## Dataset (Simulated)

Generated via `simulate_funnel.py`. Each row is a funnel event:

| Column | Description |
|---|---|
| `user_id` | Unique user identifier |
| `session_id` | Quote session |
| `step` | Funnel stage: `zip → vehicle → driver → quotes → bind` |
| `timestamp` | Event time |
| `channel` | Acquisition source: `organic`, `paid_search`, `referral`, `email` |
| `device` | `mobile`, `desktop`, `tablet` |
| `quote_count` | Number of quotes returned at the quotes step |
| `cheapest_quote_usd` | Lowest quote seen by user |
| `prior_insured` | Boolean — was user previously insured? |
| `dropped` | Boolean — did user abandon at this step? |

~50,000 sessions generated with configurable drop-off rates per step and channel.

---

## Analysis Modules

### 1. Funnel Drop-off Analysis (SQL + Pandas)
- Step-level conversion rates
- Segmented by `channel`, `device`, `quote_count`, `price_band`
- Identifies the single highest-leakage step in the funnel

### 2. Survival Analysis (`lifelines`)
- Kaplan-Meier curves: time-to-abandon by channel
- Cox Proportional Hazards model: which features accelerate drop-off?
- Key output: *"Mobile users from paid search abandon 2.3x faster at the quotes step"*

### 3. Abandonment Classifier (`XGBoost`)
- Binary classification: will this session abandon before binding?
- Features: step, device, channel, quote spread, time-on-step, prior_insured
- Evaluated with precision/recall, SHAP feature importance
- Key output: *P(abandon)* score surfaced at the quotes step — enables real-time triggers

### 4. Intervention Simulator
Three scenarios modeled as counterfactual cohort shifts:

| Intervention | Hypothesis | Simulated Lift |
|---|---|---|
| Early price reveal | Show cheapest quote at driver step, not quotes step | +X% bind rate |
| Quote count cap | Cap displayed quotes at 3 (reduce decision fatigue) | +X% bind rate |
| Re-engagement nudge | Trigger email within 2h of mobile abandonment | +X% bind rate |

Lift estimated via: (Δ conversion rate) × (session volume) × (avg policy value)

---

## Dashboard (Streamlit)

`app/dashboard.py` surfaces:
- **Funnel waterfall chart** — step-level conversion with drop-off %
- **Abandonment risk heatmap** — channel × device grid
- **Intervention ROI panel** — estimated monthly revenue lift per scenario
- **Model score distribution** — P(abandon) histogram with threshold slider

Run locally:
```bash
streamlit run app/dashboard.py
```

---

## Stack

| Layer | Tool |
|---|---|
| Data simulation | Python, Faker, NumPy |
| Local SQL engine | DuckDB |
| Analysis | pandas, lifelines, scikit-learn, XGBoost |
| Explainability | SHAP |
| Dashboard | Streamlit |
| Version control | GitHub |

---

## Key Findings (Sample)

- **Quotes step** has the highest absolute drop-off (38% of all abandonment)
- **Mobile + paid search** users abandon 2.1x faster than desktop + organic
- Users who see **5+ quotes** are 44% less likely to bind than users who see 2–3
- Early price reveal simulation projects **+$187K monthly revenue** on 10K daily sessions at a $45 avg policy commission

---

## Business Framing

This project is structured the way a DS&A team member at a growth-stage insuretech would approach it:

1. **Start with the business question** — where is conversion leaking, and what's it costing us?
2. **Define the metric** — bind rate by channel, not just overall funnel conversion
3. **Run the analysis** — EDA → survival curves → predictive model
4. **Generate a recommendation** — not "here are the findings" but "here's what to ship next and why"
5. **Size the impact** — attach a dollar estimate to each intervention

---

## Setup

```bash
git clone https://github.com/sajansshergill/insurance-funnel-analytics
cd insurance-funnel-analytics
pip install -r requirements.txt

# Generate synthetic data
python data/simulate_funnel.py

# Launch dashboard
streamlit run app/dashboard.py
```

---

## Author

**Sajan Shergill**
M.S. Data Science, Pace University · QA/SDET · Data Engineer
[linkedin.com/in/sajanshergill](https://linkedin.com/in/sajanshergill) · [sajansshergill.github.io](https://sajansshergill.github.io)
