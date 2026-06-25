"""
simulate_funnel.py
------------------
Generates a synthetic insurance quote funnel dataset.
Output: data/funnel_events.csv
"""

import numpy as np
import pandas as pd
from faker import Faker
import random
import os

fake = Faker()
random.seed(42)
np.random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────

NUM_SESSIONS = 50_000

STEPS = ["zip", "vehicle", "driver", "quotes", "bind"]

CHANNELS = ["organic", "paid_search", "referral", "email"]
CHANNEL_WEIGHTS = [0.35, 0.30, 0.20, 0.15]

DEVICES = ["mobile", "desktop", "tablet"]
DEVICE_WEIGHTS = [0.55, 0.38, 0.07]

# Base drop probability at each step (before adjustments)
BASE_DROP = {
    "zip":     0.08,
    "vehicle": 0.12,
    "driver":  0.18,
    "quotes":  0.38,
    "bind":    0.10,
}

# Multipliers applied on top of base drop rate
CHANNEL_DROP_MULT = {
    "organic":     0.85,
    "paid_search": 1.25,
    "referral":    0.90,
    "email":       1.00,
}

DEVICE_DROP_MULT = {
    "mobile":  1.30,
    "desktop": 0.80,
    "tablet":  1.00,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def pick_quote_count():
    """Users see 1–8 quotes; more quotes → higher abandonment simulated downstream."""
    return np.random.choice([1, 2, 3, 4, 5, 6, 7, 8],
                            p=[0.05, 0.15, 0.20, 0.20, 0.15, 0.12, 0.08, 0.05])

def pick_cheapest_quote(prior_insured: bool) -> float:
    """Cheapest quote in USD. Prior-insured users tend to get better rates."""
    base = np.random.normal(loc=120 if prior_insured else 160, scale=40)
    return round(max(40, base), 2)

def time_on_step(step: str, dropped: bool) -> int:
    """Seconds spent on a step. Abandoners spend more time on quotes/driver."""
    means = {"zip": 15, "vehicle": 30, "driver": 60, "quotes": 120, "bind": 45}
    noise = 1.6 if (dropped and step in ["quotes", "driver"]) else 1.0
    return max(5, int(np.random.exponential(means[step] * noise)))

# ── Core simulation ───────────────────────────────────────────────────────────

def simulate_session(session_id: int) -> list[dict]:
    channel = np.random.choice(CHANNELS, p=CHANNEL_WEIGHTS)
    device  = np.random.choice(DEVICES,  p=DEVICE_WEIGHTS)
    prior_insured = random.random() < 0.62
    quote_count   = pick_quote_count()
    cheapest_quote = pick_cheapest_quote(prior_insured)
    user_id = fake.uuid4()

    # Quote-count penalty: seeing 5+ quotes increases drop at quotes step
    quote_penalty = 1.0 + max(0, (quote_count - 4) * 0.08)

    events = []
    ts = pd.Timestamp("2024-01-01") + pd.Timedelta(seconds=random.randint(0, 365 * 86400))

    for step in STEPS:
        # Compute drop probability for this step
        p_drop = BASE_DROP[step]
        p_drop *= CHANNEL_DROP_MULT[channel]
        p_drop *= DEVICE_DROP_MULT[device]
        if step == "quotes":
            p_drop *= quote_penalty
        p_drop = min(p_drop, 0.95)

        dropped = random.random() < p_drop
        seconds = time_on_step(step, dropped)

        events.append({
            "session_id":       session_id,
            "user_id":          user_id,
            "step":             step,
            "step_order":       STEPS.index(step) + 1,
            "timestamp":        ts,
            "channel":          channel,
            "device":           device,
            "prior_insured":    prior_insured,
            "quote_count":      quote_count if step in ["quotes", "bind"] else None,
            "cheapest_quote_usd": cheapest_quote if step in ["quotes", "bind"] else None,
            "time_on_step_sec": seconds,
            "dropped":          dropped,
        })

        ts += pd.Timedelta(seconds=seconds)

        if dropped:
            break  # user left the funnel

    return events

# ── Run ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Simulating {NUM_SESSIONS:,} sessions...")
    all_events = []
    for sid in range(NUM_SESSIONS):
        all_events.extend(simulate_session(sid))

    df = pd.DataFrame(all_events)

    out_path = os.path.join(os.path.dirname(__file__), "funnel_events.csv")
    df.to_csv(out_path, index=False)

    print(f"Done. {len(df):,} events written to {out_path}")
    print(f"\nStep-level event counts:\n{df['step'].value_counts().sort_index()}")
    bound = df[df['step'] == 'bind'][df[df['step'] == 'bind']['dropped'] == False]
    print(f"\nOverall bind rate: {len(bound) / NUM_SESSIONS:.2%}")

if __name__ == "__main__":
    main()