-- intervention_cohorts.sql
-- Cohort definitions for three intervention simulations.
-- Each query tags sessions that WOULD be targeted by a given intervention.
-- Run against DuckDB with funnel_events.csv loaded as `funnel_events`

-- ── Intervention 1: Early Price Reveal ───────────────────────────────────────
-- Hypothesis: Show cheapest quote at the driver step (not quotes).
-- Target cohort: sessions that dropped at the quotes step where quote was < $120.
-- Rationale: Low-price users are price-sensitive; an early reveal pulls them through.

CREATE OR REPLACE VIEW cohort_early_price_reveal AS
SELECT
    session_id,
    channel,
    device,
    cheapest_quote_usd,
    'early_price_reveal'                            AS intervention,
    -- Estimated lift: 15% of this cohort converts that otherwise wouldn't
    0.15                                            AS assumed_lift_rate
FROM funnel_events
WHERE step = 'quotes'
  AND dropped = TRUE
  AND cheapest_quote_usd < 120;


-- ── Intervention 2: Quote Count Cap ──────────────────────────────────────────
-- Hypothesis: Cap displayed quotes at 3 to reduce decision fatigue.
-- Target cohort: sessions that dropped at quotes step with 5+ quotes shown.

CREATE OR REPLACE VIEW cohort_quote_cap AS
SELECT
    session_id,
    channel,
    device,
    quote_count,
    cheapest_quote_usd,
    'quote_count_cap'                               AS intervention,
    0.12                                            AS assumed_lift_rate
FROM funnel_events
WHERE step = 'quotes'
  AND dropped = TRUE
  AND quote_count >= 5;


-- ── Intervention 3: Re-engagement Nudge ──────────────────────────────────────
-- Hypothesis: Email users who abandoned on mobile within 2 hours.
-- Target cohort: mobile sessions that dropped at quotes or driver step.

CREATE OR REPLACE VIEW cohort_reengage_nudge AS
SELECT
    session_id,
    channel,
    device,
    step                                            AS abandoned_at_step,
    'reengage_nudge'                                AS intervention,
    0.08                                            AS assumed_lift_rate
FROM funnel_events
WHERE device = 'mobile'
  AND step IN ('driver', 'quotes')
  AND dropped = TRUE;


-- ── Combined intervention sizing ─────────────────────────────────────────────
-- How many sessions does each intervention target? What's the projected lift?

WITH sizes AS (
    SELECT intervention, COUNT(*) AS cohort_size, MAX(assumed_lift_rate) AS lift_rate
    FROM (
        SELECT session_id, intervention, assumed_lift_rate FROM cohort_early_price_reveal
        UNION ALL
        SELECT session_id, intervention, assumed_lift_rate FROM cohort_quote_cap
        UNION ALL
        SELECT session_id, intervention, assumed_lift_rate FROM cohort_reengage_nudge
    ) all_cohorts
    GROUP BY intervention
)
SELECT
    intervention,
    cohort_size,
    lift_rate,
    ROUND(cohort_size * lift_rate)                  AS projected_new_conversions,
    ROUND(cohort_size * lift_rate * 45, 0)          AS projected_revenue_lift_usd   -- $45 avg commission
FROM sizes
ORDER BY projected_revenue_lift_usd DESC;