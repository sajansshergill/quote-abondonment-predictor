-- funnel_drop_rates.sql
-- Step-level drop-off aggregation
-- Run against DuckDB with funnel_events.csv loaded as `funnel_events`

-- ── 1. Overall funnel conversion by step ─────────────────────────────────────
SELECT
    step,
    step_order,
    COUNT(*)                                        AS sessions_reached,
    SUM(dropped::INT)                               AS sessions_dropped,
    ROUND(AVG(dropped::FLOAT) * 100, 2)             AS drop_rate_pct,
    ROUND((1 - AVG(dropped::FLOAT)) * 100, 2)       AS pass_through_pct
FROM funnel_events
GROUP BY step, step_order
ORDER BY step_order;


-- ── 2. Drop rate by step × channel ──────────────────────────────────────────
SELECT
    step,
    step_order,
    channel,
    COUNT(*)                                        AS sessions_reached,
    ROUND(AVG(dropped::FLOAT) * 100, 2)             AS drop_rate_pct
FROM funnel_events
GROUP BY step, step_order, channel
ORDER BY step_order, drop_rate_pct DESC;


-- ── 3. Drop rate by step × device ───────────────────────────────────────────
SELECT
    step,
    step_order,
    device,
    COUNT(*)                                        AS sessions_reached,
    ROUND(AVG(dropped::FLOAT) * 100, 2)             AS drop_rate_pct
FROM funnel_events
GROUP BY step, step_order, device
ORDER BY step_order, drop_rate_pct DESC;


-- ── 4. Drop rate at quotes step by quote count bucket ───────────────────────
SELECT
    CASE
        WHEN quote_count <= 2 THEN '1-2 quotes'
        WHEN quote_count <= 4 THEN '3-4 quotes'
        WHEN quote_count <= 6 THEN '5-6 quotes'
        ELSE '7+ quotes'
    END                                             AS quote_bucket,
    COUNT(*)                                        AS sessions,
    ROUND(AVG(dropped::FLOAT) * 100, 2)             AS drop_rate_pct
FROM funnel_events
WHERE step = 'quotes'
GROUP BY quote_bucket
ORDER BY drop_rate_pct;


-- ── 5. Drop rate at quotes step by price band ────────────────────────────────
SELECT
    CASE
        WHEN cheapest_quote_usd < 80  THEN 'Under $80'
        WHEN cheapest_quote_usd < 120 THEN '$80–$119'
        WHEN cheapest_quote_usd < 160 THEN '$120–$159'
        WHEN cheapest_quote_usd < 200 THEN '$160–$199'
        ELSE '$200+'
    END                                             AS price_band,
    COUNT(*)                                        AS sessions,
    ROUND(AVG(dropped::FLOAT) * 100, 2)             AS drop_rate_pct
FROM funnel_events
WHERE step = 'quotes'
GROUP BY price_band
ORDER BY drop_rate_pct;


-- ── 6. Revenue leakage estimate ──────────────────────────────────────────────
-- Assumes $45 avg policy commission per bind
WITH bind_possible AS (
    SELECT COUNT(*) AS reached_quotes
    FROM funnel_events
    WHERE step = 'quotes'
),
actual_binds AS (
    SELECT COUNT(*) AS bound
    FROM funnel_events
    WHERE step = 'bind' AND dropped = FALSE
)
SELECT
    reached_quotes,
    bound,
    (reached_quotes - bound)                        AS lost_sessions,
    (reached_quotes - bound) * 45                   AS estimated_lost_revenue_usd
FROM bind_possible, actual_binds;