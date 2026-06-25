-- session_features.sql
-- One row per session with engineered features for the ML model.
-- Run against DuckDB with funnel_events.csv loaded as `funnel_events`

WITH session_meta AS (
    -- Pull session-level constants (same across all steps in a session)
    SELECT DISTINCT
        session_id,
        user_id,
        channel,
        device,
        prior_insured
    FROM funnel_events
),

step_flags AS (
    -- Which steps did the user reach, and did they drop there?
    SELECT
        session_id,
        MAX(step_order)                                         AS deepest_step,

        -- Did user reach each step?
        MAX(CASE WHEN step = 'vehicle' THEN 1 ELSE 0 END)      AS reached_vehicle,
        MAX(CASE WHEN step = 'driver'  THEN 1 ELSE 0 END)      AS reached_driver,
        MAX(CASE WHEN step = 'quotes'  THEN 1 ELSE 0 END)      AS reached_quotes,
        MAX(CASE WHEN step = 'bind'    THEN 1 ELSE 0 END)      AS reached_bind,

        -- Final outcome
        MAX(CASE WHEN step = 'bind' AND dropped = FALSE THEN 1 ELSE 0 END) AS converted

    FROM funnel_events
    GROUP BY session_id
),

time_features AS (
    -- Time spent on each step (useful signal for models)
    SELECT
        session_id,
        SUM(time_on_step_sec)                                   AS total_session_sec,
        MAX(CASE WHEN step = 'zip'     THEN time_on_step_sec END) AS time_zip_sec,
        MAX(CASE WHEN step = 'vehicle' THEN time_on_step_sec END) AS time_vehicle_sec,
        MAX(CASE WHEN step = 'driver'  THEN time_on_step_sec END) AS time_driver_sec,
        MAX(CASE WHEN step = 'quotes'  THEN time_on_step_sec END) AS time_quotes_sec
    FROM funnel_events
    GROUP BY session_id
),

quote_features AS (
    -- Quote-level signals (only populated for sessions that reached quotes step)
    SELECT
        session_id,
        quote_count,
        cheapest_quote_usd,
        CASE
            WHEN quote_count <= 2 THEN 'low'
            WHEN quote_count <= 4 THEN 'medium'
            ELSE 'high'
        END                                                     AS quote_count_bucket,
        CASE
            WHEN cheapest_quote_usd < 100 THEN 'budget'
            WHEN cheapest_quote_usd < 150 THEN 'mid'
            ELSE 'premium'
        END                                                     AS price_band
    FROM funnel_events
    WHERE step = 'quotes'
)

SELECT
    m.session_id,
    m.user_id,
    m.channel,
    m.device,
    m.prior_insured::INT                                        AS prior_insured,

    -- Step depth
    s.deepest_step,
    s.reached_vehicle,
    s.reached_driver,
    s.reached_quotes,
    s.reached_bind,

    -- Time signals
    t.total_session_sec,
    t.time_zip_sec,
    t.time_vehicle_sec,
    t.time_driver_sec,
    t.time_quotes_sec,

    -- Quote signals (NULL if user never reached quotes)
    q.quote_count,
    q.cheapest_quote_usd,
    q.quote_count_bucket,
    q.price_band,

    -- Target label
    s.converted

FROM session_meta m
JOIN step_flags   s USING (session_id)
JOIN time_features t USING (session_id)
LEFT JOIN quote_features q USING (session_id)
ORDER BY session_id;