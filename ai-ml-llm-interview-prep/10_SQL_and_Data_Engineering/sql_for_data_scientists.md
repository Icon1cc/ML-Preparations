# SQL for Data Scientists: Complete Interview Reference

> SQL is the language of data. At senior ML roles, you are expected to write production-quality SQL with window functions, CTEs, and performance awareness — not just `SELECT *`. This file covers everything from fundamentals to ML-specific patterns, with logistics-grounded examples throughout.

---

## Table of Contents

1. [SQL Fundamentals](#1-sql-fundamentals)
2. [JOINs: Complete Reference](#2-joins-complete-reference)
3. [Aggregation and NULL Handling](#3-aggregation-and-null-handling)
4. [String and Date Functions](#4-string-and-date-functions)
5. [Window Functions](#5-window-functions)
6. [CTEs and Subqueries](#6-ctes-and-subqueries)
7. [Advanced SQL Patterns](#7-advanced-sql-patterns)
8. [ML-Specific SQL Patterns](#8-ml-specific-sql-patterns)
9. [Query Performance and Optimization](#9-query-performance-and-optimization)
10. [Interview Questions with Complete Solutions](#10-interview-questions-with-complete-solutions)

---

## 1. SQL Fundamentals

### The Logical Processing Order

SQL has a specific order of clause evaluation that differs from how you write the query. Understanding this prevents bugs.

```
FROM / JOIN   →  which tables, which rows
WHERE         →  filter rows before aggregation
GROUP BY      →  define aggregation groups
HAVING        →  filter groups after aggregation
SELECT        →  compute output columns
DISTINCT      →  remove duplicate rows
ORDER BY      →  sort results
LIMIT / TOP   →  truncate result set
```

**Critical implication**: You **cannot** reference a SELECT alias in a WHERE clause because WHERE is evaluated before SELECT.

```sql
-- WRONG: alias not yet defined when WHERE runs
SELECT shipment_id, delay_minutes / 60.0 AS delay_hours
FROM deliveries
WHERE delay_hours > 2;

-- CORRECT: repeat the expression, or use a subquery/CTE
SELECT shipment_id, delay_minutes / 60.0 AS delay_hours
FROM deliveries
WHERE delay_minutes / 60.0 > 2;

-- ALSO CORRECT: use CTE
WITH enriched AS (
    SELECT shipment_id, delay_minutes / 60.0 AS delay_hours
    FROM deliveries
)
SELECT * FROM enriched WHERE delay_hours > 2;
```

### WHERE vs HAVING

```sql
-- WHERE: filters before grouping (acts on individual rows)
-- HAVING: filters after grouping (acts on aggregated groups)

-- Find carriers with avg delay > 30 min, excluding cancelled shipments
SELECT
    carrier_id,
    COUNT(*)          AS total_deliveries,
    AVG(delay_minutes) AS avg_delay_min
FROM deliveries
WHERE status != 'CANCELLED'        -- row-level filter
GROUP BY carrier_id
HAVING AVG(delay_minutes) > 30     -- group-level filter
ORDER BY avg_delay_min DESC;
```

### CASE WHEN: Conditional Logic

```sql
-- Classify delivery performance
SELECT
    shipment_id,
    delay_minutes,
    CASE
        WHEN delay_minutes <= 0   THEN 'On Time'
        WHEN delay_minutes <= 30  THEN 'Slightly Late'
        WHEN delay_minutes <= 120 THEN 'Late'
        ELSE                           'Severely Late'
    END AS performance_bucket,
    -- Inline binary flag (useful for ML feature engineering)
    CASE WHEN delay_minutes <= 0 THEN 1 ELSE 0 END AS is_on_time
FROM deliveries;

-- CASE in aggregations: pivot-like behavior
SELECT
    carrier_id,
    COUNT(*) AS total,
    SUM(CASE WHEN delay_minutes <= 0 THEN 1 ELSE 0 END) AS on_time_count,
    ROUND(100.0 * SUM(CASE WHEN delay_minutes <= 0 THEN 1 ELSE 0 END) / COUNT(*), 2) AS on_time_pct
FROM deliveries
GROUP BY carrier_id;
```

---

## 2. JOINs: Complete Reference

### JOIN Type Summary

```
Table A          Table B
+------+         +------+
| 1    |         | 1    |
| 2    |         | 3    |
| 3    |         | 4    |
+------+         +------+

INNER JOIN:  rows 1, 3         (intersection)
LEFT JOIN:   rows 1, 2, 3      (all of A, matched B or NULL)
RIGHT JOIN:  rows 1, 3, 4      (all of B, matched A or NULL)
FULL OUTER:  rows 1, 2, 3, 4   (union, NULLs where no match)
CROSS JOIN:  3 x 3 = 9 rows    (Cartesian product)
```

### INNER JOIN

Returns only rows with matching keys in both tables.

```sql
-- Shipments joined with their carrier details
SELECT
    s.shipment_id,
    s.origin_depot,
    s.destination_depot,
    s.delay_minutes,
    c.carrier_name,
    c.carrier_type
FROM shipments s
INNER JOIN carriers c ON s.carrier_id = c.carrier_id;
-- Any shipment without a matching carrier is DROPPED
```

### LEFT JOIN

Returns all rows from the left table; right table columns are NULL when no match.

```sql
-- All customers, with their most recent shipment (if any)
-- Customers with no shipments still appear, with NULL shipment fields
SELECT
    cu.customer_id,
    cu.customer_name,
    cu.signup_date,
    s.shipment_id,
    s.ship_date
FROM customers cu
LEFT JOIN shipments s ON cu.customer_id = s.customer_id
    AND s.ship_date = (
        SELECT MAX(ship_date)
        FROM shipments
        WHERE customer_id = cu.customer_id
    );

-- Common pattern: find rows in A with NO match in B
SELECT cu.customer_id, cu.customer_name
FROM customers cu
LEFT JOIN shipments s ON cu.customer_id = s.customer_id
WHERE s.shipment_id IS NULL;  -- null means no join match found
```

### FULL OUTER JOIN

Returns all rows from both tables; NULL where no match.

```sql
-- Compare carrier performance metrics from two time periods
SELECT
    COALESCE(q1.carrier_id, q2.carrier_id) AS carrier_id,
    q1.avg_delay AS q1_avg_delay,
    q2.avg_delay AS q2_avg_delay,
    q2.avg_delay - q1.avg_delay AS delay_change
FROM
    (SELECT carrier_id, AVG(delay_minutes) AS avg_delay
     FROM deliveries WHERE quarter = 'Q1' GROUP BY carrier_id) q1
FULL OUTER JOIN
    (SELECT carrier_id, AVG(delay_minutes) AS avg_delay
     FROM deliveries WHERE quarter = 'Q2' GROUP BY carrier_id) q2
    ON q1.carrier_id = q2.carrier_id;
```

### CROSS JOIN

Every row of A paired with every row of B. Use deliberately.

```sql
-- Generate a full grid of (depot, date) combinations
-- Then LEFT JOIN actuals to find missing data
WITH date_spine AS (
    SELECT CAST(generate_series(
        '2024-01-01'::date,
        '2024-01-31'::date,
        '1 day'::interval
    ) AS date) AS report_date
),
depots AS (SELECT DISTINCT depot_id FROM shipments)
SELECT
    d.depot_id,
    ds.report_date,
    COALESCE(COUNT(s.shipment_id), 0) AS daily_volume
FROM depots d
CROSS JOIN date_spine ds
LEFT JOIN shipments s
    ON s.origin_depot = d.depot_id
    AND s.ship_date = ds.report_date
GROUP BY d.depot_id, ds.report_date
ORDER BY d.depot_id, ds.report_date;
```

### SELF JOIN

A table joined to itself. Used for sequential event analysis, hierarchies.

```sql
-- Find packages where the same customer had two deliveries on consecutive days
SELECT
    a.customer_id,
    a.ship_date AS first_delivery,
    b.ship_date AS second_delivery,
    a.shipment_id AS first_shipment,
    b.shipment_id AS second_shipment
FROM shipments a
INNER JOIN shipments b
    ON a.customer_id = b.customer_id
    AND b.ship_date = a.ship_date + INTERVAL '1 day'
    AND a.shipment_id != b.shipment_id;  -- exclude self-match
```

### JOIN Performance Pitfalls

**Duplicate row explosion**: When the JOIN key is not unique in one or both tables.

```sql
-- shipments has many rows per carrier_id
-- performance_bonuses also has many rows per carrier_id (one per month)
-- INNER JOIN produces shipments * bonuses rows per carrier — probably wrong!

-- WRONG:
SELECT s.*, pb.bonus_amount
FROM shipments s
JOIN performance_bonuses pb ON s.carrier_id = pb.carrier_id;
-- This produces Cartesian product within each carrier group

-- CORRECT: join on the right granularity
SELECT s.*, pb.bonus_amount
FROM shipments s
JOIN performance_bonuses pb
    ON s.carrier_id = pb.carrier_id
    AND DATE_TRUNC('month', s.ship_date) = pb.bonus_month;
```

**Always verify row counts** after a JOIN during development:

```sql
SELECT COUNT(*) FROM result_table;  -- should match expected grain
-- If count is much larger than expected, you have a fan-out JOIN bug
```

---

## 3. Aggregation and NULL Handling

### Aggregation Functions

```sql
SELECT
    carrier_id,
    COUNT(*)                    AS total_rows,          -- counts all rows including NULLs
    COUNT(delay_minutes)        AS non_null_delays,     -- counts non-NULL only
    COUNT(DISTINCT customer_id) AS unique_customers,
    SUM(delay_minutes)          AS total_delay,         -- NULL-safe (ignores NULLs)
    AVG(delay_minutes)          AS avg_delay,           -- ignores NULLs in numerator AND denominator
    MIN(delay_minutes)          AS best_performance,
    MAX(delay_minutes)          AS worst_performance,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY delay_minutes) AS median_delay,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY delay_minutes) AS p95_delay
FROM deliveries
GROUP BY carrier_id;
```

### NULL Handling: The Complete Picture

NULLs represent **unknown or missing** values. They propagate through arithmetic and comparisons in counterintuitive ways.

```sql
-- NULL arithmetic: any operation on NULL returns NULL
SELECT NULL + 5;        -- NULL
SELECT NULL * 0;        -- NULL (NOT 0!)
SELECT NULL = NULL;     -- NULL (not TRUE!)
SELECT NULL != NULL;    -- NULL (not TRUE!)

-- Correct NULL comparisons
WHERE delay_minutes IS NULL
WHERE delay_minutes IS NOT NULL

-- COALESCE: return first non-NULL argument
SELECT COALESCE(delay_minutes, 0) AS delay_minutes_clean
FROM deliveries;

-- COALESCE for multi-column fallback
SELECT COALESCE(actual_delivery_ts, estimated_delivery_ts, promised_delivery_ts) AS best_delivery_time
FROM deliveries;

-- NULLIF: return NULL if two values are equal (prevents division by zero)
SELECT
    total_deliveries,
    on_time_deliveries,
    on_time_deliveries * 1.0 / NULLIF(total_deliveries, 0) AS on_time_rate
FROM carrier_summary;

-- NULL in aggregations: AVG ignores NULLs
-- This matters! If 10% of rows have NULL delay, AVG computes over 90%
-- Be explicit about what you want:
SELECT
    COUNT(*) AS total_shipments,
    COUNT(delay_minutes) AS shipments_with_delay_data,
    AVG(delay_minutes) AS avg_delay_excluding_nulls,
    AVG(COALESCE(delay_minutes, 0)) AS avg_delay_treating_null_as_zero
FROM deliveries;
```

---

## 4. String and Date Functions

### String Functions

```sql
-- Concatenation
SELECT CONCAT(first_name, ' ', last_name) AS full_name FROM customers;
SELECT first_name || ' ' || last_name AS full_name FROM customers;  -- PostgreSQL/SQLite

-- Substring extraction
SELECT SUBSTRING(tracking_number, 1, 3) AS carrier_prefix FROM shipments;
SELECT LEFT(tracking_number, 3) AS carrier_prefix FROM shipments;
SELECT RIGHT(zip_code, 4) AS zip_suffix FROM addresses;

-- Case transformation
SELECT UPPER(carrier_name), LOWER(carrier_name) FROM carriers;

-- Trimming whitespace
SELECT TRIM(carrier_name) FROM carriers;
SELECT LTRIM(RTRIM(carrier_name)) FROM carriers;  -- explicit left+right

-- String matching
SELECT * FROM carriers WHERE carrier_name LIKE 'Fed%';    -- starts with
SELECT * FROM carriers WHERE carrier_name LIKE '%Express'; -- ends with
SELECT * FROM carriers WHERE carrier_name LIKE '%air%';   -- contains
SELECT * FROM carriers WHERE carrier_name ILIKE '%fedex%'; -- case-insensitive (PostgreSQL)

-- REGEXP: more powerful pattern matching
SELECT * FROM shipments WHERE tracking_number ~ '^1Z[A-Z0-9]{16}$';  -- UPS format (PostgreSQL)
SELECT * FROM shipments WHERE REGEXP_LIKE(tracking_number, '^1Z[A-Z0-9]{16}$');  -- Snowflake/MySQL

-- String length and position
SELECT LENGTH(carrier_name), POSITION('Air' IN carrier_name) FROM carriers;

-- Replace
SELECT REPLACE(carrier_name, 'Corp', 'Corporation') FROM carriers;
```

### Date Functions (Critical for Time Series ML Work)

```sql
-- Current timestamps
SELECT CURRENT_DATE, CURRENT_TIMESTAMP, NOW();

-- DATE_TRUNC: truncate to a time unit (returns a timestamp)
SELECT DATE_TRUNC('day', delivered_at)    AS delivery_date
SELECT DATE_TRUNC('week', delivered_at)   AS delivery_week
SELECT DATE_TRUNC('month', delivered_at)  AS delivery_month
SELECT DATE_TRUNC('year', delivered_at)   AS delivery_year

-- EXTRACT: pull out a component as a number
SELECT EXTRACT(YEAR FROM delivered_at)    AS year_num
SELECT EXTRACT(MONTH FROM delivered_at)   AS month_num
SELECT EXTRACT(DOW FROM delivered_at)     AS day_of_week   -- 0=Sunday in PostgreSQL
SELECT EXTRACT(HOUR FROM delivered_at)    AS hour_of_day
SELECT EXTRACT(WEEK FROM delivered_at)    AS week_of_year

-- Date arithmetic
SELECT ship_date + INTERVAL '7 days' AS follow_up_date
SELECT ship_date - INTERVAL '1 month' AS prior_month_date
SELECT AGE(delivered_at, shipped_at) AS transit_time

-- DATE_DIFF: difference between dates
-- PostgreSQL style
SELECT delivered_at::date - shipped_at::date AS transit_days
-- BigQuery style
SELECT DATE_DIFF(delivered_at, shipped_at, DAY) AS transit_days
-- Snowflake style
SELECT DATEDIFF('day', shipped_at, delivered_at) AS transit_days

-- TO_TIMESTAMP: parse strings to timestamps
SELECT TO_TIMESTAMP('2024-01-15 14:30:00', 'YYYY-MM-DD HH24:MI:SS')

-- Practical: feature engineering date components for ML
SELECT
    shipment_id,
    ship_date,
    EXTRACT(DOW FROM ship_date)    AS day_of_week,     -- ML feature
    EXTRACT(MONTH FROM ship_date)  AS month,            -- ML feature
    EXTRACT(WEEK FROM ship_date)   AS week_of_year,    -- ML feature
    CASE WHEN EXTRACT(DOW FROM ship_date) IN (0, 6) THEN 1 ELSE 0 END AS is_weekend,
    -- Is this a peak season month?
    CASE WHEN EXTRACT(MONTH FROM ship_date) IN (11, 12) THEN 1 ELSE 0 END AS is_peak_season
FROM shipments;
```

---

## 5. Window Functions

Window functions compute values across a set of rows **related to the current row** without collapsing them into a group. This is the most important advanced SQL topic for data science interviews.

### Syntax

```sql
function_name(expression)
    OVER (
        [PARTITION BY partition_columns]
        [ORDER BY sort_columns]
        [ROWS/RANGE BETWEEN frame_start AND frame_end]
    )
```

### Ranking Functions

```sql
SELECT
    carrier_id,
    route_id,
    avg_delay_minutes,
    -- ROW_NUMBER: unique sequential rank, no ties
    ROW_NUMBER() OVER (PARTITION BY carrier_id ORDER BY avg_delay_minutes DESC) AS row_num,
    -- RANK: ties get same rank, next rank skips (1,2,2,4)
    RANK() OVER (PARTITION BY carrier_id ORDER BY avg_delay_minutes DESC) AS rank_with_gaps,
    -- DENSE_RANK: ties get same rank, no skip (1,2,2,3)
    DENSE_RANK() OVER (PARTITION BY carrier_id ORDER BY avg_delay_minutes DESC) AS dense_rank,
    -- NTILE: divide into N buckets (for percentile analysis)
    NTILE(4) OVER (ORDER BY avg_delay_minutes) AS delay_quartile
FROM route_performance;
```

**Interview note**: ROW_NUMBER vs RANK vs DENSE_RANK is a classic question. Know the difference cold.

```
Delay  ROW_NUMBER  RANK  DENSE_RANK
  60        1        1       1
  45        2        2       2
  45        3        2       2
  30        4        4       3    ← RANK skips 3, DENSE_RANK does not
  15        5        5       4
```

### Offset Functions: LAG and LEAD

Critical for time series feature engineering.

```sql
-- Daily volume with day-over-day comparison
SELECT
    depot_id,
    report_date,
    daily_volume,
    LAG(daily_volume, 1) OVER (PARTITION BY depot_id ORDER BY report_date) AS prev_day_volume,
    LAG(daily_volume, 7) OVER (PARTITION BY depot_id ORDER BY report_date) AS same_day_last_week,
    LEAD(daily_volume, 1) OVER (PARTITION BY depot_id ORDER BY report_date) AS next_day_volume,
    -- Day-over-day change
    daily_volume - LAG(daily_volume, 1) OVER (PARTITION BY depot_id ORDER BY report_date) AS dod_change,
    -- Week-over-week % change
    ROUND(
        100.0 * (daily_volume - LAG(daily_volume, 7) OVER (PARTITION BY depot_id ORDER BY report_date))
        / NULLIF(LAG(daily_volume, 7) OVER (PARTITION BY depot_id ORDER BY report_date), 0),
        2
    ) AS wow_pct_change
FROM daily_depot_volumes
ORDER BY depot_id, report_date;
```

### Aggregate Window Functions

```sql
-- 7-day rolling average delivery volume by depot
SELECT
    depot_id,
    report_date,
    daily_volume,
    -- Rolling 7-day average (current day + 6 preceding)
    AVG(daily_volume) OVER (
        PARTITION BY depot_id
        ORDER BY report_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS rolling_7day_avg,
    -- Rolling 7-day sum
    SUM(daily_volume) OVER (
        PARTITION BY depot_id
        ORDER BY report_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS rolling_7day_sum,
    -- Running total (cumulative sum)
    SUM(daily_volume) OVER (
        PARTITION BY depot_id
        ORDER BY report_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_volume,
    -- Running average
    AVG(daily_volume) OVER (
        PARTITION BY depot_id
        ORDER BY report_date
    ) AS running_avg,
    -- Percentage of monthly total
    daily_volume * 1.0 / SUM(daily_volume) OVER (
        PARTITION BY depot_id, DATE_TRUNC('month', report_date)
    ) AS pct_of_monthly_volume
FROM daily_depot_volumes;
```

### Frame Specification: ROWS vs RANGE

```sql
-- ROWS BETWEEN: physical row offsets (what you usually want)
AVG(value) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)

-- RANGE BETWEEN: logical value offsets (all rows with same ORDER BY value are peers)
-- Dangerous with duplicate dates — includes ALL rows with the same date
AVG(value) OVER (ORDER BY date RANGE BETWEEN 6 PRECEDING AND CURRENT ROW)
-- For numeric order columns: RANGE 6 PRECEDING means "value >= current - 6"

-- Best practice: use ROWS BETWEEN explicitly unless you specifically need RANGE
```

### FIRST_VALUE, LAST_VALUE, NTH_VALUE

```sql
SELECT
    route_id,
    carrier_id,
    delivery_date,
    delay_minutes,
    -- First delay ever recorded on this route
    FIRST_VALUE(delay_minutes) OVER (
        PARTITION BY route_id
        ORDER BY delivery_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS first_ever_delay,
    -- Most recent delay on this route
    LAST_VALUE(delay_minutes) OVER (
        PARTITION BY route_id
        ORDER BY delivery_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING  -- critical: extend frame to end!
    ) AS last_delay,
    -- Second delivery on this route
    NTH_VALUE(delay_minutes, 2) OVER (
        PARTITION BY route_id
        ORDER BY delivery_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS second_delivery_delay
FROM deliveries;
```

**Common trap**: `LAST_VALUE` without `ROWS BETWEEN ... UNBOUNDED FOLLOWING` gives you the *current* row's value because the default frame ends at `CURRENT ROW`.

### Practical Example: Rank Routes by Performance Within Each Carrier

```sql
WITH route_metrics AS (
    SELECT
        carrier_id,
        route_id,
        COUNT(*) AS total_deliveries,
        AVG(delay_minutes) AS avg_delay,
        SUM(CASE WHEN delay_minutes <= 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS on_time_pct
    FROM deliveries
    WHERE delivery_date >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY carrier_id, route_id
    HAVING COUNT(*) >= 10  -- require statistical significance
),
ranked AS (
    SELECT
        *,
        RANK() OVER (PARTITION BY carrier_id ORDER BY on_time_pct DESC) AS rank_within_carrier,
        RANK() OVER (ORDER BY on_time_pct DESC) AS overall_rank,
        NTILE(5) OVER (ORDER BY on_time_pct DESC) AS performance_quintile
    FROM route_metrics
)
SELECT *
FROM ranked
WHERE rank_within_carrier <= 3  -- Top 3 routes per carrier
ORDER BY carrier_id, rank_within_carrier;
```

---

## 6. CTEs and Subqueries

### Common Table Expressions (WITH clause)

CTEs make complex queries readable by naming intermediate results.

```sql
-- Multi-step calculation: monthly on-time rate by carrier with trend
WITH
-- Step 1: Base delivery data with on-time flag
delivery_flags AS (
    SELECT
        carrier_id,
        DATE_TRUNC('month', delivery_date) AS delivery_month,
        shipment_id,
        CASE WHEN delay_minutes <= 0 THEN 1 ELSE 0 END AS is_on_time
    FROM deliveries
    WHERE delivery_date >= '2023-01-01'
),
-- Step 2: Monthly aggregation per carrier
monthly_metrics AS (
    SELECT
        carrier_id,
        delivery_month,
        COUNT(*) AS total_deliveries,
        SUM(is_on_time) AS on_time_count,
        SUM(is_on_time) * 100.0 / COUNT(*) AS on_time_rate
    FROM delivery_flags
    GROUP BY carrier_id, delivery_month
),
-- Step 3: Add month-over-month comparison
with_trend AS (
    SELECT
        *,
        LAG(on_time_rate, 1) OVER (
            PARTITION BY carrier_id ORDER BY delivery_month
        ) AS prev_month_rate,
        on_time_rate - LAG(on_time_rate, 1) OVER (
            PARTITION BY carrier_id ORDER BY delivery_month
        ) AS mom_change
    FROM monthly_metrics
)
-- Final output
SELECT
    carrier_id,
    delivery_month,
    total_deliveries,
    ROUND(on_time_rate, 2) AS on_time_rate_pct,
    ROUND(prev_month_rate, 2) AS prev_month_pct,
    ROUND(mom_change, 2) AS mom_change_ppt
FROM with_trend
ORDER BY carrier_id, delivery_month;
```

### Recursive CTEs: Hierarchical Data

```sql
-- Traverse depot hierarchy (hub → regional → local depot)
WITH RECURSIVE depot_hierarchy AS (
    -- Base case: root depots (no parent)
    SELECT
        depot_id,
        depot_name,
        parent_depot_id,
        0 AS depth,
        CAST(depot_id AS VARCHAR) AS path
    FROM depots
    WHERE parent_depot_id IS NULL

    UNION ALL

    -- Recursive case: children of already-processed depots
    SELECT
        d.depot_id,
        d.depot_name,
        d.parent_depot_id,
        dh.depth + 1 AS depth,
        dh.path || ' -> ' || CAST(d.depot_id AS VARCHAR) AS path
    FROM depots d
    INNER JOIN depot_hierarchy dh ON d.parent_depot_id = dh.depot_id
)
SELECT depot_id, depot_name, depth, path
FROM depot_hierarchy
ORDER BY path;
```

### Subqueries vs CTEs vs Temp Tables

| Approach | When to Use | Pros | Cons |
|----------|------------|------|------|
| Inline subquery | Simple one-off filtering | No extra syntax | Hard to read if complex |
| CTE (WITH) | Multi-step logic, readability | Readable, reusable within query | May not be materialized (optimizer dependent) |
| Temp table | Very large intermediate results, reuse across multiple queries | Always materialized, can add index | More verbose, session-scoped |
| View | Shared logic across teams/queries | Shareable, documented | Can hide complexity |

```sql
-- Temp table: materialize a large computation once, query it multiple times
CREATE TEMP TABLE carrier_90day_metrics AS
SELECT
    carrier_id,
    COUNT(*) AS deliveries,
    AVG(delay_minutes) AS avg_delay
FROM deliveries
WHERE delivery_date >= CURRENT_DATE - 90
GROUP BY carrier_id;

-- Now query it multiple times without recomputation
SELECT * FROM carrier_90day_metrics WHERE avg_delay > 30;
SELECT * FROM carrier_90day_metrics WHERE deliveries > 1000;
```

### Correlated Subqueries

A correlated subquery references columns from the outer query. They execute once per outer row — avoid for large tables.

```sql
-- For each carrier, find their latest delivery (correlated subquery — slow for large tables)
SELECT
    c.carrier_id,
    c.carrier_name,
    (SELECT MAX(delivery_date)
     FROM deliveries d
     WHERE d.carrier_id = c.carrier_id) AS latest_delivery  -- references outer c.carrier_id
FROM carriers c;

-- BETTER: use a window function or join
SELECT DISTINCT
    c.carrier_id,
    c.carrier_name,
    MAX(d.delivery_date) OVER (PARTITION BY d.carrier_id) AS latest_delivery
FROM carriers c
LEFT JOIN deliveries d ON c.carrier_id = d.carrier_id;
```

---

## 7. Advanced SQL Patterns

### PIVOT: Rotating Rows to Columns

```sql
-- Daily volume for each carrier as columns (manual pivot with CASE WHEN)
SELECT
    report_date,
    SUM(CASE WHEN carrier_id = 'FEDEX' THEN daily_volume ELSE 0 END) AS fedex_volume,
    SUM(CASE WHEN carrier_id = 'UPS'   THEN daily_volume ELSE 0 END) AS ups_volume,
    SUM(CASE WHEN carrier_id = 'USPS'  THEN daily_volume ELSE 0 END) AS usps_volume
FROM daily_carrier_volumes
GROUP BY report_date
ORDER BY report_date;
```

### Gap and Island Problems

Finding consecutive sequences (where a "gap" interrupts a run of values).

```sql
-- Find consecutive days where a depot had volume > 10,000
WITH flagged AS (
    SELECT
        depot_id,
        report_date,
        daily_volume,
        CASE WHEN daily_volume > 10000 THEN 1 ELSE 0 END AS is_high_volume
    FROM daily_depot_volumes
),
grouped AS (
    SELECT
        depot_id,
        report_date,
        is_high_volume,
        -- Create a group identifier: subtract row number from date
        -- When days are consecutive, date - row_number is constant
        report_date - CAST(ROW_NUMBER() OVER (
            PARTITION BY depot_id, is_high_volume
            ORDER BY report_date
        ) AS INTEGER) AS grp_key
    FROM flagged
    WHERE is_high_volume = 1
)
SELECT
    depot_id,
    MIN(report_date) AS streak_start,
    MAX(report_date) AS streak_end,
    COUNT(*) AS consecutive_days
FROM grouped
GROUP BY depot_id, grp_key
HAVING COUNT(*) >= 3  -- streaks of 3+ days
ORDER BY consecutive_days DESC;
```

### Session Analysis

Group events into sessions based on inactivity gaps.

```sql
-- Group customer website visits into sessions (30-min inactivity = new session)
WITH visits_with_gap AS (
    SELECT
        customer_id,
        visit_ts,
        -- Flag where a new session starts (gap > 30 min from previous)
        CASE
            WHEN visit_ts - LAG(visit_ts) OVER (
                    PARTITION BY customer_id ORDER BY visit_ts
                 ) > INTERVAL '30 minutes'
              OR LAG(visit_ts) OVER (PARTITION BY customer_id ORDER BY visit_ts) IS NULL
            THEN 1 ELSE 0
        END AS is_session_start
    FROM website_visits
),
sessions AS (
    SELECT
        customer_id,
        visit_ts,
        SUM(is_session_start) OVER (
            PARTITION BY customer_id
            ORDER BY visit_ts
        ) AS session_id
    FROM visits_with_gap
)
SELECT
    customer_id,
    session_id,
    MIN(visit_ts) AS session_start,
    MAX(visit_ts) AS session_end,
    COUNT(*) AS page_views,
    EXTRACT(EPOCH FROM MAX(visit_ts) - MIN(visit_ts)) / 60.0 AS session_duration_minutes
FROM sessions
GROUP BY customer_id, session_id;
```

### Running Totals and Moving Averages

```sql
SELECT
    depot_id,
    report_date,
    daily_volume,
    -- Simple moving average (7 day)
    ROUND(AVG(daily_volume) OVER (
        PARTITION BY depot_id
        ORDER BY report_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ), 0) AS sma_7,
    -- Exponential moving average approximation in SQL (tricky — use Python for production)
    -- Standard running total
    SUM(daily_volume) OVER (
        PARTITION BY depot_id, DATE_TRUNC('month', report_date)
        ORDER BY report_date
    ) AS month_to_date_volume
FROM daily_depot_volumes;
```

---

## 8. ML-Specific SQL Patterns

### Feature Engineering in SQL

The feature store is often populated by SQL queries. Here are production patterns.

```sql
-- Create lag and rolling features for delivery delay prediction
WITH base AS (
    SELECT
        route_id,
        delivery_date,
        delay_minutes,
        weather_severity,
        carrier_id
    FROM deliveries
    WHERE delivery_date BETWEEN '2023-01-01' AND '2024-01-01'
),
features AS (
    SELECT
        route_id,
        delivery_date,
        delay_minutes,  -- this is the label for next-day prediction
        weather_severity,
        carrier_id,
        -- Lag features (previous outcomes)
        LAG(delay_minutes, 1) OVER (PARTITION BY route_id ORDER BY delivery_date) AS delay_lag_1d,
        LAG(delay_minutes, 2) OVER (PARTITION BY route_id ORDER BY delivery_date) AS delay_lag_2d,
        LAG(delay_minutes, 7) OVER (PARTITION BY route_id ORDER BY delivery_date) AS delay_lag_7d,
        -- Rolling statistics
        AVG(delay_minutes) OVER (
            PARTITION BY route_id ORDER BY delivery_date
            ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING  -- EXCLUDE current row to avoid leakage!
        ) AS rolling_avg_delay_7d,
        STDDEV(delay_minutes) OVER (
            PARTITION BY route_id ORDER BY delivery_date
            ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING
        ) AS rolling_std_delay_7d,
        MAX(delay_minutes) OVER (
            PARTITION BY route_id ORDER BY delivery_date
            ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING
        ) AS rolling_max_delay_7d,
        -- Calendar features
        EXTRACT(DOW FROM delivery_date) AS day_of_week,
        EXTRACT(MONTH FROM delivery_date) AS month,
        CASE WHEN EXTRACT(DOW FROM delivery_date) IN (0, 6) THEN 1 ELSE 0 END AS is_weekend
    FROM base
)
-- Exclude early rows where lag/rolling features are NULL
SELECT *
FROM features
WHERE delay_lag_7d IS NOT NULL  -- enough history exists
ORDER BY route_id, delivery_date;
```

### Train/Test Split in SQL

```sql
-- Temporal split: last 3 months as test set
WITH labeled_data AS (
    SELECT
        *,
        CASE
            WHEN delivery_date >= CURRENT_DATE - INTERVAL '90 days' THEN 'test'
            ELSE 'train'
        END AS split
    FROM ml_features
)
-- Training set
SELECT * FROM labeled_data WHERE split = 'train';
-- Test set
SELECT * FROM labeled_data WHERE split = 'test';

-- Random split (for non-time-series data — use hash for reproducibility)
SELECT
    *,
    CASE
        WHEN ABS(HASHTEXT(CAST(customer_id AS TEXT))) % 10 < 8 THEN 'train'
        ELSE 'test'
    END AS split
FROM customer_features;
-- HASHTEXT (PostgreSQL) gives deterministic hashing — same customer always in same split
```

### Computing Classification Metrics in SQL

```sql
-- Compute precision, recall, F1 from a predictions table
WITH confusion AS (
    SELECT
        SUM(CASE WHEN predicted = 1 AND actual = 1 THEN 1 ELSE 0 END) AS tp,
        SUM(CASE WHEN predicted = 1 AND actual = 0 THEN 1 ELSE 0 END) AS fp,
        SUM(CASE WHEN predicted = 0 AND actual = 1 THEN 1 ELSE 0 END) AS fn,
        SUM(CASE WHEN predicted = 0 AND actual = 0 THEN 1 ELSE 0 END) AS tn
    FROM model_predictions
)
SELECT
    tp, fp, fn, tn,
    ROUND(tp * 1.0 / NULLIF(tp + fp, 0), 4) AS precision,
    ROUND(tp * 1.0 / NULLIF(tp + fn, 0), 4) AS recall,
    ROUND(2.0 * tp / NULLIF(2 * tp + fp + fn, 0), 4) AS f1_score,
    ROUND((tp + tn) * 1.0 / (tp + fp + fn + tn), 4) AS accuracy
FROM confusion;
```

### Cohort Analysis

```sql
-- Customer retention by acquisition cohort
WITH cohorts AS (
    SELECT
        customer_id,
        DATE_TRUNC('month', MIN(order_date)) AS cohort_month
    FROM orders
    GROUP BY customer_id
),
orders_with_cohort AS (
    SELECT
        o.customer_id,
        o.order_date,
        c.cohort_month,
        -- Months since cohort month
        EXTRACT(YEAR FROM AGE(DATE_TRUNC('month', o.order_date), c.cohort_month)) * 12
        + EXTRACT(MONTH FROM AGE(DATE_TRUNC('month', o.order_date), c.cohort_month)) AS periods_since_cohort
    FROM orders o
    JOIN cohorts c ON o.customer_id = c.customer_id
),
cohort_data AS (
    SELECT
        cohort_month,
        periods_since_cohort,
        COUNT(DISTINCT customer_id) AS active_customers
    FROM orders_with_cohort
    GROUP BY cohort_month, periods_since_cohort
),
cohort_sizes AS (
    SELECT cohort_month, active_customers AS cohort_size
    FROM cohort_data
    WHERE periods_since_cohort = 0
)
SELECT
    cd.cohort_month,
    cd.periods_since_cohort,
    cd.active_customers,
    cs.cohort_size,
    ROUND(100.0 * cd.active_customers / cs.cohort_size, 1) AS retention_rate
FROM cohort_data cd
JOIN cohort_sizes cs ON cd.cohort_month = cs.cohort_month
ORDER BY cd.cohort_month, cd.periods_since_cohort;
```

### A/B Test Results in SQL

```sql
-- Compute statistical significance of an A/B test on conversion rate
WITH group_stats AS (
    SELECT
        treatment_group,
        COUNT(*) AS n,
        SUM(converted) AS conversions,
        AVG(converted) AS conversion_rate,
        AVG(converted) * (1 - AVG(converted)) AS variance
    FROM ab_test_results
    WHERE experiment_id = 'EXP_2024_01'
    GROUP BY treatment_group
),
control AS (
    SELECT n, conversions, conversion_rate, variance
    FROM group_stats WHERE treatment_group = 'control'
),
treatment AS (
    SELECT n, conversions, conversion_rate, variance
    FROM group_stats WHERE treatment_group = 'treatment'
)
SELECT
    c.conversion_rate AS control_rate,
    t.conversion_rate AS treatment_rate,
    t.conversion_rate - c.conversion_rate AS absolute_lift,
    ROUND(100.0 * (t.conversion_rate - c.conversion_rate) / c.conversion_rate, 2) AS relative_lift_pct,
    -- Standard error of difference
    SQRT(c.variance / c.n + t.variance / t.n) AS se_diff,
    -- Z-score
    (t.conversion_rate - c.conversion_rate) / SQRT(c.variance / c.n + t.variance / t.n) AS z_score
FROM control c, treatment t;
-- p-value must be looked up from z_score using a statistics table or computed in Python
-- For z_score > 1.96: p < 0.05 (two-tailed, α=0.05)
```

### Funnel Analysis

```sql
-- Customer journey funnel: tracking → order → delivery → review
WITH funnel_steps AS (
    SELECT
        customer_id,
        MAX(CASE WHEN event_type = 'TRACKING_VIEWED' THEN 1 ELSE 0 END) AS viewed_tracking,
        MAX(CASE WHEN event_type = 'ORDER_PLACED' THEN 1 ELSE 0 END) AS placed_order,
        MAX(CASE WHEN event_type = 'PACKAGE_DELIVERED' THEN 1 ELSE 0 END) AS received_delivery,
        MAX(CASE WHEN event_type = 'REVIEW_SUBMITTED' THEN 1 ELSE 0 END) AS left_review
    FROM customer_events
    WHERE event_date >= '2024-01-01'
    GROUP BY customer_id
)
SELECT
    COUNT(*) AS total_customers,
    SUM(viewed_tracking) AS step1_tracking,
    SUM(placed_order) AS step2_order,
    SUM(received_delivery) AS step3_delivered,
    SUM(left_review) AS step4_review,
    -- Conversion rates
    ROUND(100.0 * SUM(placed_order) / COUNT(*), 1) AS order_conversion_pct,
    ROUND(100.0 * SUM(received_delivery) / NULLIF(SUM(placed_order), 0), 1) AS delivery_pct,
    ROUND(100.0 * SUM(left_review) / NULLIF(SUM(received_delivery), 0), 1) AS review_pct
FROM funnel_steps;
```

---

## 9. Query Performance and Optimization

### EXPLAIN / EXPLAIN ANALYZE

```sql
-- PostgreSQL: see query plan without running
EXPLAIN SELECT * FROM deliveries WHERE carrier_id = 'FEDEX';

-- PostgreSQL: run the query and see actual timing
EXPLAIN ANALYZE
SELECT carrier_id, AVG(delay_minutes)
FROM deliveries
WHERE delivery_date >= '2024-01-01'
GROUP BY carrier_id;

-- Key things to look for in query plan:
-- Seq Scan: full table scan (bad for large tables without index)
-- Index Scan: using an index (good)
-- Hash Join vs Nested Loop vs Merge Join: optimizer's join strategy
-- Sort: expensive, look for opportunities to exploit indexes
-- Actual rows vs Estimated rows: large discrepancy means stale statistics
```

### Index Fundamentals

```sql
-- B-tree index: default, good for equality and range queries
CREATE INDEX idx_deliveries_carrier ON deliveries(carrier_id);
CREATE INDEX idx_deliveries_date ON deliveries(delivery_date);

-- Composite index: column order matters!
-- Useful for queries filtering on both carrier_id AND delivery_date
CREATE INDEX idx_deliveries_carrier_date ON deliveries(carrier_id, delivery_date);
-- This index helps: WHERE carrier_id = 'X' AND delivery_date > 'Y'
-- This index DOESN'T help: WHERE delivery_date > 'Y' (leading column not used)

-- Covering index: include additional columns to avoid table lookup
CREATE INDEX idx_deliveries_covering ON deliveries(carrier_id, delivery_date)
    INCLUDE (delay_minutes, status);
-- Query can be satisfied entirely from the index without reading the main table

-- Partial index: only index a subset of rows
CREATE INDEX idx_delayed_deliveries ON deliveries(delivery_date)
    WHERE delay_minutes > 0;
-- Much smaller index if most deliveries are on-time
```

### Partitioning

```sql
-- Range partition a large deliveries table by month
CREATE TABLE deliveries (
    shipment_id BIGINT,
    delivery_date DATE,
    carrier_id VARCHAR(20),
    delay_minutes INTEGER
) PARTITION BY RANGE (delivery_date);

CREATE TABLE deliveries_2024_01 PARTITION OF deliveries
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE deliveries_2024_02 PARTITION OF deliveries
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- Queries with WHERE delivery_date BETWEEN '2024-01-01' AND '2024-01-31'
-- will only scan the January partition (partition pruning)
```

### Common Performance Anti-Patterns

```sql
-- ANTI-PATTERN 1: SELECT * — reads unnecessary columns, prevents index-only scans
SELECT * FROM deliveries WHERE carrier_id = 'FEDEX';  -- Bad
SELECT shipment_id, delivery_date, delay_minutes FROM deliveries WHERE carrier_id = 'FEDEX';  -- Good

-- ANTI-PATTERN 2: Functions on indexed columns in WHERE (index cannot be used)
WHERE EXTRACT(YEAR FROM delivery_date) = 2024  -- Bad: function prevents index use
WHERE delivery_date BETWEEN '2024-01-01' AND '2024-12-31'  -- Good: range scan uses index

-- ANTI-PATTERN 3: OR conditions (may prevent index use)
WHERE carrier_id = 'FEDEX' OR carrier_id = 'UPS'  -- May not use index efficiently
WHERE carrier_id IN ('FEDEX', 'UPS')  -- Better

-- ANTI-PATTERN 4: Leading wildcard prevents index use
WHERE carrier_name LIKE '%express%'  -- Bad: full scan required
WHERE carrier_name LIKE 'express%'   -- Good: index usable

-- ANTI-PATTERN 5: Correlated subquery executed N times
-- Already shown above — replace with JOIN or window function
```

---

## 10. Interview Questions with Complete Solutions

### Q1: Second-Highest Delivery Delay

```sql
-- Method 1: DENSE_RANK (handles ties correctly)
SELECT delay_minutes
FROM (
    SELECT delay_minutes,
           DENSE_RANK() OVER (ORDER BY delay_minutes DESC) AS rnk
    FROM deliveries
) ranked
WHERE rnk = 2;

-- Method 2: Subquery with NOT IN
SELECT MAX(delay_minutes) AS second_highest
FROM deliveries
WHERE delay_minutes < (SELECT MAX(delay_minutes) FROM deliveries);

-- Method 3: OFFSET (simple but returns one row)
SELECT DISTINCT delay_minutes
FROM deliveries
ORDER BY delay_minutes DESC
LIMIT 1 OFFSET 1;
-- OFFSET 0 = first highest, OFFSET 1 = second highest
```

### Q2: 7-Day Rolling Average of Package Volume Per Region

```sql
-- Requirement: for each region and each day, compute avg of that day + 6 preceding days
WITH daily_volume AS (
    SELECT
        region_id,
        DATE_TRUNC('day', ship_date) AS report_date,
        COUNT(*) AS package_volume
    FROM shipments
    GROUP BY region_id, DATE_TRUNC('day', ship_date)
)
SELECT
    region_id,
    report_date,
    package_volume,
    ROUND(
        AVG(package_volume) OVER (
            PARTITION BY region_id
            ORDER BY report_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ),
        1
    ) AS rolling_7day_avg
FROM daily_volume
ORDER BY region_id, report_date;
```

### Q3: Customers in January But Not February

```sql
-- Method 1: LEFT JOIN with NULL check
SELECT DISTINCT jan.customer_id
FROM (
    SELECT DISTINCT customer_id
    FROM deliveries
    WHERE delivery_date >= '2024-01-01' AND delivery_date < '2024-02-01'
) jan
LEFT JOIN (
    SELECT DISTINCT customer_id
    FROM deliveries
    WHERE delivery_date >= '2024-02-01' AND delivery_date < '2024-03-01'
) feb ON jan.customer_id = feb.customer_id
WHERE feb.customer_id IS NULL;

-- Method 2: EXCEPT (cleaner)
SELECT DISTINCT customer_id FROM deliveries
WHERE delivery_date >= '2024-01-01' AND delivery_date < '2024-02-01'
EXCEPT
SELECT DISTINCT customer_id FROM deliveries
WHERE delivery_date >= '2024-02-01' AND delivery_date < '2024-03-01';

-- Method 3: NOT IN (watch out for NULLs!)
SELECT DISTINCT customer_id FROM deliveries
WHERE delivery_date >= '2024-01-01' AND delivery_date < '2024-02-01'
AND customer_id NOT IN (
    SELECT DISTINCT customer_id FROM deliveries
    WHERE delivery_date >= '2024-02-01' AND delivery_date < '2024-03-01'
    AND customer_id IS NOT NULL  -- critical: NOT IN with NULLs returns no rows!
);
```

### Q4: Median Delivery Time by Carrier (No Native MEDIAN)

```sql
-- Use PERCENTILE_CONT (available in most modern warehouses)
SELECT
    carrier_id,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY delivery_time_hours) AS median_delivery_hrs,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY delivery_time_hours) AS p25_delivery_hrs,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY delivery_time_hours) AS p75_delivery_hrs
FROM deliveries
GROUP BY carrier_id;

-- Manual median if PERCENTILE_CONT not available (double-pass with ROW_NUMBER)
WITH numbered AS (
    SELECT
        carrier_id,
        delivery_time_hours,
        ROW_NUMBER() OVER (PARTITION BY carrier_id ORDER BY delivery_time_hours) AS rn,
        COUNT(*) OVER (PARTITION BY carrier_id) AS total_rows
    FROM deliveries
)
SELECT
    carrier_id,
    AVG(delivery_time_hours) AS median_delivery_hrs  -- average of 1 or 2 middle values
FROM numbered
WHERE rn IN (FLOOR((total_rows + 1) / 2.0), CEIL((total_rows + 1) / 2.0))
GROUP BY carrier_id;
```

### Q5: Find Consecutive Days with Volume > 10,000

```sql
-- Classic gap-and-island solution
WITH high_volume_days AS (
    SELECT
        depot_id,
        report_date,
        daily_volume
    FROM daily_depot_volumes
    WHERE daily_volume > 10000
),
with_island_key AS (
    SELECT
        depot_id,
        report_date,
        daily_volume,
        -- When days are consecutive, date minus sequential row_number is constant
        report_date - (ROW_NUMBER() OVER (
            PARTITION BY depot_id ORDER BY report_date
        ) * INTERVAL '1 day') AS island_key
    FROM high_volume_days
)
SELECT
    depot_id,
    MIN(report_date) AS streak_start,
    MAX(report_date) AS streak_end,
    COUNT(*) AS consecutive_days,
    AVG(daily_volume) AS avg_volume_during_streak
FROM with_island_key
GROUP BY depot_id, island_key
ORDER BY consecutive_days DESC;
```

### Q6: Customer Retention Rate Month Over Month

```sql
WITH monthly_actives AS (
    SELECT
        DATE_TRUNC('month', order_date) AS activity_month,
        customer_id
    FROM orders
    GROUP BY DATE_TRUNC('month', order_date), customer_id
),
retention AS (
    SELECT
        curr.activity_month,
        COUNT(DISTINCT curr.customer_id) AS current_month_customers,
        COUNT(DISTINCT prev.customer_id) AS retained_from_prev_month
    FROM monthly_actives curr
    LEFT JOIN monthly_actives prev
        ON curr.customer_id = prev.customer_id
        AND prev.activity_month = curr.activity_month - INTERVAL '1 month'
    GROUP BY curr.activity_month
)
SELECT
    activity_month,
    current_month_customers,
    retained_from_prev_month,
    ROUND(100.0 * retained_from_prev_month / LAG(current_month_customers, 1) OVER (
        ORDER BY activity_month
    ), 2) AS retention_rate_pct
FROM retention
ORDER BY activity_month;
```

### Bonus: On-Time Delivery Rate by Carrier with Statistical Context

```sql
-- Production-quality OTD metric with confidence intervals
WITH carrier_stats AS (
    SELECT
        carrier_id,
        COUNT(*) AS n,
        SUM(CASE WHEN delay_minutes <= 0 THEN 1 ELSE 0 END) AS on_time_count,
        AVG(CASE WHEN delay_minutes <= 0 THEN 1.0 ELSE 0.0 END) AS on_time_rate
    FROM deliveries
    WHERE delivery_date >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY carrier_id
    HAVING COUNT(*) >= 50  -- minimum sample size for meaningful estimate
)
SELECT
    carrier_id,
    n AS sample_size,
    ROUND(100.0 * on_time_rate, 2) AS on_time_rate_pct,
    -- 95% confidence interval for a proportion: p ± 1.96 * sqrt(p(1-p)/n)
    ROUND(100.0 * (on_time_rate - 1.96 * SQRT(on_time_rate * (1 - on_time_rate) / n)), 2) AS ci_lower,
    ROUND(100.0 * (on_time_rate + 1.96 * SQRT(on_time_rate * (1 - on_time_rate) / n)), 2) AS ci_upper
FROM carrier_stats
ORDER BY on_time_rate DESC;
```

---

## Quick Reference: SQL Cheat Sheet

```sql
-- Window function template
<AGG_FUNC>(<col>) OVER (
    PARTITION BY <group_col>
    ORDER BY <sort_col>
    ROWS BETWEEN <start> AND <end>
)
-- Frame options: UNBOUNDED PRECEDING, N PRECEDING, CURRENT ROW, N FOLLOWING, UNBOUNDED FOLLOWING

-- CTE template
WITH cte_name AS (
    SELECT ...
),
cte_name_2 AS (
    SELECT ... FROM cte_name
)
SELECT ... FROM cte_name_2;

-- Find duplicates
SELECT col1, col2, COUNT(*) FROM table GROUP BY col1, col2 HAVING COUNT(*) > 1;

-- Deduplicate (keep latest row per key)
SELECT * FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY key_col ORDER BY updated_at DESC) AS rn
    FROM table
) t WHERE rn = 1;

-- Null-safe division
numerator * 1.0 / NULLIF(denominator, 0)

-- Convert NULL to zero
COALESCE(possibly_null_col, 0)
```

---

*Master these patterns and you can write the SQL that feeds real production ML systems. The gap between knowing SQL syntax and writing production feature queries is the gap between junior and senior.*
