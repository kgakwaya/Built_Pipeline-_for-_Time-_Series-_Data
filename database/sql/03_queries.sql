-- =====================================================================
--  Analytical queries on power_db (MySQL 8+ / MariaDB 10.5+).
--  Each block starts with "-- name: <id>" followed by a description line;
--  database/scripts/run_sql_queries.py executes every block and stores the
--  result in database/results/sql_query_results.md.
-- =====================================================================

-- name: Q1_dataset_coverage
-- desc: Row counts, time range and data-quality summary (joins households -> hourly_readings).
SELECT h.name AS household,
       MIN(r.reading_ts)                              AS first_hour,
       MAX(r.reading_ts)                              AS last_hour,
       COUNT(*)                                       AS hours_total,
       SUM(r.is_valid_hour)                           AS hours_valid,
       COUNT(*) - SUM(r.is_valid_hour)                AS hours_invalid,
       ROUND(100 * (COUNT(*) - SUM(r.is_valid_hour)) / COUNT(*), 2) AS pct_invalid,
       ROUND(AVG(r.global_active_power_kw), 4)        AS mean_kw
FROM hourly_readings r
JOIN households h ON h.household_id = r.household_id
GROUP BY h.household_id, h.name;

-- name: Q2_monthly_energy_by_year
-- desc: Monthly energy use (kWh) and average load; each hour of mean kW equals kWh, so SUM(kW) = kWh. Only valid hours counted. Shows the winter peak / summer trough.
SELECT YEAR(reading_ts)  AS yr,
       MONTH(reading_ts) AS mon,
       COUNT(*)          AS valid_hours,
       ROUND(SUM(global_active_power_kw), 1) AS energy_kwh,
       ROUND(AVG(global_active_power_kw), 3) AS avg_kw,
       ROUND(MAX(global_active_power_kw), 3) AS peak_hour_kw
FROM hourly_readings
WHERE is_valid_hour = 1
GROUP BY YEAR(reading_ts), MONTH(reading_ts)
ORDER BY yr, mon
LIMIT 15;

-- name: Q3_avg_load_by_hour_weekday_vs_weekend
-- desc: Average load profile by hour of day, split weekday/weekend (supports the daily-cycle analysis, Q2 of the notebook).
SELECT HOUR(reading_ts) AS hour_of_day,
       ROUND(AVG(CASE WHEN WEEKDAY(reading_ts) < 5  THEN global_active_power_kw END), 3) AS avg_kw_weekday,
       ROUND(AVG(CASE WHEN WEEKDAY(reading_ts) >= 5 THEN global_active_power_kw END), 3) AS avg_kw_weekend
FROM hourly_readings
WHERE is_valid_hour = 1
GROUP BY HOUR(reading_ts)
ORDER BY hour_of_day;

-- name: Q4_energy_share_by_circuit
-- desc: Share of total energy by circuit (3-table join: readings -> circuit_energy -> circuits).
SELECT c.name AS circuit,
       c.is_derived,
       ROUND(SUM(e.energy_wh) / 1000, 1) AS total_kwh,
       ROUND(100 * SUM(e.energy_wh) / SUM(SUM(e.energy_wh)) OVER (), 2) AS pct_of_total
FROM circuit_energy e
JOIN circuits c        ON c.circuit_id = e.circuit_id
JOIN hourly_readings r ON r.reading_id = e.reading_id
WHERE r.is_valid_hour = 1
GROUP BY c.circuit_id, c.name, c.is_derived
ORDER BY total_kwh DESC;

-- name: Q5_top10_peak_hours
-- desc: The ten highest-consumption hours with their voltage and the water-heater/AC energy at that hour.
SELECT r.reading_ts, ROUND(r.global_active_power_kw, 3) AS kw,
       ROUND(r.voltage_v, 1) AS voltage_v,
       ROUND(e.energy_wh, 0) AS water_heater_ac_wh
FROM hourly_readings r
JOIN circuit_energy e ON e.reading_id = r.reading_id
JOIN circuits c       ON c.circuit_id = e.circuit_id AND c.code = 'sub_metering_3'
WHERE r.is_valid_hour = 1
ORDER BY r.global_active_power_kw DESC
LIMIT 10;

-- name: Q6_lag_and_moving_average_window_functions
-- desc: Lagged value (1 h, 24 h) and 24-hour moving average computed in SQL with window functions for one day (ties to the lag / moving-average features of Task 1).
SELECT reading_ts,
       ROUND(global_active_power_kw, 3)                                            AS kw,
       ROUND(LAG(global_active_power_kw, 1)  OVER w, 3)                            AS kw_lag_1h,
       ROUND(LAG(global_active_power_kw, 24) OVER w, 3)                            AS kw_lag_24h,
       ROUND(AVG(global_active_power_kw) OVER (w ROWS BETWEEN 23 PRECEDING AND CURRENT ROW), 3) AS kw_ma_24h
FROM hourly_readings
WHERE household_id = 1
WINDOW w AS (ORDER BY reading_ts)
ORDER BY reading_ts
LIMIT 8 OFFSET 2000;

-- name: Q7_longest_missing_data_gaps
-- desc: Gaps-and-islands: the longest runs of consecutive invalid hours (important for the preprocessing decisions).
SELECT MIN(reading_ts) AS gap_start, MAX(reading_ts) AS gap_end, COUNT(*) AS hours_missing
FROM (
    SELECT reading_ts, is_valid_hour,
           ROW_NUMBER() OVER (ORDER BY reading_ts)
         - ROW_NUMBER() OVER (PARTITION BY is_valid_hour ORDER BY reading_ts) AS grp
    FROM hourly_readings
) t
WHERE is_valid_hour = 0
GROUP BY grp
ORDER BY hours_missing DESC
LIMIT 5;

-- name: Q8_voltage_band_vs_load
-- desc: Does consumption differ between low, normal and high supply voltage bands? (bucketing with CASE)
SELECT CASE WHEN voltage_v < 238 THEN '1: < 238 V'
            WHEN voltage_v < 242 THEN '2: 238-242 V'
            ELSE '3: >= 242 V' END AS voltage_band,
       COUNT(*) AS hours,
       ROUND(AVG(global_active_power_kw), 3) AS avg_kw
FROM hourly_readings
WHERE is_valid_hour = 1
GROUP BY voltage_band
ORDER BY voltage_band;
