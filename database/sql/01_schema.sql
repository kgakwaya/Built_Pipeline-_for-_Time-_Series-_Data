-- =====================================================================
--  Household electricity consumption  |  Relational schema (MySQL 8 / MariaDB 10.5+)
--  Formative 1 - Task 2.  Source data: hourly aggregation of the UCI
--  "Individual household electric power consumption" minute data.
-- =====================================================================
--  Design summary
--    households      1 --- N  hourly_readings      (one row per household per hour)
--    hourly_readings 1 --- N  circuit_energy       (energy per sub-meter / circuit, long format)
--    circuits        1 --- N  circuit_energy
--    households      1 --- N  circuits
--    ml_models       1 --- N  forecasts            (model registry -> predictions)
--    households      1 --- N  forecasts
--  Time is stored as DATETIME in the household's *local clock time*: the source file
--  carries no time zone, so no conversion is applied (see docs/database_design.md).
-- =====================================================================

CREATE DATABASE IF NOT EXISTS power_db
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE power_db;

-- Drop in dependency order so the script can be re-run safely.
DROP VIEW  IF EXISTS v_hourly_wide;
DROP TABLE IF EXISTS forecasts;
DROP TABLE IF EXISTS ml_models;
DROP TABLE IF EXISTS circuit_energy;
DROP TABLE IF EXISTS hourly_readings;
DROP TABLE IF EXISTS circuits;
DROP TABLE IF EXISTS households;

-- ---------------------------------------------------------------------
-- 1. households : who/what is being measured
-- ---------------------------------------------------------------------
CREATE TABLE households (
    household_id  INT UNSIGNED  NOT NULL AUTO_INCREMENT,
    name          VARCHAR(100)  NOT NULL,
    description   VARCHAR(255)  NULL,
    data_source   VARCHAR(255)  NOT NULL,
    created_at    TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (household_id),
    UNIQUE KEY uq_households_name (name)
) ENGINE=InnoDB COMMENT='Metered households (this dataset contains one).';

-- ---------------------------------------------------------------------
-- 2. circuits : the metered circuits of a household (dimension table)
--    is_derived = 1 for "other" energy, which is computed per hour:
--    global_active_power_kw*1000 - (sub1 + sub2 + sub3)  [Wh]  (not sub-metered consumption)
-- ---------------------------------------------------------------------
CREATE TABLE circuits (
    circuit_id    INT UNSIGNED  NOT NULL AUTO_INCREMENT,
    household_id  INT UNSIGNED  NOT NULL,
    code          VARCHAR(30)   NOT NULL,
    name          VARCHAR(100)  NOT NULL,
    description   VARCHAR(255)  NULL,
    unit          VARCHAR(10)   NOT NULL DEFAULT 'Wh',
    is_derived    TINYINT(1)    NOT NULL DEFAULT 0,
    PRIMARY KEY (circuit_id),
    UNIQUE KEY uq_circuits_household_code (household_id, code),
    CONSTRAINT fk_circuits_household FOREIGN KEY (household_id)
        REFERENCES households (household_id) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB COMMENT='Sub-metered circuits plus the derived "other" bucket.';

-- ---------------------------------------------------------------------
-- 3. hourly_readings : the time series (one row per household-hour)
--    Measurement columns are NULL for hours that failed the quality rule
--    (is_valid_hour = 0, i.e. fewer than 45 of 60 minutes observed). Those
--    hours are kept so the hourly grid stays regular (lags need it).
-- ---------------------------------------------------------------------
CREATE TABLE hourly_readings (
    reading_id               BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    household_id             INT UNSIGNED    NOT NULL,
    reading_ts               DATETIME        NOT NULL COMMENT 'start of the hour, local clock time',
    global_active_power_kw   DOUBLE          NULL COMMENT 'TARGET: hourly mean active power (kW)',
    global_reactive_power_kvar DOUBLE        NULL,
    voltage_v                DOUBLE          NULL,
    global_intensity_a       DOUBLE          NULL,
    n_obs_minutes            TINYINT UNSIGNED NOT NULL DEFAULT 0 COMMENT 'minutes observed in the hour (0-60)',
    is_valid_hour            TINYINT(1)      NOT NULL DEFAULT 0 COMMENT '1 if enough minutes were observed',
    created_at               TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at               TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (reading_id),
    UNIQUE KEY uq_readings_household_ts (household_id, reading_ts),   -- one row per hour; also serves latest/range queries
    KEY ix_readings_ts (reading_ts),
    CONSTRAINT fk_readings_household FOREIGN KEY (household_id)
        REFERENCES households (household_id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT ck_readings_nobs  CHECK (n_obs_minutes BETWEEN 0 AND 60),
    CONSTRAINT ck_readings_gap   CHECK (global_active_power_kw IS NULL OR global_active_power_kw >= 0),
    CONSTRAINT ck_readings_volt  CHECK (voltage_v IS NULL OR voltage_v >= 0)
) ENGINE=InnoDB COMMENT='Hourly grid-level measurements.';

-- ---------------------------------------------------------------------
-- 4. circuit_energy : energy per circuit per hour (long format, N rows/hour)
--    Adding another meter = adding a circuits row, not altering the schema.
-- ---------------------------------------------------------------------
CREATE TABLE circuit_energy (
    reading_id    BIGINT UNSIGNED NOT NULL,
    circuit_id    INT UNSIGNED    NOT NULL,
    energy_wh     DOUBLE          NOT NULL COMMENT 'watt-hours consumed during the hour (sum of the minute values)',
    PRIMARY KEY (reading_id, circuit_id),
    KEY ix_energy_circuit (circuit_id),
    CONSTRAINT fk_energy_reading FOREIGN KEY (reading_id)
        REFERENCES hourly_readings (reading_id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT fk_energy_circuit FOREIGN KEY (circuit_id)
        REFERENCES circuits (circuit_id) ON DELETE RESTRICT ON UPDATE CASCADE,
    CONSTRAINT ck_energy_nonneg CHECK (energy_wh >= 0)
) ENGINE=InnoDB COMMENT='Per-circuit hourly energy.';

-- ---------------------------------------------------------------------
-- 5. ml_models : registry of trained models (links Task 1 -> Task 4)
-- ---------------------------------------------------------------------
CREATE TABLE ml_models (
    model_id        INT UNSIGNED NOT NULL AUTO_INCREMENT,
    experiment_id   VARCHAR(20)  NOT NULL COMMENT 'e.g. E4 from the experiment table',
    model_name      VARCHAR(100) NOT NULL,
    version         VARCHAR(30)  NOT NULL DEFAULT '1.0',
    target          VARCHAR(60)  NOT NULL DEFAULT 'Global_active_power',
    horizon_hours   SMALLINT UNSIGNED NOT NULL DEFAULT 1,
    hyperparameters JSON         NULL,
    val_mae         DOUBLE       NULL,
    test_mae        DOUBLE       NULL,
    test_rmse       DOUBLE       NULL,
    test_r2         DOUBLE       NULL,
    trained_at      DATETIME     NULL,
    created_at      TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (model_id),
    UNIQUE KEY uq_models_exp_version (experiment_id, version)
) ENGINE=InnoDB COMMENT='Registered forecasting models.';

-- ---------------------------------------------------------------------
-- 6. forecasts : predictions produced by the prediction script
--    One row per (household, model, target hour). actual_kw can be filled in
--    later once the true value is observed, enabling live error tracking.
-- ---------------------------------------------------------------------
CREATE TABLE forecasts (
    forecast_id    BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    household_id   INT UNSIGNED NOT NULL,
    model_id       INT UNSIGNED NOT NULL,
    made_at        DATETIME     NOT NULL COMMENT 'time of the last observation used (forecast origin)',
    target_ts      DATETIME     NOT NULL COMMENT 'hour being predicted = made_at + horizon',
    predicted_kw   DOUBLE       NOT NULL,
    actual_kw      DOUBLE       NULL,
    created_at     TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (forecast_id),
    UNIQUE KEY uq_forecast (household_id, model_id, target_ts),
    KEY ix_forecasts_target (target_ts),
    CONSTRAINT fk_forecasts_household FOREIGN KEY (household_id)
        REFERENCES households (household_id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT fk_forecasts_model FOREIGN KEY (model_id)
        REFERENCES ml_models (model_id) ON DELETE RESTRICT ON UPDATE CASCADE,
    CONSTRAINT ck_forecast_pred CHECK (predicted_kw >= 0)
) ENGINE=InnoDB COMMENT='Stored forecasts.';

-- ---------------------------------------------------------------------
-- Convenience view: wide layout identical to the modelling dataset in Task 1
-- ---------------------------------------------------------------------
CREATE VIEW v_hourly_wide AS
SELECT  r.reading_id, r.household_id, r.reading_ts,
        r.global_active_power_kw, r.global_reactive_power_kvar, r.voltage_v, r.global_intensity_a,
        MAX(CASE WHEN c.code = 'sub_metering_1' THEN e.energy_wh END) AS sub_metering_1_wh,
        MAX(CASE WHEN c.code = 'sub_metering_2' THEN e.energy_wh END) AS sub_metering_2_wh,
        MAX(CASE WHEN c.code = 'sub_metering_3' THEN e.energy_wh END) AS sub_metering_3_wh,
        MAX(CASE WHEN c.code = 'other'          THEN e.energy_wh END) AS other_energy_wh,
        r.n_obs_minutes, r.is_valid_hour
FROM hourly_readings r
LEFT JOIN circuit_energy e ON e.reading_id = r.reading_id
LEFT JOIN circuits c       ON c.circuit_id = e.circuit_id
GROUP BY r.reading_id, r.household_id, r.reading_ts, r.global_active_power_kw, r.global_reactive_power_kvar,
         r.voltage_v, r.global_intensity_a, r.n_obs_minutes, r.is_valid_hour;
