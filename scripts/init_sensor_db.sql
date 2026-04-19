-- Schema and tables for sensor / maintenance data
CREATE SCHEMA IF NOT EXISTS maintenance;

-- Sensor catalog: one row per physical sensor
CREATE TABLE IF NOT EXISTS maintenance.sensor_catalog (
    sensor_id       TEXT PRIMARY KEY,
    tag             TEXT NOT NULL,
    base_tag        TEXT,
    sensor_name     TEXT NOT NULL,
    machine         TEXT NOT NULL,
    machine_type    TEXT,
    machine_location TEXT,
    unit            TEXT,
    nominal_value   DOUBLE PRECISION,
    warn_lo         DOUBLE PRECISION,
    warn_hi         DOUBLE PRECISION,
    crit_lo         DOUBLE PRECISION,
    crit_hi         DOUBLE PRECISION,
    fault_correlation TEXT,
    sampling_rate_sec INTEGER,
    active          BOOLEAN
);

-- Sensor readings: time-series data (~800k rows)
CREATE TABLE IF NOT EXISTS maintenance.sensor_readings (
    id              BIGINT PRIMARY KEY,
    timestamp       TIMESTAMP NOT NULL,
    machine         TEXT NOT NULL,
    machine_type    TEXT,
    tag             TEXT NOT NULL,
    sensor_name     TEXT,
    value           DOUBLE PRECISION,
    unit            TEXT,
    status          TEXT,
    warn_lo         DOUBLE PRECISION,
    warn_hi         DOUBLE PRECISION,
    fault_corr      TEXT
);

-- Component remaining life estimates
CREATE TABLE IF NOT EXISTS maintenance.remaining_life (
    component_id    TEXT PRIMARY KEY,
    component_name  TEXT NOT NULL,
    machine         TEXT NOT NULL,
    machine_type    TEXT,
    install_date    DATE,
    expected_life_hours DOUBLE PRECISION,
    current_hours   DOUBLE PRECISION,
    remaining_hours DOUBLE PRECISION,
    remaining_pct   DOUBLE PRECISION,
    condition       TEXT,
    unit_cost_eur   DOUBLE PRECISION,
    last_inspection DATE,
    next_inspection DATE,
    notes           TEXT
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_readings_machine_ts
    ON maintenance.sensor_readings (machine, timestamp);

CREATE INDEX IF NOT EXISTS idx_readings_tag_ts
    ON maintenance.sensor_readings (tag, timestamp);

CREATE INDEX IF NOT EXISTS idx_remaining_life_machine
    ON maintenance.remaining_life (machine);
