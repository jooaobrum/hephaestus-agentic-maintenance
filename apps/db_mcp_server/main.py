from fastmcp import FastMCP
from sqlalchemy import create_engine, text
import pandas as pd
from core.config import settings

mcp = FastMCP("db-mcp-server")

pg_engine = create_engine(settings.PG_URL)


@mcp.tool()
def get_sensor_catalog(machine: str) -> str:
    """Return the sensor catalog for a given machine (tags, thresholds, units, fault correlations)."""
    query = text("""
        SELECT sensor_id, tag, sensor_name, unit,
               nominal_value, warn_lo, warn_hi, crit_lo, crit_hi,
               fault_correlation, active
        FROM maintenance.sensor_catalog
        WHERE machine = :machine
        ORDER BY tag
    """)
    with pg_engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"machine": machine})
    if df.empty:
        return f"No sensors found for machine '{machine}'."
    return df.to_markdown(index=False)


@mcp.tool()
def get_sensor_readings(
    machine: str,
    start_date: str,
    end_date: str,
    tag: str | None = None,
) -> str:
    """Return sensor readings for a machine within a time window.

    Args:
        machine: Machine ID (e.g. 'HX-200').
        start_date: Start of window (ISO format, e.g. '2024-12-01').
        end_date: End of window (ISO format, e.g. '2024-12-18').
        tag: Optional sensor tag to filter (e.g. 'TS-101'). If omitted, returns all sensors.
    """
    if tag:
        query = text("""
            SELECT timestamp, tag, sensor_name, value, unit, status, warn_lo, warn_hi
            FROM maintenance.sensor_readings
            WHERE machine = :machine
              AND tag = :tag
              AND timestamp >= :start_date
              AND timestamp <= :end_date
            ORDER BY timestamp
        """)
        params = {
            "machine": machine,
            "tag": tag,
            "start_date": start_date,
            "end_date": end_date,
        }
    else:
        query = text("""
            SELECT timestamp, tag, sensor_name, value, unit, status, warn_lo, warn_hi
            FROM maintenance.sensor_readings
            WHERE machine = :machine
              AND timestamp >= :start_date
              AND timestamp <= :end_date
            ORDER BY tag, timestamp
        """)
        params = {"machine": machine, "start_date": start_date, "end_date": end_date}

    with pg_engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params)
    if df.empty:
        return (
            f"No readings found for machine '{machine}' between {start_date} and {end_date}"
            + (f", tag '{tag}'" if tag else "")
            + "."
        )
    return df.to_markdown(index=False)


@mcp.tool()
def get_remaining_life(machine: str) -> str:
    """Return remaining useful life (RUL) for all components of a machine.

    Shows condition (EXCELLENT/GOOD/MONITOR/END_OF_LIFE/OVERDUE), remaining hours/pct,
    replacement cost, and next inspection date.
    """
    query = text("""
        SELECT component_id, component_name, condition,
               current_hours, remaining_hours, remaining_pct,
               unit_cost_eur, last_inspection, next_inspection, notes
        FROM maintenance.remaining_life
        WHERE machine = :machine
        ORDER BY remaining_pct ASC
    """)
    with pg_engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"machine": machine})
    if df.empty:
        return f"No component life data found for machine '{machine}'."
    return df.to_markdown(index=False)


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8001)
