import os
import tempfile
from pathlib import Path

GENERATION_MODEL = "gpt-5.4-mini"

CONFIRM_SIGNAL = "CONFIRM_SIGNAL"
TEMPLATE_SAVED_SIGNAL = "TEMPLATE_SAVED_SIGNAL"

CASES_DIR = Path("data/cases")
TEMPLATES_DIR = Path("data/known_case_templates")

POSTGRES_URL = os.getenv(
    "PG_URL",
    "postgresql://langgraph_user:langgraph_password@postgres:5432/langgraph_db",
)
WORKSPACE_BASE_DIR = Path(os.getenv("WORKSPACE_BASE_DIR", tempfile.gettempdir()))

MACHINE_DOMAIN = {
    "HX": "Heat Exchanger. Failures: fin fouling, seal wear, pump cavitation, oil contamination. Sensors: TEMP_OIL, TEMP_COOLANT, PRESSURE_OIL. MTBF ~18 mo.",
    "CNC": "CNC Machining Center. Failures: spindle bearing wear, coolant contamination, axis servo fault. Sensors: VIBRATION, TEMP_SPINDLE.",
    "IH": "Induction Heater. Failures: coil cooling flow fault, insulation breakdown, capacitor degradation. Sensors: TEMP_COIL, COOLANT_FLOW.",
    "CB": "Conveyor Belt. Failures: belt tension loss, bearing wear, motor overload, guide rail misalignment. Sensors: MOTOR_CURRENT, BELT_SPEED.",
    "CM": "Compressor. Failures: valve wear, intercooler fouling, lube failure, suction filter blockage. Sensors: PRESSURE_DISCHARGE, TEMP_DISCHARGE.",
}

DOMAIN_KNOWLEDGE_MD = """\
# Maintenance Domain Knowledge

## Glossary
- PM — Preventive Maintenance
- CM — Corrective Maintenance: unplanned repair after failure
- RCA — Root Cause Analysis
- MTBF — Mean Time Between Failures
- WO — Work Order
- TAG — Sensor identifier (e.g. TEMP_OIL, VIBRATION_X)

## Machine Families

### HX — Heat Exchanger
Common failures: fin fouling, seal wear, pump cavitation, oil contamination.
Key sensors: TEMP_OIL, TEMP_COOLANT, PRESSURE_OIL. MTBF ~18 months.

### CNC — CNC Machining Center
Common failures: spindle bearing wear, coolant contamination, axis servo fault.
Key sensors: VIBRATION_X/Y/Z, TEMP_SPINDLE, CURRENT_SPINDLE.

### IH — Induction Heater
Common failures: coil cooling flow fault, insulation breakdown, capacitor degradation.
Key sensors: TEMP_COIL, COOLANT_FLOW, CURRENT_COIL.

### CB — Conveyor Belt
Common failures: belt tension loss, idler bearing wear, motor overload.
Key sensors: MOTOR_CURRENT, BELT_SPEED, TEMP_MOTOR.

### CM — Compressor
Common failures: valve wear, intercooler fouling, lube failure, suction filter blockage.
Key sensors: PRESSURE_DISCHARGE, TEMP_DISCHARGE, OIL_PRESSURE.
"""


def get_domain_hint(machine_id: str) -> str:
    prefix = machine_id.split("-")[0].upper()
    hint = MACHINE_DOMAIN.get(prefix)
    return f"\n[Domain context — {prefix} family: {hint}]" if hint else ""
