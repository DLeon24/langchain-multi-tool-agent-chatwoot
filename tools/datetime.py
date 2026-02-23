"""
Tool: Current Date & Time (by region)
Returns the current date and time in a timezone. No external APIs; production-friendly.
"""

import os
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import find_dotenv, load_dotenv
from langchain_core.tools import tool

load_dotenv(find_dotenv())

# Default timezone (e.g. America/Lima, America/Mexico_City, Europe/Madrid)
DEFAULT_TIMEZONE = os.getenv("AGENT_TIMEZONE", "America/Lima")

# Day and month names in English
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]


@tool
def get_current_datetime(timezone: str = "") -> str:
    """
    Gets the current date and time in a timezone.

    Use this tool when the user asks:
    - What time/date it is (now)
    - The current moment/schedule
    - The time in another city/region (pass the IANA timezone, e.g. Europe/Madrid, America/Mexico_City)

    Args:
        timezone: Optional IANA timezone (e.g. America/Lima, Europe/Madrid).
                  If empty, defaults to the agent timezone (AGENT_TIMEZONE).
    """
    resolved_tz = (timezone or "").strip() or DEFAULT_TIMEZONE
    print(f"   🕐 Getting date/time: {resolved_tz}")
    return _get_current_datetime(resolved_tz)


def _get_current_datetime(timezone: str) -> str:
    """Gets the current date/time in the given timezone (stdlib only: zoneinfo)."""
    try:
        tz = ZoneInfo(timezone)
    except Exception:
        tz = ZoneInfo(DEFAULT_TIMEZONE)
        timezone = DEFAULT_TIMEZONE
    now = datetime.now(tz)
    day_name = DAYS[now.weekday()]
    month_name = MONTHS[now.month - 1]
    readable_date = f"{day_name}, {month_name} {now.day}, {now.year}"
    time_str = now.strftime("%H:%M:%S")
    return (
        f"Timezone: {timezone}\n"
        f"Date: {readable_date}\n"
        f"Time: {time_str}\n"
        f"ISO: {now.strftime('%Y-%m-%dT%H:%M:%S%z')}"
    )
