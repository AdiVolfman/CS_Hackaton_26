"""Time-window parsing for /simulate.

Resolves the two ways a client can specify when the drift started
(elapsed time vs absolute drowning_time) into a single
(start_time, now_utc, elapsed_hours) tuple.
"""

import datetime

from fastapi import HTTPException


MAX_SIMULATION_DAYS = 7
MAX_SIMULATION_HOURS = MAX_SIMULATION_DAYS * 24


def resolve_simulation_window(days, hours, minutes, drowning_time):
    now_utc = datetime.datetime.now(datetime.timezone.utc)

    if drowning_time is not None and any(value is not None for value in (days, hours, minutes)):
        raise HTTPException(
            status_code=400,
            detail="Use either elapsed time or drowning_time, not both."
        )

    if drowning_time is not None:
        if drowning_time.tzinfo is None:
            start_time = drowning_time.replace(tzinfo=datetime.timezone.utc)
        else:
            start_time = drowning_time.astimezone(datetime.timezone.utc)
        elapsed_hours = (now_utc - start_time).total_seconds() / 3600
    else:
        if days is None and hours is None and minutes is None:
            elapsed_hours = 5
        else:
            elapsed_hours = (
                (0 if days is None else days * 24)
                + (0 if hours is None else hours)
                + (0 if minutes is None else minutes / 60)
            )
        start_time = now_utc - datetime.timedelta(hours=elapsed_hours)

    if elapsed_hours <= 0:
        raise HTTPException(status_code=400, detail="The drowning time must be in the past.")

    if elapsed_hours > MAX_SIMULATION_HOURS:
        raise HTTPException(status_code=400, detail=f"Simulation is limited to {MAX_SIMULATION_DAYS} days.")

    return start_time, now_utc, elapsed_hours
