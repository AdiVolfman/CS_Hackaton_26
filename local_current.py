"""Local current — manual override the lifeguard provides on top of the model.

Direction in compass degrees (TO direction the water flows toward),
strength as weak/medium/strong (mapped to fixed speeds in m/s),
duration as first_hour or all (whether to apply only at hour 0 or every step).
"""

import numpy as np
from fastapi import HTTPException


LOCAL_CURRENT_SPEEDS = {
    "weak": 0.2,
    "medium": 0.5,
    "strong": 0.8,
}
LOCAL_CURRENT_DURATIONS = {"first_hour", "all"}


def parse_local_current(direction_deg, strength, duration):
    if direction_deg is None and strength is None and duration is None:
        return None

    if direction_deg is None:
        raise HTTPException(status_code=400, detail="Local current direction is required.")

    try:
        normalized_direction = float(direction_deg) % 360
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Local current direction must be a number.")

    strength_key = (strength or "medium").lower()
    duration_key = (duration or "first_hour").lower()

    if strength_key not in LOCAL_CURRENT_SPEEDS:
        raise HTTPException(status_code=400, detail="Local current strength must be weak, medium, or strong.")

    if duration_key not in LOCAL_CURRENT_DURATIONS:
        raise HTTPException(status_code=400, detail="Local current duration must be first_hour or all.")

    speed_mps = LOCAL_CURRENT_SPEEDS[strength_key]
    direction_radians = np.radians(normalized_direction)

    return {
        "direction_deg": normalized_direction,
        "strength": strength_key,
        "duration": duration_key,
        "speed_mps": speed_mps,
        "uo": float(speed_mps * np.sin(direction_radians)),
        "vo": float(speed_mps * np.cos(direction_radians)),
    }


def get_local_current_vector(local_current, hour_index):
    if not local_current:
        return 0.0, 0.0

    if local_current["duration"] == "first_hour" and hour_index > 0:
        return 0.0, 0.0

    return local_current["uo"], local_current["vo"]
