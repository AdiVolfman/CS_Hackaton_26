from datetime import datetime
import numpy as np
import pandas as pd
import requests


class OpenMeteoFetcher:
    """Fetches ocean current data from Open-Meteo Marine API.

    Returns a DataFrame with columns: time, latitude, longitude, uo (East m/s), vo (North m/s).
    """

    name = "open-meteo"

    def fetch(self, lat: float, lon: float, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "ocean_current_velocity,ocean_current_direction",
            "start_hour": start_time.strftime("%Y-%m-%dT%H:%M"),
            "end_hour": end_time.strftime("%Y-%m-%dT%H:%M"),
            "cell_selection": "sea",
        }
        response = requests.get("https://marine-api.open-meteo.com/v1/marine", params=params, timeout=20).json()

        hourly = response.get("hourly", {})
        times = hourly.get("time", [])
        velocities = hourly.get("ocean_current_velocity", [])  # km/h
        directions = hourly.get("ocean_current_direction", [])  # degrees

        rows = []
        for t, v, d in zip(times, velocities, directions):
            speed_ms = (v * 1000) / 3600
            rad = np.radians(d)
            uo = speed_ms * np.sin(rad)
            vo = speed_ms * np.cos(rad)
            rows.append({"time": pd.to_datetime(t), "latitude": lat, "longitude": lon, "uo": uo, "vo": vo})

        return pd.DataFrame(rows)
