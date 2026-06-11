"""Kinneret (Lake) drift source — wind-driven uo/vo dataframe.

Wind comes from Open-Meteo (forecast → archive).
The fetcher returns a Copernicus-shaped dataframe so the trajectory
algorithm in main.py works unchanged.
"""

import datetime
import math
from typing import List, Sequence, Tuple

import pandas as pd
import requests


DEFAULT_WINDAGE_ALPHA = 0.02
DEEP_REVERSE_FACTOR = -0.2  # sinking body: damped, reversed surface drift
MAX_HOURS = 72


# Rough Kinneret boundary polygon (lat, lon).
# TODO: replace with an accurate OpenStreetMap GeoJSON boundary.
KINNERET_BOUNDARY: List[Tuple[float, float]] = [
    (32.918, 35.560),
    (32.908, 35.625),
    (32.885, 35.700),
    (32.830, 35.705),
    (32.780, 35.690),
    (32.735, 35.642),
    (32.705, 35.604),
    (32.695, 35.570),
    (32.712, 35.530),
    (32.755, 35.510),
    (32.815, 35.505),
    (32.875, 35.512),
]


def _point_on_segment(point, start, end, epsilon=1e-10):
    py, px = point
    ay, ax = start
    by, bx = end
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > epsilon:
        return False
    return (
        min(ax, bx) - epsilon <= px <= max(ax, bx) + epsilon
        and min(ay, by) - epsilon <= py <= max(ay, by) + epsilon
    )


def point_in_polygon(point, polygon: Sequence[Tuple[float, float]]):
    lat, lon = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if _point_on_segment(point, previous, current):
            return True
        lat_i, lon_i = current
        lat_j, lon_j = previous
        intersects = (
            (lat_i > lat) != (lat_j > lat)
            and lon < (lon_j - lon_i) * (lat - lat_i) / ((lat_j - lat_i) or 1e-12) + lon_i
        )
        if intersects:
            inside = not inside
        previous = current
    return inside


def is_kinneret(lat, lon):
    return point_in_polygon((lat, lon), KINNERET_BOUNDARY)


def wind_to_uv(speed_mps, direction_degrees):
    """Convert meteorological FROM direction into movement TO vector in m/s."""
    to_degrees = (direction_degrees + 180.0) % 360.0
    radians = math.radians(to_degrees)
    u = speed_mps * math.sin(radians)
    v = speed_mps * math.cos(radians)
    return u, v


def _build_open_meteo_wind_df(payload, start_time, end_time, source):
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    speeds = hourly.get("wind_speed_10m", [])
    directions = hourly.get("wind_direction_10m", [])
    rows = []
    start_ts = pd.Timestamp(start_time).tz_convert("UTC")
    end_ts = pd.Timestamp(end_time).tz_convert("UTC")

    for raw_time, speed, direction in zip(times, speeds, directions):
        if speed is None or direction is None:
            continue
        time = pd.Timestamp(raw_time)
        time = time.tz_localize("UTC") if time.tzinfo is None else time.tz_convert("UTC")
        if start_ts - pd.Timedelta(hours=1) <= time <= end_ts + pd.Timedelta(hours=1):
            rows.append({
                "time": time.to_pydatetime(),
                "wind_speed_mps": float(speed),
                "wind_direction_degrees": float(direction),
            })

    if not rows:
        raise RuntimeError("No hourly wind rows returned for requested time window.")

    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    df.attrs["source"] = source
    return df


def _fetch_open_meteo_forecast_wind(start_time, end_time, lat, lon):
    past_hours = min(MAX_HOURS + 6, max(1, int(math.ceil((end_time - start_time).total_seconds() / 3600)) + 3))
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "wind_speed_10m,wind_direction_10m",
        "wind_speed_unit": "ms",
        "timezone": "UTC",
        "past_hours": past_hours,
        "forecast_hours": 1,
        "cell_selection": "nearest",
    }
    response = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=10)
    response.raise_for_status()
    return _build_open_meteo_wind_df(response.json(), start_time, end_time, "open-meteo-forecast")


def _fetch_open_meteo_archive_wind(start_time, end_time, lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_time.date().isoformat(),
        "end_date": end_time.date().isoformat(),
        "hourly": "wind_speed_10m,wind_direction_10m",
        "wind_speed_unit": "ms",
        "timezone": "UTC",
        "cell_selection": "nearest",
    }
    response = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=10)
    response.raise_for_status()
    return _build_open_meteo_wind_df(response.json(), start_time, end_time, "open-meteo-archive")


def fetch_kinneret_wind(start_time, end_time, lat, lon):
    try:
        return _fetch_open_meteo_forecast_wind(start_time, end_time, lat, lon)
    except Exception as error:
        print(f"Open-Meteo forecast wind failed ({error}). Trying archive API...")
    return _fetch_open_meteo_archive_wind(start_time, end_time, lat, lon)


class KinneretFetcher:
    """Build a Copernicus-shaped uo/vo dataframe from Open-Meteo wind on Lake Kinneret.

    floating=True  → surface drift: uo/vo = 0.02 * wind vector.
    floating=False → sunken body: damped, reversed deep flow ≈ -0.2 * surface.
    """

    def fetch(self, lat, lon, start_time, end_time, floating=True):
        wind_df = fetch_kinneret_wind(start_time, end_time, lat, lon)
        scale = DEFAULT_WINDAGE_ALPHA if floating else DEFAULT_WINDAGE_ALPHA * DEEP_REVERSE_FACTOR
        rows = []
        for row in wind_df.itertuples(index=False):
            wu, wv = wind_to_uv(float(row.wind_speed_mps), float(row.wind_direction_degrees))
            rows.append({
                "time": pd.Timestamp(row.time),
                "latitude": lat,
                "longitude": lon,
                "uo": scale * wu,
                "vo": scale * wv,
            })
        return pd.DataFrame(rows, columns=["time", "latitude", "longitude", "uo", "vo"])
