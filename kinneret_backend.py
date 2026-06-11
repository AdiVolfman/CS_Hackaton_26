# Run with:
# python -m uvicorn kinneret_backend:app --reload --port 8001

import datetime
import io
import math
import os
import pathlib
import threading
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


app = FastAPI(title="SafeCurrent - Kinneret Mode API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_HOURS = 72
MAX_PARTICLES = 5000
DEFAULT_WINDAGE_ALPHA = 0.02
NOISE_STD_MPS = 0.03
SAMPLE_PATH_COUNT = 12
CURRENT_LOOKUP_MAX_AGE_HOURS = 3
DEFAULT_CURRENT_DATA_FILE = pathlib.Path(__file__).resolve().parent / "data" / "kinneret_currents.csv"

# TODO: replace this rough outline with an accurate OpenStreetMap GeoJSON boundary.
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


class Point(BaseModel):
    lat: float
    lon: float


class ManualCurrent(BaseModel):
    enabled: bool = False
    direction_degrees: float = 270.0
    speed_mps: float = 0.0


class KinneretSimulationRequest(BaseModel):
    polygon: List[Point] = Field(..., min_items=3)
    hours: Optional[float] = Field(None, gt=0, le=MAX_HOURS)
    drowning_time: Optional[datetime.datetime] = None
    time_uncertainty_minutes: float = Field(0, ge=0)
    particles: int = Field(1000, ge=1, le=MAX_PARTICLES)
    manual_current: Optional[ManualCurrent] = None


@app.on_event("startup")
def print_frontend_url():
    index_path = pathlib.Path(__file__).resolve().parent / "index_kinneret.html"
    if index_path.exists():
        message = f"Kinneret frontend: {index_path.as_uri()}"
    else:
        message = f"Kinneret frontend: index_kinneret.html not found at {index_path}"
    threading.Timer(0.3, lambda: print(message, flush=True)).start()


def fetch_kinneret_wind(start_time, end_time, lat, lon):
    """Return hourly wind near Lake Kinneret.

    Open-Meteo is used as the first real data source because it is documented
    and works without API keys. IMS support is left as a future direct-station
    integration.
    """
    # TODO: connect to IMS station observations near Lake Kinneret.
    # TODO: select nearest station dynamically.
    try:
        return fetch_open_meteo_forecast_wind(start_time, end_time, lat, lon)
    except Exception as error:
        print(f"Open-Meteo forecast wind failed ({error}). Trying archive API...")

    try:
        return fetch_open_meteo_archive_wind(start_time, end_time, lat, lon)
    except Exception as error:
        print(f"Open-Meteo archive wind failed ({error}). Using fallback wind.")

    return fallback_kinneret_wind(start_time, end_time)


def build_open_meteo_wind_df(payload, start_time, end_time, source):
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
        if time.tzinfo is None:
            time = time.tz_localize("UTC")
        else:
            time = time.tz_convert("UTC")

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


def fetch_open_meteo_forecast_wind(start_time, end_time, lat, lon):
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
    df = build_open_meteo_wind_df(response.json(), start_time, end_time, "open-meteo-forecast")
    print(f"Using Open-Meteo forecast wind: {len(df)} hourly rows.")
    return df


def fetch_open_meteo_archive_wind(start_time, end_time, lat, lon):
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
    df = build_open_meteo_wind_df(response.json(), start_time, end_time, "open-meteo-archive")
    print(f"Using Open-Meteo archive wind: {len(df)} hourly rows.")
    return df


def fallback_kinneret_wind(start_time, end_time):
    print("Using fallback Kinneret wind provider. TODO: connect IMS observations.")
    total_hours = max(1, int(math.ceil((end_time - start_time).total_seconds() / 3600)))
    rows = []

    for hour in range(total_hours + 1):
        time = start_time + datetime.timedelta(hours=hour)
        speed = 5.0 + 1.4 * math.sin(hour / 3.0)
        direction = 270.0 + 18.0 * math.sin(hour / 5.0)
        rows.append({
            "time": time,
            "wind_speed_mps": round(speed, 3),
            "wind_direction_degrees": round(direction, 2),
        })

    df = pd.DataFrame(rows)
    df.attrs["source"] = "fallback"
    return df


def resolve_simulation_window(request: KinneretSimulationRequest):
    now_utc = datetime.datetime.now(datetime.timezone.utc)

    if request.hours is not None and request.drowning_time is not None:
        raise HTTPException(status_code=400, detail="Use either hours or drowning_time, not both.")

    if request.drowning_time is not None:
        if request.drowning_time.tzinfo is None:
            start_time = request.drowning_time.replace(tzinfo=datetime.timezone.utc)
        else:
            start_time = request.drowning_time.astimezone(datetime.timezone.utc)
        elapsed_hours = (now_utc - start_time).total_seconds() / 3600
    elif request.hours is not None:
        elapsed_hours = request.hours
        start_time = now_utc - datetime.timedelta(hours=elapsed_hours)
    else:
        raise HTTPException(status_code=400, detail="Provide hours or drowning_time.")

    if elapsed_hours <= 0:
        raise HTTPException(status_code=400, detail="The drowning time must be in the past.")

    if elapsed_hours > MAX_HOURS:
        raise HTTPException(status_code=400, detail=f"Kinneret simulation is limited to {MAX_HOURS} hours.")

    return start_time, now_utc, elapsed_hours


def wind_to_uv(speed_mps, direction_degrees):
    """Convert meteorological FROM direction into movement TO vector in m/s."""
    to_degrees = (direction_degrees + 180.0) % 360.0
    radians = math.radians(to_degrees)
    u = speed_mps * math.sin(radians)
    v = speed_mps * math.cos(radians)
    return u, v


def direction_to_uv(speed_mps, direction_degrees):
    """Convert a current direction that already points TO travel direction."""
    radians = math.radians(direction_degrees % 360.0)
    u = speed_mps * math.sin(radians)
    v = speed_mps * math.cos(radians)
    return u, v


def validate_lat_lon(lat, lon):
    if not math.isfinite(lat) or not math.isfinite(lon):
        raise HTTPException(status_code=400, detail="Coordinates must be finite numbers.")
    if lat < -90 or lat > 90 or lon < -180 or lon > 180:
        raise HTTPException(status_code=400, detail="Coordinates are outside valid lat/lon ranges.")


def point_on_segment(point, start, end, epsilon=1e-10):
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
        if point_on_segment(point, previous, current):
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


def polygon_bounds(polygon):
    lats = [point[0] for point in polygon]
    lons = [point[1] for point in polygon]
    return min(lats), max(lats), min(lons), max(lons)


def parse_polygon(points: List[Point]):
    polygon = []
    for point in points:
        validate_lat_lon(point.lat, point.lon)
        polygon.append((float(point.lat), float(point.lon)))

    if len(polygon) < 3:
        raise HTTPException(status_code=400, detail="Polygon must contain at least 3 points.")

    return polygon


def sample_start_points(polygon, count, rng):
    min_lat, max_lat, min_lon, max_lon = polygon_bounds(polygon)
    samples = []
    attempts = 0
    max_attempts = max(2000, count * 250)

    while len(samples) < count and attempts < max_attempts:
        attempts += 1
        lat = rng.uniform(min_lat, max_lat)
        lon = rng.uniform(min_lon, max_lon)

        if point_in_polygon((lat, lon), polygon) and point_in_polygon((lat, lon), KINNERET_BOUNDARY):
            samples.append((lat, lon))

    if len(samples) < count:
        print("Start area has limited overlap with the rough lake boundary. Sampling remaining points from the requested area.")

    attempts = 0
    while len(samples) < count and attempts < max_attempts:
        attempts += 1
        lat = rng.uniform(min_lat, max_lat)
        lon = rng.uniform(min_lon, max_lon)

        if point_in_polygon((lat, lon), polygon):
            samples.append((lat, lon))

    if len(samples) < count:
        raise HTTPException(status_code=400, detail="Could not sample enough start points inside the polygon.")

    return samples


def get_wind_record(wind_records, time):
    start_time = wind_records[0][0]
    index = int(round((time - start_time).total_seconds() / 3600))
    index = min(max(index, 0), len(wind_records) - 1)
    return wind_records[index]


def advance_position(lat, lon, u_mps, v_mps, step_hours):
    meters_per_degree_lat = 111000.0
    meters_per_degree_lon = 111000.0 * math.cos(math.radians(lat))
    meters_per_degree_lon = max(abs(meters_per_degree_lon), 1.0)
    elapsed_seconds = step_hours * 3600.0
    next_lat = lat + (v_mps * elapsed_seconds) / meters_per_degree_lat
    next_lon = lon + (u_mps * elapsed_seconds) / meters_per_degree_lon
    return next_lat, next_lon


def build_wind_records(wind_df):
    rows = []
    for row in wind_df.itertuples(index=False):
        rows.append((
            pd.Timestamp(row.time).to_pydatetime(),
            float(row.wind_speed_mps),
            float(row.wind_direction_degrees),
        ))
    return rows


def find_column(df, candidates):
    normalized = {str(column).strip().lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def normalize_current_dataframe(df, source, start_time, end_time):
    if df.empty:
        return pd.DataFrame()

    time_col = find_column(df, ["time", "timestamp", "datetime", "date_time"])
    lat_col = find_column(df, ["lat", "latitude"])
    lon_col = find_column(df, ["lon", "lng", "longitude"])
    u_col = find_column(df, ["u_mps", "u", "east_mps", "current_u", "current_u_mps"])
    v_col = find_column(df, ["v_mps", "v", "north_mps", "current_v", "current_v_mps"])
    speed_col = find_column(df, ["speed_mps", "current_speed_mps", "speed", "current_speed"])
    direction_col = find_column(df, ["direction_degrees", "current_direction_degrees", "direction", "current_direction"])

    if time_col is None or lat_col is None or lon_col is None:
        raise ValueError("Current data must include time, lat, and lon columns.")

    if (u_col is None or v_col is None) and (speed_col is None or direction_col is None):
        raise ValueError("Current data must include u_mps/v_mps or speed_mps/direction_degrees columns.")

    rows = []
    start_ts = pd.Timestamp(start_time).tz_convert("UTC")
    end_ts = pd.Timestamp(end_time).tz_convert("UTC")

    for row in df.to_dict("records"):
        time = pd.to_datetime(row[time_col], utc=True, errors="coerce")
        if pd.isna(time):
            continue

        if not (start_ts - pd.Timedelta(hours=1) <= time <= end_ts + pd.Timedelta(hours=1)):
            continue

        try:
            lat = float(row[lat_col])
            lon = float(row[lon_col])
            if u_col is not None and v_col is not None:
                u = float(row[u_col])
                v = float(row[v_col])
            else:
                u, v = direction_to_uv(float(row[speed_col]), float(row[direction_col]))
        except (TypeError, ValueError):
            continue

        rows.append({
            "time": time.to_pydatetime(),
            "latitude": lat,
            "longitude": lon,
            "u_mps": u,
            "v_mps": v,
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("time").reset_index(drop=True)
        result.attrs["source"] = source
    return result


def read_current_dataframe_from_text(text):
    stripped = text.lstrip()
    if stripped.startswith("[") or stripped.startswith("{"):
        return pd.read_json(io.StringIO(text))
    return pd.read_csv(io.StringIO(text))


def load_kinneret_current_data(start_time, end_time):
    """Load measured/model lake current data when supplied by URL or local file.

    Expected columns:
    time, lat, lon, u_mps, v_mps

    Alternative current columns:
    time, lat, lon, speed_mps, direction_degrees

    Current direction is interpreted as direction TO which water flows.
    """
    source_url = os.getenv("KINNERET_CURRENT_DATA_URL")
    if source_url:
        try:
            response = requests.get(source_url, timeout=10)
            response.raise_for_status()
            df = read_current_dataframe_from_text(response.text)
            normalized = normalize_current_dataframe(df, "configured-current-url", start_time, end_time)
            if not normalized.empty:
                print(f"Using configured Kinneret current URL: {len(normalized)} rows.")
                return normalized
        except Exception as error:
            print(f"Configured current URL failed ({error}). Trying local current file...")

    configured_path = os.getenv("KINNERET_CURRENT_DATA_FILE")
    current_path = pathlib.Path(configured_path) if configured_path else DEFAULT_CURRENT_DATA_FILE
    if current_path.exists():
        try:
            df = pd.read_csv(current_path)
            normalized = normalize_current_dataframe(df, f"file:{current_path.name}", start_time, end_time)
            if not normalized.empty:
                print(f"Using Kinneret current file {current_path}: {len(normalized)} rows.")
                return normalized
            print(f"Kinneret current file {current_path} had no rows for the requested window.")
        except Exception as error:
            print(f"Kinneret current file failed ({error}). Continuing without measured current data.")

    empty = pd.DataFrame(columns=["time", "latitude", "longitude", "u_mps", "v_mps"])
    empty.attrs["source"] = "none"
    return empty


def build_current_records(current_df):
    if current_df.empty:
        return []

    records = []
    for row in current_df.itertuples(index=False):
        records.append((
            pd.Timestamp(row.time).to_pydatetime(),
            float(row.latitude),
            float(row.longitude),
            float(row.u_mps),
            float(row.v_mps),
        ))
    return records


def get_current_uv(current_records, lat, lon, time):
    if not current_records:
        return 0.0, 0.0, False

    best = None
    best_score = None
    for record_time, record_lat, record_lon, u, v in current_records:
        time_diff_hours = abs((record_time - time).total_seconds()) / 3600
        if time_diff_hours > CURRENT_LOOKUP_MAX_AGE_HOURS:
            continue

        spatial_score = (record_lat - lat) ** 2 + (record_lon - lon) ** 2
        score = time_diff_hours * 100 + spatial_score
        if best_score is None or score < best_score:
            best_score = score
            best = (u, v)

    if best is None:
        return 0.0, 0.0, False

    return best[0], best[1], True


def simulate_particles(request, elapsed_hours, start_points, wind_df, current_df, now_utc, rng):
    uncertainty_hours = request.time_uncertainty_minutes / 60.0

    if uncertainty_hours >= elapsed_hours:
        raise HTTPException(
            status_code=400,
            detail="Time uncertainty must be smaller than the requested hours.",
        )

    low_hours = elapsed_hours - uncertainty_hours
    high_hours = elapsed_hours + uncertainty_hours
    effective_hours = (
        rng.uniform(low_hours, high_hours, request.particles)
        if uncertainty_hours > 0
        else np.full(request.particles, elapsed_hours)
    )

    manual_u = 0.0
    manual_v = 0.0
    manual = request.manual_current
    if manual and manual.enabled:
        if manual.speed_mps < 0 or manual.speed_mps > 2:
            raise HTTPException(status_code=400, detail="Manual current speed must be between 0 and 2 m/s.")
        manual_u, manual_v = direction_to_uv(manual.speed_mps, manual.direction_degrees)

    wind_records = build_wind_records(wind_df)
    current_records = build_current_records(current_df)
    final_particles = []
    sample_paths = []
    likely_path_points = {}

    for index, (start_lat, start_lon) in enumerate(start_points):
        lat = start_lat
        lon = start_lon
        was_inside_lake = point_in_polygon((lat, lon), KINNERET_BOUNDARY)
        beached = False
        remaining_hours = float(effective_hours[index])
        particle_start_time = now_utc - datetime.timedelta(hours=remaining_hours)
        elapsed_so_far = 0.0
        path = [{"hour": 0, "lat": round(lat, 6), "lon": round(lon, 6), "beached": beached}] if index < SAMPLE_PATH_COUNT else None
        likely_path_points.setdefault(0.0, []).append((lat, lon))

        while remaining_hours > 1e-9 and not beached:
            step_hours = min(1.0, remaining_hours)
            sample_time = particle_start_time + datetime.timedelta(hours=elapsed_so_far)
            _, wind_speed, wind_direction = get_wind_record(wind_records, sample_time)
            wind_u, wind_v = wind_to_uv(wind_speed, wind_direction)
            current_u, current_v, _ = get_current_uv(current_records, lat, lon, sample_time)
            u = current_u + DEFAULT_WINDAGE_ALPHA * wind_u + manual_u + rng.normal(0.0, NOISE_STD_MPS)
            v = current_v + DEFAULT_WINDAGE_ALPHA * wind_v + manual_v + rng.normal(0.0, NOISE_STD_MPS)
            next_lat, next_lon = advance_position(lat, lon, u, v, step_hours)

            if point_in_polygon((next_lat, next_lon), KINNERET_BOUNDARY):
                lat, lon = next_lat, next_lon
                was_inside_lake = True
            elif not was_inside_lake:
                lat, lon = next_lat, next_lon
            else:
                beached = True

            elapsed_so_far += step_hours
            remaining_hours = max(0.0, remaining_hours - step_hours)

            if path is not None:
                path.append({
                    "hour": round(elapsed_so_far, 2),
                    "lat": round(lat, 6),
                    "lon": round(lon, 6),
                    "beached": beached,
                })

            likely_path_points.setdefault(round(elapsed_so_far, 2), []).append((lat, lon))

        final_particles.append({
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "beached": beached,
        })

        if path is not None:
            sample_paths.append(path)

    likely_path = []
    for hour in sorted(likely_path_points):
        points = likely_path_points[hour]
        if not points:
            continue

        likely_path.append({
            "hour": hour,
            "lat": round(float(np.mean([point[0] for point in points])), 6),
            "lon": round(float(np.mean([point[1] for point in points])), 6),
        })

    return final_particles, sample_paths, likely_path


def response_center(final_particles):
    if not final_particles:
        return {"lat": 32.82, "lon": 35.59}

    return {
        "lat": round(float(np.mean([point["lat"] for point in final_particles])), 6),
        "lon": round(float(np.mean([point["lon"] for point in final_particles])), 6),
    }


@app.post("/simulate_kinneret")
def simulate_kinneret(request: KinneretSimulationRequest):
    polygon = parse_polygon(request.polygon)
    rng = np.random.default_rng()
    uncertainty_hours = request.time_uncertainty_minutes / 60.0
    start_time, now_utc, elapsed_hours = resolve_simulation_window(request)
    wind_start_time = start_time - datetime.timedelta(hours=uncertainty_hours)

    center_lat = float(np.mean([point[0] for point in polygon]))
    center_lon = float(np.mean([point[1] for point in polygon]))
    start_points = sample_start_points(polygon, request.particles, rng)
    wind_df = fetch_kinneret_wind(wind_start_time, now_utc, center_lat, center_lon)
    current_df = load_kinneret_current_data(wind_start_time, now_utc)
    final_particles, sample_paths, likely_path = simulate_particles(request, elapsed_hours, start_points, wind_df, current_df, now_utc, rng)

    return {
        "status": "success",
        "mode": "kinneret",
        "hours_requested": round(elapsed_hours, 2),
        "time_uncertainty_minutes": request.time_uncertainty_minutes,
        "particles_count": request.particles,
        "wind_source": wind_df.attrs.get("source", "fallback"),
        "current_source": current_df.attrs.get("source", "none"),
        "current_records_count": int(len(current_df)),
        "wind_window_start": wind_start_time.isoformat(),
        "wind_window_end": now_utc.isoformat(),
        "final_particles": final_particles,
        "sample_paths": sample_paths,
        "likely_path": likely_path,
        "center": response_center(final_particles),
    }
