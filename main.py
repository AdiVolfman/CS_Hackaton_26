from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import datetime
import json
import pathlib
import threading
from typing import Optional
import pandas as pd
import numpy as np

from fetch_current import CopernicusFetcher, KinneretFetcher, is_kinneret
from simulation_window import resolve_simulation_window
from share_store import router as share_router
from local_current import parse_local_current, get_local_current_vector

app = FastAPI(title="SafeCurrent - Search & Rescue API")
app.include_router(share_router)

# CRITICAL FOR HACKATHON: Allows your Frontend (React/Vue/HTML) to talk to this Backend without CORS blocks
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
@app.get("/index.html")
def serve_frontend():
    index_path = pathlib.Path(__file__).resolve().parent / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)


@app.on_event("startup")
def print_frontend_url():
    index_path = pathlib.Path(__file__).resolve().parent / "index.html"
    if index_path.exists():
        message = f"Frontend: {index_path.as_uri()}"
    else:
        message = f"Frontend: index.html not found at {index_path}"
    threading.Timer(0.3, lambda: print(message, flush=True)).start()

# --- CORE SIMULATION LOGIC ---

def get_current_data(lat, lon, start_time, end_time, floating):
    if is_kinneret(lat, lon):
        df = KinneretFetcher().fetch(lat, lon, start_time, end_time, floating=floating)
        return df, "kinneret-wind-2d" if floating else "kinneret-wind-3d"
    df = CopernicusFetcher().fetch(lat, lon, start_time, end_time, floating=floating)
    return df, "copernicus-2d" if floating else "copernicus-3d"


def calculate_next_position(current_lat, current_lon, target_time, df, hour_index, source, local_current=None, step_hours=1.0):
    df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
    target_time = pd.to_datetime(target_time).tz_localize(None)

    if df.empty:
        return current_lat, current_lon

    try:
        closest_time = df['time'].iloc[(df['time'] - target_time).abs().argsort()[:1]].values[0]
        hourly_df = df[df['time'] == closest_time]
    except Exception:
        hourly_df = df.tail(1)

    valid = hourly_df.dropna(subset=["uo", "vo"])
    if valid.empty:
        return current_lat, current_lon

    distances = np.sqrt(
        (valid['latitude'] - current_lat) ** 2
        + (valid['longitude'] - current_lon) ** 2
    )
    closest_row = valid.loc[distances.idxmin()]

    uo = closest_row['uo']
    vo = closest_row['vo']

    if pd.isna(uo) or pd.isna(vo):
        return current_lat, current_lon

    # Israeli rip current — sea only, first hour only, skipped when lifeguard provided a local override
    rip_push = -0.75 if hour_index == 0 and source.startswith("copernicus") and local_current is None else 0.0
    local_uo, local_vo = get_local_current_vector(local_current, hour_index)
    total_uo = uo + rip_push + local_uo
    total_vo = vo + local_vo

    meters_per_degree_lat = 111000
    meters_per_degree_lon = 111000 * np.cos(np.radians(current_lat))

    elapsed_seconds = step_hours * 3600
    delta_lat = (total_vo * elapsed_seconds) / meters_per_degree_lat
    delta_lon = (total_uo * elapsed_seconds) / meters_per_degree_lon

    return current_lat + delta_lat, current_lon + delta_lon

def parse_polygon_points(polygon):
    if polygon is None:
        return None

    try:
        raw_points = json.loads(polygon)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Polygon must be valid JSON.")

    if not isinstance(raw_points, list) or len(raw_points) < 3:
        raise HTTPException(status_code=400, detail="Polygon must contain at least 3 points.")

    points = []
    for point in raw_points:
        if isinstance(point, dict):
            raw_lat = point.get("lat")
            raw_lon = point.get("lon")
        elif isinstance(point, list) and len(point) >= 2:
            raw_lat, raw_lon = point[0], point[1]
        else:
            raise HTTPException(status_code=400, detail="Each polygon point must include lat and lon.")

        try:
            point_lat = float(raw_lat)
            point_lon = float(raw_lon)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="Polygon coordinates must be numbers.")

        if not (-90 <= point_lat <= 90 and -180 <= point_lon <= 180):
            raise HTTPException(status_code=400, detail="Polygon coordinates are outside valid lat/lon ranges.")

        points.append((point_lat, point_lon))

    return points

def run_trajectory(lat, lon, start_time, elapsed_hours, df, source, local_current=None):
    trajectory = [{"hour": 0, "lat": lat, "lon": lon}]
    current_lat = lat
    current_lon = lon
    current_time = pd.to_datetime(start_time)
    remaining_hours = elapsed_hours
    elapsed_so_far = 0.0
    hour_index = 0

    while remaining_hours > 1e-9:
        step_hours = min(1.0, remaining_hours)
        current_time += pd.to_timedelta(step_hours, unit='h')
        next_lat, next_lon = calculate_next_position(
            current_lat,
            current_lon,
            current_time,
            df,
            hour_index=hour_index,
            source=source,
            local_current=local_current,
            step_hours=step_hours,
        )
        elapsed_so_far += step_hours

        trajectory.append({
            "hour": round(elapsed_so_far, 2),
            "lat": round(float(next_lat), 5),
            "lon": round(float(next_lon), 5)
        })
        current_lat, current_lon = next_lat, next_lon
        remaining_hours = max(0.0, remaining_hours - step_hours)
        hour_index += 1

    return trajectory


@app.get("/simulate")
def simulate_drift(
    lat: float = Query(..., description="Latitude of last seen position"),
    lon: float = Query(..., description="Longitude of last seen position"),
    days: Optional[float] = Query(None),
    hours: Optional[float] = Query(None),
    minutes: Optional[float] = Query(None),
    drowning_time: Optional[datetime.datetime] = Query(None),
    polygon: Optional[str] = Query(None),
    floating: bool = Query(False, description="True = use surface (2D); False = use seabed currents (3D)"),
    local_current_direction_deg: Optional[float] = Query(None),
    local_current_strength: Optional[str] = Query(None),
    local_current_duration: Optional[str] = Query(None),
):
    start_time, now_utc, elapsed_hours = resolve_simulation_window(days, hours, minutes, drowning_time)
    end_time = now_utc + datetime.timedelta(hours=2)
    local_current = parse_local_current(
        local_current_direction_deg,
        local_current_strength,
        local_current_duration
    )

    df, source = get_current_data(lat, lon, start_time, end_time, floating=floating)
    trajectory = run_trajectory(lat, lon, start_time, elapsed_hours, df, source, local_current)

    area_trajectories = []
    polygon_points = parse_polygon_points(polygon)
    if polygon_points:
        for point_lat, point_lon in polygon_points:
            area_trajectories.append({
                "origin": {"lat": point_lat, "lon": point_lon},
                "trajectory": run_trajectory(point_lat, point_lon, start_time, elapsed_hours, df, source, local_current),
            })

    return {
        "status": "success",
        "trajectory": trajectory,
        "area_trajectories": area_trajectories,
        "hours_simulated": round(elapsed_hours, 2),
        "data_source_used": source,
        "local_current_used": local_current,
    }
