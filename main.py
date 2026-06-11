from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import datetime
import json
import pathlib
import threading
from typing import Optional
import pandas as pd
import numpy as np

from fetch_current import CopernicusFetcher, OpenMeteoFetcher

app = FastAPI(title="SafeCurrent - Search & Rescue API")
MAX_SIMULATION_DAYS = 7
MAX_SIMULATION_HOURS = MAX_SIMULATION_DAYS * 24

# CRITICAL FOR HACKATHON: Allows your Frontend (React/Vue/HTML) to talk to this Backend without CORS blocks
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def print_frontend_url():
    index_path = pathlib.Path(__file__).resolve().parent / "index.html"
    if index_path.exists():
        message = f"Frontend: {index_path.as_uri()}"
    else:
        message = f"Frontend: index.html not found at {index_path}"
    threading.Timer(0.3, lambda: print(message, flush=True)).start()

# --- CORE SIMULATION LOGIC ---

def get_current_data(lat, lon, start_time, end_time):
    """
    Tries to fetch from Copernicus. If it fails, instantly switches to Open-Meteo API
    to ensure the hackathon app never crashes.
    """
    try:
        print("Attempting to fetch from Copernicus Official API...")
        df = CopernicusFetcher().fetch(lat, lon, start_time, end_time)
        return df, "copernicus"
    except Exception as e:
        print(f"Copernicus failed ({e}). Switching to Open-Meteo Marine API Backup...")
        df = OpenMeteoFetcher().fetch(lat, lon, start_time, end_time)
        return df, "open-meteo"


def calculate_next_position(current_lat, current_lon, target_time, df, hour_index, source, step_hours=1.0):
    df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
    target_time = pd.to_datetime(target_time).tz_localize(None)
    
    # Safety Check: If dataframe is empty, stop immediately
    if df.empty:
        return current_lat, current_lon
        
    # Find the closest hour available. If target_time is too new, fallback to the latest available hour
    try:
        closest_time = df['time'].iloc[(df['time'] - target_time).abs().argsort()[:1]].values[0]
        hourly_df = df[df['time'] == closest_time]
    except Exception:
        hourly_df = df.tail(1) # fallback to the last known data row
    
    if hourly_df.empty:
        return current_lat, current_lon
    
    if source == "copernicus":
        distances = np.sqrt((hourly_df['latitude'] - current_lat)**2 + (hourly_df['longitude'] - current_lon)**2)
        closest_row = hourly_df.loc[distances.idxmin()]
    else:
        closest_row = hourly_df.iloc[0]
        
    uo = closest_row['uo']
    vo = closest_row['vo']
    
    # Israeli Rip Current Logic (First hour push Westward)
    rip_push = -0.75 if hour_index == 0 else 0.0
    total_uo = uo + rip_push
    total_vo = vo
    
    meters_per_degree_lat = 111000
    meters_per_degree_lon = 111000 * np.cos(np.radians(current_lat))
    
    elapsed_seconds = step_hours * 3600
    delta_lat = (total_vo * elapsed_seconds) / meters_per_degree_lat
    delta_lon = (total_uo * elapsed_seconds) / meters_per_degree_lon
    
    return current_lat + delta_lat, current_lon + delta_lon

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

def run_trajectory(lat, lon, start_time, elapsed_hours, df, source):
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
            step_hours=step_hours
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
):
    start_time, now_utc, elapsed_hours = resolve_simulation_window(days, hours, minutes, drowning_time)
    end_time = now_utc + datetime.timedelta(hours=2)

    df, source = get_current_data(lat, lon, start_time, end_time)
    trajectory = run_trajectory(lat, lon, start_time, elapsed_hours, df, source)

    area_trajectories = []
    polygon_points = parse_polygon_points(polygon)
    if polygon_points:
        for point_lat, point_lon in polygon_points:
            area_trajectories.append({
                "origin": {"lat": point_lat, "lon": point_lon},
                "trajectory": run_trajectory(point_lat, point_lon, start_time, elapsed_hours, df, source),
            })

    return {
        "status": "success",
        "trajectory": trajectory,
        "area_trajectories": area_trajectories,
        "hours_simulated": round(elapsed_hours, 2),
        "data_source_used": source,
    }