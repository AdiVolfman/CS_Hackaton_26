from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import datetime
import json
import pathlib
import random
import threading
from typing import Literal, Optional
import pandas as pd
import numpy as np
from opendrift.models.leeway import Leeway
from opendrift.readers import reader_constant

from fetch_current import CopernicusFetcher, OpenMeteoFetcher

app = FastAPI(title="SafeCurrent - Search & Rescue API")
MAX_SIMULATION_DAYS = 7
MAX_SIMULATION_HOURS = MAX_SIMULATION_DAYS * 24
MAX_MONTE_CARLO_SAMPLES = 2000

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

class LatLonPoint(BaseModel):
    lat: float
    lon: float


class PersonProfile(BaseModel):
    age: Optional[int] = None
    weight_kg: Optional[float] = None
    swimming_ability: Literal["unknown", "non_swimmer", "average", "strong"] = "unknown"


class AreaSimulationRequest(BaseModel):
    polygon: list[LatLonPoint] = Field(..., min_length=3)
    earliest_time: datetime.datetime
    latest_time: datetime.datetime
    samples: int = Field(1000, ge=1, le=MAX_MONTE_CARLO_SAMPLES)
    person: PersonProfile = Field(default_factory=PersonProfile)
    object_type: int = 27
    current_source: Literal["copernicus", "open-meteo", "auto"] = "copernicus"

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


def get_current_data_for_area(lat, lon, start_time, end_time, current_source):
    if current_source == "open-meteo":
        return OpenMeteoFetcher().fetch(lat, lon, start_time, end_time), "open-meteo"

    if current_source == "copernicus":
        return CopernicusFetcher().fetch(lat, lon, start_time, end_time), "copernicus"

    return get_current_data(lat, lon, start_time, end_time)


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


def validate_polygon_points(points):
    if len(points) < 3:
        raise HTTPException(status_code=400, detail="Polygon must contain at least 3 points.")

    for point_lat, point_lon in points:
        if not (-90 <= point_lat <= 90 and -180 <= point_lon <= 180):
            raise HTTPException(status_code=400, detail="Polygon coordinates are outside valid lat/lon ranges.")


def point_in_polygon(lat, lon, polygon_points):
    inside = False
    j = len(polygon_points) - 1

    for i, (lat_i, lon_i) in enumerate(polygon_points):
        lat_j, lon_j = polygon_points[j]
        intersects = ((lon_i > lon) != (lon_j > lon)) and (
            lat < (lat_j - lat_i) * (lon - lon_i) / ((lon_j - lon_i) or 1e-12) + lat_i
        )
        if intersects:
            inside = not inside
        j = i

    return inside


def sample_point_in_polygon(polygon_points):
    min_lat = min(point[0] for point in polygon_points)
    max_lat = max(point[0] for point in polygon_points)
    min_lon = min(point[1] for point in polygon_points)
    max_lon = max(point[1] for point in polygon_points)

    for _ in range(10000):
        lat = random.uniform(min_lat, max_lat)
        lon = random.uniform(min_lon, max_lon)
        if point_in_polygon(lat, lon, polygon_points):
            return lat, lon

    raise HTTPException(status_code=400, detail="Could not sample points inside polygon.")


def normalize_utc(value):
    if value.tzinfo is None:
        return value.replace(tzinfo=datetime.timezone.utc)
    return value.astimezone(datetime.timezone.utc)


def sample_consciousness_delay_minutes(person):
    ability_ranges = {
        "non_swimmer": (0.5, 3.0),
        "average": (1.0, 10.0),
        "strong": (5.0, 30.0),
        "unknown": (0.5, 15.0),
    }
    low, high = ability_ranges[person.swimming_ability]

    if person.age is not None and (person.age < 12 or person.age > 70):
        high *= 0.75

    if person.weight_kg is not None and person.weight_kg < 45:
        high *= 0.85

    mode = low + (high - low) * 0.35
    return random.triangular(low, high, mode)


def current_reader_from_dataframe(df):
    if df.empty or "uo" not in df.columns or "vo" not in df.columns:
        raise HTTPException(status_code=502, detail="Current data is empty or missing uo/vo columns.")

    uo = float(pd.to_numeric(df["uo"], errors="coerce").dropna().mean())
    vo = float(pd.to_numeric(df["vo"], errors="coerce").dropna().mean())

    if np.isnan(uo) or np.isnan(vo):
        raise HTTPException(status_code=502, detail="Current data did not contain valid velocity values.")

    return reader_constant.Reader({
        "x_sea_water_velocity": uo,
        "y_sea_water_velocity": vo,
        "x_wind": 2.0,
        "y_wind": -2.0,
    })


def seed_monte_carlo_elements(model, lats, lons, seed_times, object_type):
    try:
        model.seed_elements(
            lon=np.array(lons),
            lat=np.array(lats),
            time=np.array(seed_times, dtype=object),
            number=len(lats),
            object_type=object_type,
        )
    except Exception:
        for lat, lon, seed_time in zip(lats, lons, seed_times):
            model.seed_elements(
                lon=lon,
                lat=lat,
                time=seed_time,
                number=1,
                object_type=object_type,
            )


def run_opendrift_area_simulation(request):
    polygon_points = [(point.lat, point.lon) for point in request.polygon]
    validate_polygon_points(polygon_points)

    earliest_time = normalize_utc(request.earliest_time)
    latest_time = normalize_utc(request.latest_time)
    now_utc = datetime.datetime.now(datetime.timezone.utc)

    if latest_time < earliest_time:
        raise HTTPException(status_code=400, detail="latest_time must be after earliest_time.")

    if earliest_time > now_utc:
        raise HTTPException(status_code=400, detail="earliest_time must be in the past.")

    latest_time = min(latest_time, now_utc)
    elapsed_hours = (now_utc - earliest_time).total_seconds() / 3600
    if elapsed_hours > MAX_SIMULATION_HOURS:
        raise HTTPException(status_code=400, detail=f"Simulation is limited to {MAX_SIMULATION_DAYS} days.")

    center_lat = sum(point[0] for point in polygon_points) / len(polygon_points)
    center_lon = sum(point[1] for point in polygon_points) / len(polygon_points)

    df, source = get_current_data_for_area(
        center_lat,
        center_lon,
        earliest_time,
        now_utc + datetime.timedelta(hours=2),
        request.current_source,
    )

    lats = []
    lons = []
    incident_times = []
    seed_times = []
    consciousness_delays = []
    time_span_seconds = (latest_time - earliest_time).total_seconds()

    for _ in range(request.samples):
        lat, lon = sample_point_in_polygon(polygon_points)
        delay_minutes = sample_consciousness_delay_minutes(request.person)
        incident_time = earliest_time + datetime.timedelta(seconds=random.uniform(0, time_span_seconds))
        seed_time = min(incident_time + datetime.timedelta(minutes=delay_minutes), now_utc)

        lats.append(lat)
        lons.append(lon)
        incident_times.append(incident_time)
        seed_times.append(seed_time.replace(tzinfo=None))
        consciousness_delays.append(delay_minutes)

    model = Leeway(loglevel=30)
    model.add_reader(current_reader_from_dataframe(df))
    seed_monte_carlo_elements(model, lats, lons, seed_times, request.object_type)
    model.run(end_time=now_utc.replace(tzinfo=None), time_step=900, time_step_output=900)

    result = model.result
    lat_history = result["lat"].values
    lon_history = result["lon"].values

    particles = []
    for index, (lat_values, lon_values) in enumerate(zip(lat_history, lon_history)):
        valid_indices = np.where(~np.isnan(lat_values) & ~np.isnan(lon_values))[0]
        if len(valid_indices) == 0:
            continue

        final_index = valid_indices[-1]
        final_lat = lat_values[final_index]
        final_lon = lon_values[final_index]

        particles.append({
            "lat": round(float(final_lat), 6),
            "lon": round(float(final_lon), 6),
            "start_lat": round(float(lats[index]), 6),
            "start_lon": round(float(lons[index]), 6),
            "incident_time": incident_times[index].isoformat(),
            "drift_start_time": seed_times[index].replace(tzinfo=datetime.timezone.utc).isoformat(),
            "consciousness_delay_minutes": round(float(consciousness_delays[index]), 2),
            "probability": 1 / request.samples,
        })

    if not particles:
        raise HTTPException(status_code=502, detail="OpenDrift did not return any final particle positions.")

    return {
        "status": "success",
        "simulation": "opendrift_monte_carlo",
        "samples_requested": request.samples,
        "samples_returned": len(particles),
        "data_source_used": source,
        "time_window": {
            "earliest_time": earliest_time.isoformat(),
            "latest_time": latest_time.isoformat(),
            "evaluated_until": now_utc.isoformat(),
        },
        "person_model": request.person.model_dump(),
        "particles": particles,
        "center": {
            "lat": round(float(np.mean([particle["lat"] for particle in particles])), 6),
            "lon": round(float(np.mean([particle["lon"] for particle in particles])), 6),
        },
    }

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


@app.post("/simulate-area")
def simulate_area(request: AreaSimulationRequest):
    return run_opendrift_area_simulation(request)
