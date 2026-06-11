from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import datetime
from typing import Optional
import pandas as pd
import numpy as np
import requests
import copernicusmarine

app = FastAPI(title="SafeCurrent - Search & Rescue API")
MAX_SIMULATION_HOURS = 24

# CRITICAL FOR HACKATHON: Allows your Frontend (React/Vue/HTML) to talk to this Backend without CORS blocks
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CORE SIMULATION LOGIC ---

def get_current_data_fallback(lat, lon, start_time, end_time):
    """
    Tries to fetch from Copernicus. If it fails, instantly switches to Open-Meteo API
    to ensure the hackathon app never crashes.
    """
    try:
        print("Attempting to fetch from Copernicus Official API...")
        ds = copernicusmarine.read(
            dataset_id="med-cmcc-phys-an-fc-h",
            variables=["uo", "vo"],
            longitude_min=lon - 0.5,
            longitude_max=lon + 0.5,
            latitude_min=lat - 0.5,
            latitude_max=lat + 0.5,
            start_date=start_time.strftime("%Y-%m-%dT%H:%M:%S"),
            end_date=end_time.strftime("%Y-%m-%dT%H:%M:%S")
        )
        df = ds.to_dataframe().reset_index()
        return df, "copernicus"
    except Exception as e:
        print(f"Copernicus failed ({e}). Switching to Open-Meteo Marine API Backup...")
        
        # Open-Meteo fallback url
        url = f"https://marine-api.open-meteo.com/v1/marine?latitude={lat}&longitude={lon}&hourly=ocean_current_velocity,ocean_current_direction"
        response = requests.get(url).json()
        
        # Parse JSON into a simple dataframe structure that matches our needs
        hourly = response.get('hourly', {})
        times = hourly.get('time', [])
        velocities = hourly.get('ocean_current_velocity', []) # given in km/h
        directions = hourly.get('ocean_current_direction', []) # given in degrees
        
        rows = []
        for t, v, d in zip(times, velocities, directions):
            # Convert km/h to m/s
            speed_ms = (v * 1000) / 3600
            rad = np.radians(d)
            # Break speed into uo (East) and vo (North) vectors
            uo = speed_ms * np.sin(rad)
            vo = speed_ms * np.cos(rad)
            rows.append({"time": pd.to_datetime(t), "latitude": lat, "longitude": lon, "uo": uo, "vo": vo})
            
        return pd.DataFrame(rows), "open-meteo"

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

def resolve_simulation_window(hours, minutes, drowning_time):
    now_utc = datetime.datetime.now(datetime.timezone.utc)

    if drowning_time is not None and (hours is not None or minutes is not None):
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
        if hours is None and minutes is None:
            elapsed_hours = 5
        else:
            elapsed_hours = (0 if hours is None else hours) + (0 if minutes is None else minutes / 60)
        start_time = now_utc - datetime.timedelta(hours=elapsed_hours)

    if elapsed_hours <= 0:
        raise HTTPException(status_code=400, detail="The drowning time must be in the past.")

    if elapsed_hours > MAX_SIMULATION_HOURS:
        raise HTTPException(status_code=400, detail=f"Simulation is limited to {MAX_SIMULATION_HOURS} hours.")

    return start_time, now_utc, elapsed_hours

# --- API ENDPOINT ---

@app.get("/simulate")
def simulate_drift(
    lat: float = Query(..., description="Latitude of last seen position"),
    lon: float = Query(..., description="Longitude of last seen position"),
    hours: Optional[float] = Query(None, ge=0, le=MAX_SIMULATION_HOURS, description="Hours since drowning"),
    minutes: Optional[float] = Query(None, ge=0, le=MAX_SIMULATION_HOURS * 60, description="Minutes since drowning"),
    drowning_time: Optional[datetime.datetime] = Query(None, description="UTC drowning time as an ISO timestamp")
):
    start_time, now_utc, elapsed_hours = resolve_simulation_window(hours, minutes, drowning_time)
    end_time = now_utc + datetime.timedelta(hours=1)
    
    # Fetch Data
    df, source = get_current_data_fallback(lat, lon, start_time, end_time)
    
    # Run Simulation
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
            "lat": round(next_lat, 5),
            "lon": round(next_lon, 5)
        })
        current_lat, current_lon = next_lat, next_lon
        remaining_hours = max(0.0, remaining_hours - step_hours)
        hour_index += 1
        
    return {
        "status": "success",
        "data_source_used": source,
        "initial_position": {"lat": lat, "lon": lon},
        "hours_simulated": round(elapsed_hours, 2),
        "trajectory": trajectory
    }
