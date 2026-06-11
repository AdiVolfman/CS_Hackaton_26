from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import datetime
import pandas as pd
import numpy as np
import requests

from fetch_current import CopernicusFetcher

app = FastAPI(title="SafeCurrent - Search & Rescue API")

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
        df = CopernicusFetcher().fetch(lat, lon, start_time, end_time)
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

def calculate_next_position(current_lat, current_lon, target_time, df, hour_index, source):
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
    
    delta_lat = (total_vo * 3600) / meters_per_degree_lat
    delta_lon = (total_uo * 3600) / meters_per_degree_lon
    
    return current_lat + delta_lat, current_lon + delta_lon

# --- API ENDPOINT ---

@app.get("/simulate")
def simulate_drift(
    lat: float = Query(..., description="Latitude of last seen position"),
    lon: float = Query(..., description="Longitude of last seen position"),
    hours: int = Query(5, description="Number of hours to simulate simulation")
):
    start_time = datetime.datetime.now(datetime.timezone.utc)
    end_time = start_time + datetime.timedelta(hours=hours + 2)
    
    # Fetch Data
    df, source = get_current_data_fallback(lat, lon, start_time, end_time)
    
    # Run Simulation
    trajectory = [{"hour": 0, "lat": lat, "lon": lon}]
    current_lat = lat
    current_lon = lon
    current_time = pd.to_datetime(start_time)
    
    for h in range(hours):
        current_time += pd.to_timedelta(1, unit='h')
        next_lat, next_lon = calculate_next_position(current_lat, current_lon, current_time, df, hour_index=h, source=source)
        
        trajectory.append({
            "hour": h + 1,
            "lat": round(float(next_lat), 5),
            "lon": round(float(next_lon), 5)
        })
        current_lat, current_lon = next_lat, next_lon
        
    return {
        "status": "success",
        "data_source_used": source,
        "initial_position": {"lat": lat, "lon": lon},
        "hours_simulated": hours,
        "trajectory": trajectory
    }