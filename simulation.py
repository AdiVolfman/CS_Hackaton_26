import pandas as pd
import numpy as np

def calculate_next_position(current_lat, current_lon, target_time, df, hour_index):
    """
    Calculates the next Lat/Lon coordinates after 1 hour of drifting.
    """
    # 1. Find the rows matching the closest hour in the dataset
    df['time'] = pd.to_datetime(df['time'])
    closest_time = df['time'].iloc[(df['time'] - target_time).abs().argsort()[:1]].values[0]
    hourly_df = df[df['time'] == closest_time]
    
    if hourly_df.empty:
        return current_lat, current_lon
    
    # 2. Find the closest grid point in Copernicus data to our current person's location
    distances = np.sqrt((hourly_df['latitude'] - current_lat)**2 + (hourly_df['longitude'] - current_lon)**2)
    closest_row = hourly_df.iloc[distances.idxmin()]
    
    # Extract velocities (meters per second)
    uo = closest_row['uo']  # East/West
    vo = closest_row['vo']  # North/South
    
    # 3. CRITICAL ISRAELI COASTAL LOGIC: Rip Current Simulation
    # If it's the first hour of drowning, add a strong artificial push Westward (out to sea)
    rip_current_push_uo = 0.0
    if hour_index == 0:
        # Pushing Westward roughly 1.5 knots (~0.75 m/s)
        rip_current_push_uo = -0.75 
    
    # Total velocity components
    total_uo = uo + rip_current_push_uo
    total_vo = vo
    
    # 4. Convert meters per second to change in Degrees (Lat/Lon) over 1 hour (3600 seconds)
    # Earth radius approximation for degree conversion
    meters_per_degree_lat = 111000
    meters_per_degree_lon = 111000 * np.cos(np.radians(current_lat))
    
    delta_lat = (total_vo * 3600) / meters_per_degree_lat
    delta_lon = (total_uo * 3600) / meters_per_degree_lon
    
    # New coordinates
    next_lat = current_lat + delta_lat
    next_lon = current_lon + delta_lon
    
    return next_lat, next_lon

def run_drift_simulation(start_lat, start_lon, start_time, hours_elapsed, df):
    """
    Runs the simulation hour by hour and returns a list of coordinates.
    """
    trajectory = [{"hour": 0, "lat": start_lat, "lon": start_lon}]
    
    current_lat = start_lat
    current_lon = start_lon
    current_time = pd.to_datetime(start_time)
    
    for h in range(hours_elapsed):
        current_time += pd.to_timedelta(1, unit='h')
        next_lat, next_lon = calculate_next_position(current_lat, current_lon, current_time, df, hour_index=h)
        
        trajectory.append({
            "hour": h + 1,
            "lat": next_lat,
            "lon": next_lon
        })
        current_lat, current_lon = next_lat, next_lon
        
    return trajectory

# --- HOW TO INTEGRATE THIS WITH YOUR CODE ---
# inside your main block after creating 'df':
# start_lat = 32.080  # Example: Gordon Beach Tel Aviv
# start_lon = 34.765
# result = run_drift_simulation(start_lat, start_lon, start_time, hours_elapsed=5, df=df)
# print(result)