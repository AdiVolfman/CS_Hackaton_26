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
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from fetch_current import CopernicusFetcher, KinneretFetcher, is_kinneret
from simulation_window import resolve_simulation_window
from share_store import router as share_router
from local_current import parse_local_current, get_local_current_vector
import xarray as xr
from opendrift.models.oceandrift import OceanDrift
from opendrift.readers.reader_netCDF_CF_generic import Reader as NetCDFReader

app = FastAPI(title="SafeCurrent - Search & Rescue API")
app.include_router(share_router)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("safecurrent")

app = FastAPI(title="SafeCurrent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
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


def run_oceandrift(
    polygon_lonlat: list[tuple[float, float]],
    point_latlon: Optional[tuple[float, float]],
    point_radius_m: float,
    entry_start: _dt.datetime,
    entry_end: _dt.datetime,
    forecast_time: _dt.datetime,
    n_particles: int,
) -> dict:
    """Run OpenDrift OceanDrift (currents-only) and return final particle positions.

    Particles are seeded:
      - uniformly inside `polygon_lonlat` if it has >= 3 points,
      - else in a Gaussian cloud of std-dev `point_radius_m / 2` around `point_latlon`.

    Each particle is given a random release time in [entry_start, entry_end]
    so that the time-uncertainty from the operator's input is captured.
    Wind is NOT used; this is pure ocean-current advection.
    """
    if polygon_lonlat and len(polygon_lonlat) >= 3:
        lons = [p[0] for p in polygon_lonlat]
        lats = [p[1] for p in polygon_lonlat]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        seed_kind = "polygon"
    elif point_latlon is not None:
        plat, plon = point_latlon
        deg_pad = max(0.005, point_radius_m / 90000.0)
        min_lon, max_lon = plon - deg_pad, plon + deg_pad
        min_lat, max_lat = plat - deg_pad, plat + deg_pad
        seed_kind = "point"
    else:
        raise ValueError("Provide a polygon or a point.")

    fetch_pad = 0.3
    cur_path = fetch_currents_copernicus(
        min_lon - fetch_pad, min_lat - fetch_pad,
        max_lon + fetch_pad, max_lat + fetch_pad,
        entry_start - _dt.timedelta(hours=2),
        forecast_time + _dt.timedelta(hours=2),
    )

    o = OceanDrift(loglevel=30)
    o.add_reader([NetCDFReader(cur_path)])
    # OceanDrift can apply a random horizontal diffusivity to spread the
    # ensemble realistically. 1 m^2/s is a reasonable coastal SAR default.
    o.set_config("drift:horizontal_diffusivity", 1.0)
    # Don't deactivate particles when they reach the coastline; we want all
    # final positions for the heatmap.
    o.set_config("general:coastline_action", "previous")

    rng = np.random.default_rng()
    window_s = max(1.0, (entry_end - entry_start).total_seconds())
    offsets = rng.uniform(0.0, window_s, size=n_particles)
    times = [entry_start + _dt.timedelta(seconds=float(s)) for s in offsets]

    if seed_kind == "polygon":
        poly_lons, poly_lats = _sample_polygon(polygon_lonlat, n_particles, rng)
        o.seed_elements(
            lon=poly_lons.tolist(),
            lat=poly_lats.tolist(),
            time=times,
        )
    else:
        plat, plon = point_latlon
        sigma_deg_lat = point_radius_m / 111000.0
        sigma_deg_lon = point_radius_m / (111000.0 * np.cos(np.radians(plat)))
        seed_lats = plat + rng.normal(0.0, sigma_deg_lat, size=n_particles)
        seed_lons = plon + rng.normal(0.0, sigma_deg_lon, size=n_particles)
        o.seed_elements(
            lon=seed_lons.tolist(),
            lat=seed_lats.tolist(),
            time=times,
        )

    o.run(end_time=forecast_time, time_step=600, time_step_output=3600)

    final_lons = np.atleast_1d(o.elements.lon)
    final_lats = np.atleast_1d(o.elements.lat)
    deactivated_lons = np.atleast_1d(o.elements_deactivated.lon) if hasattr(o.elements_deactivated, 'lon') and len(o.elements_deactivated.lon) > 0 else np.array([])
    deactivated_lats = np.atleast_1d(o.elements_deactivated.lat) if hasattr(o.elements_deactivated, 'lat') and len(o.elements_deactivated.lat) > 0 else np.array([])
    all_lons = np.concatenate([final_lons, deactivated_lons])
    all_lats = np.concatenate([final_lats, deactivated_lats])

    try:
        os.unlink(cur_path)
    except Exception:
        pass

    return {
        "lons": all_lons,
        "lats": all_lats,
        "n_particles": int(all_lons.size),
        "currents_source": "copernicus",
        "winds_source": "none (currents-only)",
    }



def particles_to_heatmap(lons, lats, grid_resolution=80, padding_deg=0.01):
    """Bin particle positions into a normalised 2D probability grid.

    Equation: intensity[i,j] = count[i,j] / max(count). The output is a list
    of [lat, lon, intensity] entries — Leaflet.heat's expected format.
    """
    if lons.size == 0:
        return [], None

    min_lon = float(lons.min()) - padding_deg
    max_lon = float(lons.max()) + padding_deg
    min_lat = float(lats.min()) - padding_deg
    max_lat = float(lats.max()) + padding_deg
    span_lon = max_lon - min_lon
    span_lat = max_lat - min_lat
    if span_lon <= 0 or span_lat <= 0:
        c_lat = float(lats.mean())
        c_lon = float(lons.mean())
        return [[c_lat, c_lon, 1.0]], {
            "min_lat": c_lat - 0.001, "max_lat": c_lat + 0.001,
            "min_lon": c_lon - 0.001, "max_lon": c_lon + 0.001,
        }

    if span_lon >= span_lat:
        nx = grid_resolution
        ny = max(2, int(round(grid_resolution * span_lat / span_lon)))
    else:
        ny = grid_resolution
        nx = max(2, int(round(grid_resolution * span_lon / span_lat)))

    counts, lon_edges, lat_edges = np.histogram2d(
        lons, lats, bins=[nx, ny], range=[[min_lon, max_lon], [min_lat, max_lat]]
    )
    if counts.max() <= 0:
        return [], None
    intensity = counts / counts.max()
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])

    points = []
    for ix in range(nx):
        for iy in range(ny):
            w = float(intensity[ix, iy])
            if w <= 0.01:
                continue
            points.append([float(lat_centers[iy]), float(lon_centers[ix]), w])

    return points, {
        "min_lat": min_lat, "max_lat": max_lat,
        "min_lon": min_lon, "max_lon": max_lon,
    }



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
