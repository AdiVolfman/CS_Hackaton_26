"""SafeCurrent backend — single-file pipeline.

Flow:
  Frontend  ->  POST /simulate with a polygon (or single point) + time window
              backend fetches Copernicus currents + Open-Meteo wind
              backend runs OpenDrift Leeway with N particles
              backend bins final positions into a probability grid (heatmap)
  Frontend  <-  list of [lat, lon, intensity] for Leaflet.heat

Equations are documented inline in the simulation step.
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
import tempfile
from typing import Optional

import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import xarray as xr
from opendrift.models.leeway import Leeway
from opendrift.readers.reader_netCDF_CF_generic import Reader as NetCDFReader

import copernicusmarine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("safecurrent")

app = FastAPI(title="SafeCurrent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Med Sea hourly currents at 4.2 km, 2D surface
COPERNICUS_DATASET = "cmems_mod_med_phy-cur_anfc_4.2km-2D_PT1H-m"
# Open-Meteo wind endpoint
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


# ---------------------------------------------------------------------------
# Schemas


class PolygonPoint(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)


class SimulateRequest(BaseModel):
    polygon: Optional[list[PolygonPoint]] = None
    point: Optional[PolygonPoint] = None
    point_radius_m: float = Field(150.0, ge=10.0, le=5000.0)
    entry_window_start: _dt.datetime
    entry_window_end: _dt.datetime
    forecast_time: Optional[_dt.datetime] = None
    n_particles: int = Field(1000, ge=50, le=10000)
    # Leeway category. PIW-1 = Person In Water, unknown state (default).
    # Others: PIW-2 vertical conscious, PIW-3 sitting, PIW-4 survival suit,
    # PIW-5 scuba, PIW-6 deceased face-down.
    object_type: int = Field(1, ge=1, le=6)


# ---------------------------------------------------------------------------
# Data fetching


def fetch_currents_copernicus(min_lon, min_lat, max_lon, max_lat, t0, t1) -> str:
    """Download the Copernicus current field as a NetCDF file and return its path.

    OpenDrift's NetCDF reader works on a local file rather than an in-memory
    xarray, which is why we materialise to disk.
    """
    out = tempfile.NamedTemporaryFile(suffix=".nc", delete=False).name
    log.info("Downloading Copernicus currents to %s", out)
    copernicusmarine.subset(
        dataset_id=COPERNICUS_DATASET,
        variables=["uo", "vo"],
        minimum_longitude=min_lon,
        maximum_longitude=max_lon,
        minimum_latitude=min_lat,
        maximum_latitude=max_lat,
        start_datetime=t0.strftime("%Y-%m-%dT%H:%M:%S"),
        end_datetime=t1.strftime("%Y-%m-%dT%H:%M:%S"),
        output_filename=os.path.basename(out),
        output_directory=os.path.dirname(out),
    )
    return out


def fetch_wind_open_meteo(lat, lon, t0, t1) -> str:
    """Fetch hourly 10-m wind from Open-Meteo and write a CF-compliant NetCDF.

    OpenDrift's reader_netCDF_CF_generic accepts files with x_wind_10m and
    y_wind_10m variables on a (time, lat, lon) grid; for a single point we
    create a 1x1 spatial grid that the reader will broadcast.
    """
    now = _dt.datetime.utcnow()
    past_days = max(0, int((now - t0).total_seconds() // 86400) + 1)
    forecast_days = max(1, int((t1 - now).total_seconds() // 86400) + 1)
    past_days = min(past_days, 7)
    forecast_days = min(max(forecast_days, 1), 7)

    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "wind_speed_10m,wind_direction_10m",
        "wind_speed_unit": "ms",
        "past_days": past_days, "forecast_days": forecast_days,
        "timezone": "UTC",
    }
    r = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=20)
    r.raise_for_status()
    payload = r.json()
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    speeds = hourly.get("wind_speed_10m", [])
    dirs = hourly.get("wind_direction_10m", [])

    # Convert "direction from" + speed to (u, v) "direction toward" components.
    # Equation: u = -speed * sin(dir_rad);  v = -speed * cos(dir_rad)
    rad = np.radians(np.array(dirs, dtype=float))
    speed = np.array(speeds, dtype=float)
    u10 = -speed * np.sin(rad)
    v10 = -speed * np.cos(rad)

    # Build a tiny 1x1 lat/lon grid spanning Israel to keep OpenDrift happy.
    lats = np.array([lat - 0.5, lat + 0.5], dtype=float)
    lons = np.array([lon - 0.5, lon + 0.5], dtype=float)
    nT = len(times)
    u_grid = np.broadcast_to(u10[:, None, None], (nT, 2, 2)).astype(np.float32)
    v_grid = np.broadcast_to(v10[:, None, None], (nT, 2, 2)).astype(np.float32)

    times_dt = np.array([np.datetime64(t) for t in times], dtype="datetime64[ns]")
    ds = xr.Dataset(
        data_vars={
            "x_wind_10m": (("time", "latitude", "longitude"), u_grid, {"units": "m s-1", "standard_name": "x_wind"}),
            "y_wind_10m": (("time", "latitude", "longitude"), v_grid, {"units": "m s-1", "standard_name": "y_wind"}),
        },
        coords={
            "time": times_dt,
            "latitude": ("latitude", lats, {"standard_name": "latitude", "units": "degrees_north"}),
            "longitude": ("longitude", lons, {"standard_name": "longitude", "units": "degrees_east"}),
        },
    )
    out = tempfile.NamedTemporaryFile(suffix=".nc", delete=False).name
    ds.to_netcdf(out)
    return out


# ---------------------------------------------------------------------------
# Simulation


def run_leeway(
    polygon_lonlat: list[tuple[float, float]],
    point_latlon: Optional[tuple[float, float]],
    point_radius_m: float,
    entry_start: _dt.datetime,
    entry_end: _dt.datetime,
    forecast_time: _dt.datetime,
    n_particles: int,
    object_type: int,
) -> dict:
    """Run OpenDrift Leeway and return final particle positions.

    Particles are seeded:
      - uniformly inside `polygon_lonlat` if it has >= 3 points,
      - else in a Gaussian cloud of std-dev `point_radius_m / 2` around `point_latlon`.

    Each particle is given a random release time in [entry_start, entry_end]
    so that the time-uncertainty from the operator's input is captured.
    """
    if polygon_lonlat and len(polygon_lonlat) >= 3:
        lons = [p[0] for p in polygon_lonlat]
        lats = [p[1] for p in polygon_lonlat]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        seed_kind = "polygon"
    elif point_latlon is not None:
        plat, plon = point_latlon
        # Seed bbox a bit larger than the radius for safety.
        deg_pad = max(0.005, point_radius_m / 90000.0)
        min_lon, max_lon = plon - deg_pad, plon + deg_pad
        min_lat, max_lat = plat - deg_pad, plat + deg_pad
        seed_kind = "point"
    else:
        raise ValueError("Provide a polygon or a point.")

    # Pad the data-fetch bbox so OpenDrift always has surrounding cells.
    fetch_pad = 0.3
    cur_path = fetch_currents_copernicus(
        min_lon - fetch_pad, min_lat - fetch_pad,
        max_lon + fetch_pad, max_lat + fetch_pad,
        entry_start - _dt.timedelta(hours=2),
        forecast_time + _dt.timedelta(hours=2),
    )
    centroid_lat = 0.5 * (min_lat + max_lat)
    centroid_lon = 0.5 * (min_lon + max_lon)
    wind_path = fetch_wind_open_meteo(
        centroid_lat, centroid_lon,
        entry_start - _dt.timedelta(hours=2),
        forecast_time + _dt.timedelta(hours=2),
    )

    o = Leeway(loglevel=30)
    o.add_reader([NetCDFReader(cur_path), NetCDFReader(wind_path)])

    # Random per-particle release times within the entry window.
    rng = np.random.default_rng()
    window_s = max(1.0, (entry_end - entry_start).total_seconds())
    offsets = rng.uniform(0.0, window_s, size=n_particles)
    times = [entry_start + _dt.timedelta(seconds=float(s)) for s in offsets]

    if seed_kind == "polygon":
        # Sample uniformly inside the polygon.
        poly_lons, poly_lats = _sample_polygon(polygon_lonlat, n_particles, rng)
        o.seed_elements(
            lon=poly_lons.tolist(),
            lat=poly_lats.tolist(),
            time=times,
            object_type=object_type,
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
            object_type=object_type,
        )

    # Simulate to forecast_time. OpenDrift advances each particle from its
    # release time to the simulation end. Time step 600s is the SAR default.
    duration = forecast_time - entry_start
    o.run(end_time=forecast_time, time_step=600, time_step_output=3600)

    final_lons = np.atleast_1d(o.elements.lon)
    final_lats = np.atleast_1d(o.elements.lat)
    deactivated_lons = np.atleast_1d(o.elements_deactivated.lon) if o.elements_deactivated.num_elements_active() > 0 else np.array([])
    deactivated_lats = np.atleast_1d(o.elements_deactivated.lat) if o.elements_deactivated.num_elements_active() > 0 else np.array([])
    all_lons = np.concatenate([final_lons, deactivated_lons])
    all_lats = np.concatenate([final_lats, deactivated_lats])

    # Cleanup temp files
    try:
        os.unlink(cur_path)
        os.unlink(wind_path)
    except Exception:
        pass

    return {
        "lons": all_lons,
        "lats": all_lats,
        "n_particles": int(all_lons.size),
        "currents_source": "copernicus",
        "winds_source": "open-meteo",
    }


def _sample_polygon(polygon_lonlat, n, rng):
    """Uniform sample n points inside a 2D polygon by rejection."""
    from shapely.geometry import Polygon, Point as ShPoint
    poly = Polygon(polygon_lonlat)
    if not poly.is_valid:
        poly = poly.buffer(0)
    minx, miny, maxx, maxy = poly.bounds
    out_lons = np.empty(n, dtype=float)
    out_lats = np.empty(n, dtype=float)
    filled = 0
    while filled < n:
        bx = rng.uniform(minx, maxx, size=max(n - filled, 256))
        by = rng.uniform(miny, maxy, size=bx.size)
        for x, y in zip(bx, by):
            if filled >= n:
                break
            if poly.contains(ShPoint(x, y)):
                out_lons[filled] = x
                out_lats[filled] = y
                filled += 1
    return out_lons, out_lats


# ---------------------------------------------------------------------------
# Heatmap


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


# ---------------------------------------------------------------------------
# Endpoint


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/simulate")
def simulate(req: SimulateRequest):
    polygon = [(p.lon, p.lat) for p in (req.polygon or [])]
    if not polygon and req.point is None:
        raise HTTPException(400, "Provide a polygon or a point.")

    entry_start = req.entry_window_start.replace(tzinfo=None) if req.entry_window_start.tzinfo else req.entry_window_start
    entry_end = req.entry_window_end.replace(tzinfo=None) if req.entry_window_end.tzinfo else req.entry_window_end
    if entry_end < entry_start:
        raise HTTPException(400, "entry_window_end must be after entry_window_start.")

    forecast_time = req.forecast_time
    if forecast_time is None:
        forecast_time = _dt.datetime.utcnow()
    elif forecast_time.tzinfo:
        forecast_time = forecast_time.astimezone(_dt.timezone.utc).replace(tzinfo=None)
    if forecast_time <= entry_start:
        raise HTTPException(400, "forecast_time must be after entry_window_start.")

    try:
        result = run_leeway(
            polygon_lonlat=polygon,
            point_latlon=(req.point.lat, req.point.lon) if req.point else None,
            point_radius_m=req.point_radius_m,
            entry_start=entry_start,
            entry_end=entry_end,
            forecast_time=forecast_time,
            n_particles=req.n_particles,
            object_type=req.object_type,
        )
    except Exception as exc:
        log.exception("Simulation failed")
        raise HTTPException(500, f"Simulation failed: {exc}")

    points, bbox = particles_to_heatmap(result["lons"], result["lats"])

    return {
        "status": "success",
        "n_particles": result["n_particles"],
        "currents_source": result["currents_source"],
        "winds_source": result["winds_source"],
        "forecast_time": forecast_time.isoformat() + "Z",
        "initial_polygon": [{"lat": p[1], "lon": p[0]} for p in polygon],
        "heatmap": points,
        "bbox": bbox,
    }
