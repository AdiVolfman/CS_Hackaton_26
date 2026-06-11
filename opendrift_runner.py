"""OpenDrift adapter.

Takes a pandas DataFrame from our fetchers (columns: time, latitude, longitude, uo, vo)
and runs OpenDrift's OceanDrift model on it, returning final particle positions.

Why a temp NetCDF: OpenDrift's `Reader` family reads files with CF-compliant grids,
not in-memory pandas. Cheapest bridge is df → xarray.Dataset → tmp .nc → NetCDFReader.
"""

from __future__ import annotations

import datetime
import os
import tempfile
import uuid
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from opendrift.models.oceandrift import OceanDrift
from opendrift.readers.reader_netCDF_CF_generic import Reader as NetCDFReader


def df_to_netcdf(df: pd.DataFrame) -> str:
    """Convert our (time, latitude, longitude, uo, vo) DataFrame to a CF-compliant
    NetCDF on disk. Returns the path.

    Handles two input shapes:
      - Gridded (Copernicus): many (lat, lon) per time. Pivoted to a real 2D grid.
      - Single-point (Kinneret): one (lat, lon) per time. Replicated across a tiny
        synthetic grid so OpenDrift has a field to advect through. Particles spread
        within the lake will all see the same uo/vo, which matches the underlying
        wind-derived assumption.
    """
    if df.empty:
        raise ValueError("Current DataFrame is empty; nothing to feed OpenDrift.")

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)

    times = np.array(sorted(df["time"].unique()))
    unique_lats = np.array(sorted(df["latitude"].unique()))
    unique_lons = np.array(sorted(df["longitude"].unique()))

    if len(unique_lats) == 1 and len(unique_lons) == 1:
        # Kinneret: synthesize a small grid by replicating the single point.
        center_lat = float(unique_lats[0])
        center_lon = float(unique_lons[0])
        pad = 0.05  # ~5 km — covers Kinneret comfortably
        unique_lats = np.linspace(center_lat - pad, center_lat + pad, 5)
        unique_lons = np.linspace(center_lon - pad, center_lon + pad, 5)

        uo_3d = np.zeros((len(times), len(unique_lats), len(unique_lons)), dtype=float)
        vo_3d = np.zeros_like(uo_3d)
        for t_idx, t in enumerate(times):
            row = df[df["time"] == t].iloc[0]
            uo_3d[t_idx, :, :] = float(row["uo"])
            vo_3d[t_idx, :, :] = float(row["vo"])
    else:
        # Copernicus: pivot to a real grid.
        df = df.sort_values(["time", "latitude", "longitude"])
        uo_3d = np.full((len(times), len(unique_lats), len(unique_lons)), np.nan)
        vo_3d = np.full_like(uo_3d, np.nan)
        lat_idx = {lat: i for i, lat in enumerate(unique_lats)}
        lon_idx = {lon: i for i, lon in enumerate(unique_lons)}
        time_idx = {t: i for i, t in enumerate(times)}
        for row in df.itertuples(index=False):
            ti = time_idx[row.time]
            la = lat_idx[row.latitude]
            lo = lon_idx[row.longitude]
            uo_3d[ti, la, lo] = row.uo
            vo_3d[ti, la, lo] = row.vo
        uo_3d = np.nan_to_num(uo_3d, nan=0.0)
        vo_3d = np.nan_to_num(vo_3d, nan=0.0)

    ds = xr.Dataset(
        data_vars={
            "uo": (("time", "latitude", "longitude"), uo_3d, {
                "standard_name": "eastward_sea_water_velocity",
                "units": "m s-1",
            }),
            "vo": (("time", "latitude", "longitude"), vo_3d, {
                "standard_name": "northward_sea_water_velocity",
                "units": "m s-1",
            }),
        },
        coords={
            "time": ("time", times),
            "latitude": ("latitude", unique_lats, {
                "standard_name": "latitude",
                "units": "degrees_north",
            }),
            "longitude": ("longitude", unique_lons, {
                "standard_name": "longitude",
                "units": "degrees_east",
            }),
        },
        attrs={"Conventions": "CF-1.7"},
    )

    out_dir = tempfile.gettempdir()
    out_path = os.path.join(out_dir, f"safecurrent_{uuid.uuid4().hex}.nc")
    ds.to_netcdf(out_path)
    return out_path


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


def run_oceandrift(
    df: pd.DataFrame,
    polygon_lonlat: Optional[list[tuple[float, float]]],
    point_latlon: Optional[tuple[float, float]],
    point_radius_m: float,
    entry_start: datetime.datetime,
    entry_end: datetime.datetime,
    forecast_time: datetime.datetime,
    n_particles: int,
) -> dict:
    """Run OpenDrift on currents already fetched into a DataFrame.

    Particles are seeded uniformly in the polygon (>=3 points) or in a Gaussian
    cloud of std `point_radius_m / 2` around `point_latlon`. Each gets a random
    release time in [entry_start, entry_end] to capture entry-time uncertainty.
    Wind is not used; pure ocean-current advection.
    """
    if (not polygon_lonlat or len(polygon_lonlat) < 3) and point_latlon is None:
        raise ValueError("Provide a polygon or a point.")

    nc_path = df_to_netcdf(df)

    try:
        o = OceanDrift(loglevel=30)
        o.add_reader([NetCDFReader(nc_path)])
        o.set_config("drift:horizontal_diffusivity", 1.0)
        o.set_config("general:coastline_action", "previous")

        rng = np.random.default_rng()
        window_s = max(1.0, (entry_end - entry_start).total_seconds())
        offsets = rng.uniform(0.0, window_s, size=n_particles)
        times = [entry_start + datetime.timedelta(seconds=float(s)) for s in offsets]

        if polygon_lonlat and len(polygon_lonlat) >= 3:
            poly_lons, poly_lats = _sample_polygon(polygon_lonlat, n_particles, rng)
            o.seed_elements(lon=poly_lons.tolist(), lat=poly_lats.tolist(), time=times)
        else:
            plat, plon = point_latlon
            sigma_deg_lat = point_radius_m / 111000.0
            sigma_deg_lon = point_radius_m / (111000.0 * np.cos(np.radians(plat)))
            seed_lats = plat + rng.normal(0.0, sigma_deg_lat, size=n_particles)
            seed_lons = plon + rng.normal(0.0, sigma_deg_lon, size=n_particles)
            o.seed_elements(lon=seed_lons.tolist(), lat=seed_lats.tolist(), time=times)

        o.run(end_time=forecast_time, time_step=600, time_step_output=3600)

        final_lons = np.atleast_1d(o.elements.lon)
        final_lats = np.atleast_1d(o.elements.lat)
        deactivated_lons = (
            np.atleast_1d(o.elements_deactivated.lon)
            if hasattr(o.elements_deactivated, "lon") and len(o.elements_deactivated.lon) > 0
            else np.array([])
        )
        deactivated_lats = (
            np.atleast_1d(o.elements_deactivated.lat)
            if hasattr(o.elements_deactivated, "lat") and len(o.elements_deactivated.lat) > 0
            else np.array([])
        )
        all_lons = np.concatenate([final_lons, deactivated_lons])
        all_lats = np.concatenate([final_lats, deactivated_lats])

        return {
            "lons": all_lons,
            "lats": all_lats,
            "n_particles": int(all_lons.size),
        }
    finally:
        try:
            os.unlink(nc_path)
        except OSError:
            pass


def particles_to_heatmap(lons, lats, grid_resolution=80, padding_deg=0.01):
    """Bin particle positions into a normalised 2D probability grid.

    intensity[i,j] = count[i,j] / max(count). Output is a list of
    [lat, lon, intensity] entries — Leaflet.heat's expected format.
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
