from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import datetime
import json
import logging
import pathlib
import threading
from typing import Optional

from fetch_current import CopernicusFetcher, KinneretFetcher, is_kinneret
from simulation_window import resolve_simulation_window
from share_store import router as share_router
from opendrift_runner import run_oceandrift, particles_to_heatmap

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("safecurrent")

app = FastAPI(title="SafeCurrent - Search & Rescue API")
app.include_router(share_router)
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


def parse_polygon_points(polygon):
    """Parse a JSON polygon query param into a list of (lat, lon) tuples."""
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


@app.get("/simulate")
def simulate_drift(
    lat: float = Query(..., description="Latitude of last seen position"),
    lon: float = Query(..., description="Longitude of last seen position"),
    days: Optional[float] = Query(None),
    hours: Optional[float] = Query(None),
    minutes: Optional[float] = Query(None),
    drowning_time: Optional[datetime.datetime] = Query(None),
    polygon: Optional[str] = Query(None),
    floating: bool = Query(False, description="True = surface (2D); False = seabed (3D)"),
    n_particles: int = Query(1000, ge=50, le=10000),
    point_radius_m: float = Query(150.0, ge=10.0, le=5000.0),
):
    """Run an OpenDrift particle ensemble for SAR drift prediction.

    Particles are seeded inside the polygon (if given) or in a Gaussian cloud
    around (lat, lon) with std `point_radius_m`. They release at random times
    inside the simulation window and are advected by ocean currents until
    `now_utc`. Final positions are binned into a heatmap.
    """
    start_time, now_utc, elapsed_hours = resolve_simulation_window(days, hours, minutes, drowning_time)
    end_time = now_utc + datetime.timedelta(hours=2)

    df, source = get_current_data(lat, lon, start_time, end_time, floating=floating)
    polygon_points = parse_polygon_points(polygon)
    polygon_lonlat = [(p_lon, p_lat) for p_lat, p_lon in polygon_points] if polygon_points else None

    try:
        result = run_oceandrift(
            df=df,
            polygon_lonlat=polygon_lonlat,
            point_latlon=(lat, lon),
            point_radius_m=point_radius_m,
            entry_start=start_time.replace(tzinfo=None),
            entry_end=start_time.replace(tzinfo=None),  # deterministic release at start_time
            forecast_time=now_utc.replace(tzinfo=None),
            n_particles=n_particles,
        )
    except Exception as exc:
        log.exception("OpenDrift simulation failed")
        raise HTTPException(status_code=500, detail=f"Simulation failed: {exc}")

    heatmap, bbox = particles_to_heatmap(result["lons"], result["lats"])

    return {
        "status": "success",
        "data_source_used": source,
        "n_particles": result["n_particles"],
        "hours_simulated": round(elapsed_hours, 2),
        "forecast_time": now_utc.isoformat(),
        "origin": {"lat": lat, "lon": lon},
        "polygon": [{"lat": pt[0], "lon": pt[1]} for pt in polygon_points] if polygon_points else [],
        "heatmap": heatmap,
        "bbox": bbox,
    }
