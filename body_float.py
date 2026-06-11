import math
import requests

OPEN_METEO_MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
OPENTOPODATA_GEBCO_URL = "https://api.opentopodata.org/v1/gebco2020"

BODY_DENSITY_BY_TYPE = {
    "lean": 1055,
    "average": 1045,
    "fat": 1030,
}

# Decomposition kinetics — temperature-driven gas production.
# Calibrated so an "average" body in 5 m of salt water refloats in:
#   ~2 days at 25°C, ~5 days at 18°C, ~1 week at 10°C, never below 4°C.
# Q10 ≈ 2.5 captures the standard biological doubling-per-10°C trend.
DECOMP_K_REF = 0.0061     # 1/h at the reference temperature
DECOMP_T_REF = 25.0       # °C
DECOMP_Q10 = 2.5
DECOMP_T_FLOOR = 4.0      # below this, decomposition effectively halts

# Putrefactive gas can inflate the abdomen/chest by ~10–15% of body volume.
MAX_GAS_VOLUME_FRACTION = 0.12


def fetch_water_temperature(lat, lon):
    """Sea-surface temperature (°C) from Open-Meteo Marine."""
    response = requests.get(
        OPEN_METEO_MARINE_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "sea_surface_temperature",
        },
        timeout=15,
    )
    response.raise_for_status()
    return float(response.json()["current"]["sea_surface_temperature"])


def fetch_water_depth(lat, lon):
    """Water depth in meters at lat/lon from the GEBCO 2020 bathymetry grid."""
    response = requests.get(
        OPENTOPODATA_GEBCO_URL,
        params={"locations": f"{lat},{lon}"},
        timeout=15,
    )
    response.raise_for_status()
    elevation = response.json()["results"][0]["elevation"]
    if elevation is None:
        raise ValueError(f"No bathymetry data for ({lat}, {lon})")
    if elevation > 0:
        raise ValueError(f"Point ({lat}, {lon}) is on land (elevation {elevation} m)")
    return -float(elevation)


def check_buoyancy_pure(water_temperature, water_type, water_depth_meters, current_time_hours, body_type="average"):
    if body_type not in BODY_DENSITY_BY_TYPE:
        raise ValueError(f"body_type must be one of {sorted(BODY_DENSITY_BY_TYPE)}")
    body_baseline_density = BODY_DENSITY_BY_TYPE[body_type]

    # Water density: salt water is denser than fresh.
    rho_water = 1025 if water_type == "salt" else 1000

    # Temperature-dependent decomposition rate (Q10 model, halts in cold water).
    if water_temperature <= DECOMP_T_FLOOR:
        k = 0.0
    else:
        k = DECOMP_K_REF * DECOMP_Q10 ** ((water_temperature - DECOMP_T_REF) / 10.0)

    # Cumulative gas volume produced (fraction of body volume), saturating at MAX_GAS_VOLUME_FRACTION.
    v_gas_ratio = MAX_GAS_VOLUME_FRACTION * (1 - math.exp(-k * current_time_hours))

    # Boyle's law: ambient pressure compresses the gas (1 atm at surface + 1 atm per 10 m).
    pressure_atm = 1.0 + max(water_depth_meters, 0.0) / 10.0
    v_gas_ratio_compressed = v_gas_ratio / pressure_atm

    # Effective body density after bloating.
    body_current_density = body_baseline_density / (1 + v_gas_ratio_compressed)

    # Archimedes: floats when surrounding water is denser than the body.
    return rho_water > body_current_density


def check_buoyancy_at_location(lat, lon, current_time_hours, water_type="salt", body_type="average"):
    water_temperature = fetch_water_temperature(lat, lon)
    water_depth_meters = fetch_water_depth(lat, lon)
    floating = check_buoyancy_pure(water_temperature, water_type, water_depth_meters, current_time_hours, body_type)
    return {
        "lat": lat,
        "lon": lon,
        "water_temperature": water_temperature,
        "water_depth_meters": water_depth_meters,
        "water_type": water_type,
        "body_type": body_type,
        "current_time_hours": current_time_hours,
        "floating": floating,
    }

# how to use: check_buoyancy_at_location(lat=32.08, lon=34.76, current_time_hours=50, body_type="average")