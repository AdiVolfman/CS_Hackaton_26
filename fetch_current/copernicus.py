from datetime import datetime

import copernicusmarine
import pandas as pd

from body_float import fetch_water_depth

DATASET_2D_HOURLY = "cmems_mod_med_phy-cur_anfc_4.2km-2D_PT1H-m"
DATASET_3D_HOURLY = "cmems_mod_med_phy-cur_anfc_4.2km-3D_PT1H-m"


class CopernicusFetcher:
    """Fetches ocean current data from Copernicus Marine.

    Returns a DataFrame with columns: time, latitude, longitude, uo, vo
    (and `depth` when fetched from the 3D dataset).
    """

    name = "copernicus"

    def __init__(self, bbox_radius: float = 0.5, depth_window: float = 2.0):
        self.bbox_radius = bbox_radius
        self.depth_window = depth_window

    def fetch(
        self,
        lat: float,
        lon: float,
        start_time: datetime,
        end_time: datetime,
        floating: bool,
    ) -> pd.DataFrame:
        if floating:
            dataset_id = DATASET_2D_HOURLY
            depth = None
        else:
            dataset_id = DATASET_3D_HOURLY
            depth = fetch_water_depth(lat, lon)
            print(f"Fetching seabed currents at depth ≈ {depth:.1f} m...")

        kwargs = dict(
            dataset_id=dataset_id,
            variables=["uo", "vo"],
            minimum_longitude=lon - self.bbox_radius,
            maximum_longitude=lon + self.bbox_radius,
            minimum_latitude=lat - self.bbox_radius,
            maximum_latitude=lat + self.bbox_radius,
            start_datetime=start_time.strftime("%Y-%m-%dT%H:%M:%S"),
            end_datetime=end_time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        if depth is not None:
            kwargs["minimum_depth"] = max(0.0, depth - self.depth_window)
            kwargs["maximum_depth"] = depth + self.depth_window

        df = copernicusmarine.read_dataframe(**kwargs).reset_index()

        if depth is not None and "depth" in df.columns and not df.empty:
            available = df["depth"].unique()
            closest = available[(abs(available - depth)).argmin()]
            df = df[df["depth"] == closest].reset_index(drop=True)

        return df

