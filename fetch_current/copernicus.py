from datetime import datetime
import copernicusmarine
import pandas as pd


class CopernicusFetcher:
    """Fetches ocean current data from Copernicus Marine.

    Returns a DataFrame with columns: time, latitude, longitude, uo (East m/s), vo (North m/s).
    """

    name = "copernicus"

    def __init__(self, dataset_id: str = "cmems_mod_med_phy-cur_anfc_4.2km-2D_PT1H-m", bbox_radius: float = 0.5):
        self.dataset_id = dataset_id
        self.bbox_radius = bbox_radius

    def fetch(self, lat: float, lon: float, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        return copernicusmarine.read_dataframe(
            dataset_id=self.dataset_id,
            variables=["uo", "vo"],
            minimum_longitude=lon - self.bbox_radius,
            maximum_longitude=lon + self.bbox_radius,
            minimum_latitude=lat - self.bbox_radius,
            maximum_latitude=lat + self.bbox_radius,
            start_datetime=start_time.strftime("%Y-%m-%dT%H:%M:%S"),
            end_datetime=end_time.strftime("%Y-%m-%dT%H:%M:%S"),
        ).reset_index()