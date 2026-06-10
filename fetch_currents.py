import copernicusmarine
import datetime

# Define the time range in UTC (Copernicus models use UTC time)
start_time = datetime.datetime.now(datetime.timezone.utc)
end_time = start_time + datetime.timedelta(hours=24)

print("--- Starting Copernicus Data Fetch for Israel Coastline ---")
print(f"Start Time (UTC): {start_time.isoformat()}")
print(f"End Time (UTC): {end_time.isoformat()}")

try:
    # Fetching data from Copernicus within Israel's geographic boundaries
    # Bounding box: Lat 31.5 to 33.5, Lon 34.0 to 35.0 (Covers the Mediterranean coast)
    ds = copernicusmarine.read(
        dataset_id="med-cmcc-phys-an-fc-h",  # Mediterranean Sea Hourly Physics Model
        variables=["uo", "vo"],
        longitude_min=34.0,
        longitude_max=35.0,
        latitude_min=31.5,
        latitude_max=33.5,
        start_date=start_time.strftime("%Y-%m-%dT%H:%M:%S"),
        end_date=end_time.strftime("%Y-%m-%dT%H:%M:%S")
    )

    # Convert the NetCDF dataset into a clean Pandas DataFrame
    df = ds.to_dataframe().reset_index()

    print("\n--- Data Successfully Retrieved! ---")
    print(df.head())  # Displays the first 5 rows of the dataframe

except Exception as e:
    print(f"\n❌ Error fetching data: {e}")
    print("Tip: Make sure you ran 'copernicusmarine login' in your terminal first.")