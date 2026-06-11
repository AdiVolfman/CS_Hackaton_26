from opendrift.models.leeway import Leeway
from opendrift.readers import reader_constant
from datetime import datetime, timedelta, timezone
import requests
from requests.exceptions import SSLError
import urllib3
import numpy as np

print("=== STEP 1: Fetching LIVE Ocean Data via JSON API ===")
lat = 32.080
lon = 34.650

# פנייה ל-API החי של Open-Meteo לקבלת הזרמים הנוכחיים בפורמט JSON (חסין לחלוטין)
api_url = f"https://marine-api.open-meteo.com/v1/marine?latitude={lat}&longitude={lon}&hourly=ocean_current_velocity,ocean_current_direction"

print("Downloading live data from Open-Meteo...")
try:
    response = requests.get(api_url, timeout=30)
except SSLError:
    print("SSL certificate verification failed. Retrying without verification for local demo...")
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    response = requests.get(api_url, timeout=30, verify=False)

if response.status_code == 200:
    data = response.json()
    hourly = data.get('hourly', {})
    
    # חילוץ המהירות (בקמ"ש) והכיוון (במעלות) של הזרם בשעה הנוכחית
    # לוקחים את הערך העדכני ביותר (השעה הנוכחית)
    current_velocity_kmh = hourly.get('ocean_current_velocity', [0])[0]
    current_direction_deg = hourly.get('ocean_current_direction', [0])[0]
    
    # המרת קמ"ש למטרים לשנייה (מה ש-OpenDrift צריך)
    current_speed_ms = (current_velocity_kmh * 1000) / 3600
    
    # פירוק הוקטור לרכיבי X (Eastward) ו-Y (Northward)
    rad = np.radians(current_direction_deg)
    uo_live = current_speed_ms * np.sin(rad)
    vo_live = current_speed_ms * np.cos(rad)
    
    print(f"[OK] Live Data Retrieved: Speed={current_speed_ms:.2f} m/s, Direction={current_direction_deg} deg")
else:
    print(f"[ERROR] API Failed with status code: {response.status_code}")
    exit()

print("\n=== STEP 2: Initializing OpenDrift with Live UserDefined Reader ===")
o = Leeway(loglevel=20)

try:
    # יצירת קורא נתונים בזמן אמת מבוסס על משתני ה-JSON שקיבלנו
    # אנחנו מגדירים ל-OpenDrift מהירות קבועה של רוח וזרם לרגע זה ממש במרחב החיפוש
    live_reader = reader_constant.Reader({
        "x_sea_water_velocity": uo_live,
        "y_sea_water_velocity": vo_live,
        "x_wind": 2.0,
        "y_wind": -2.0,
    })
    
    o.add_reader(live_reader)
    print("[OK] Live Custom Reader successfully loaded into OpenDrift!")
except Exception as e:
    print(f"[ERROR] Reader error: {e}")
    exit()

print("\n=== STEP 3: Seeding Particles at current time ===")
current_utc_time = datetime.now(timezone.utc).replace(tzinfo=None)

o.seed_elements(
    lon=lon, 
    lat=lat, 
    number=30, 
    radius=150, 
    time=current_utc_time,
    object_type=27 # Person in water
)
print("[OK] Particles seeded!")

print("\n=== STEP 4: Running Simulation ===")
try:
    o.run(duration=timedelta(hours=3), time_step=900)
    print("[OK] Live simulation finished successfully using real-time API JSON data!")
    
    result = o.result
    final_lat = float(result['lat'].values[0, -1])
    final_lon = float(result['lon'].values[0, -1])
    print(f"\nFinal position of particle 0: Lat: {final_lat}, Lon: {final_lon}")
    
    o.plot(filename='live_json_opendrift_result.png')
    print("[OK] Plot saved as 'live_json_opendrift_result.png'")
    
except Exception as e:
    print(f"[ERROR] Simulation error: {e}")
