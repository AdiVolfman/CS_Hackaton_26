# SafeCurrent

SafeCurrent is a search-and-rescue drift simulation prototype for the Israeli coastline.

The app has:

- A FastAPI backend in `main.py`
- A browser frontend in `index.html`
- Copernicus Marine data support with an Open-Meteo fallback

## Requirements

- Python 3.10 or newer
- A Copernicus Marine account
- Internet access for map tiles and marine data

## Setup

Clone the repository:

```powershell
git clone https://github.com/AdiVolfman/CS_Hackaton_26.git
cd CS_Hackaton_26
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Log in to Copernicus Marine:

```powershell
copernicusmarine login
```

## Run

Start the backend:

```powershell
python -m uvicorn main:app --reload --port 8000
```

Keep that terminal open.

Then open `index.html` in a browser.

## Use

1. Click a point in the sea on the map.
2. Adjust `Hours Since Drowning` if needed.
3. Click `Recalculate Current Points`.
4. The map shows the predicted drift path and search area.

The frontend calls the backend at:

```text
http://127.0.0.1:8000/simulate
```

You can test the API directly with:

```text
http://127.0.0.1:8000/simulate?lat=32.0800&lon=34.7650&hours=4
```

More detailed setup notes are in `SETUP_TUTORIAL.txt`.
