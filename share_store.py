"""Share snapshot store — POST /share saves a snapshot, GET /share/{id} retrieves it.

Snapshots are persisted to a JSON file next to main.py. The store is capped
at 200 entries (oldest pruned by created_at).
"""

import datetime
import json
import pathlib
import secrets
import threading

from fastapi import APIRouter, Body, HTTPException


SHARE_STORE_PATH = pathlib.Path(__file__).resolve().parent / ".shared_snapshots.json"
SHARE_STORE_MAX_ENTRIES = 200

share_store_lock = threading.Lock()
router = APIRouter()


def load_share_store():
    if not SHARE_STORE_PATH.exists():
        return {}

    try:
        with SHARE_STORE_PATH.open("r", encoding="utf-8") as store_file:
            data = json.load(store_file)
    except (OSError, json.JSONDecodeError):
        return {}

    return data if isinstance(data, dict) else {}


def save_share_store(store):
    with SHARE_STORE_PATH.open("w", encoding="utf-8") as store_file:
        json.dump(store, store_file, separators=(",", ":"))


@router.post("/share")
def create_share_snapshot(snapshot: dict = Body(...)):
    if not isinstance(snapshot, dict):
        raise HTTPException(status_code=400, detail="Share snapshot must be an object.")

    snapshot_id = secrets.token_urlsafe(8)
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    with share_store_lock:
        store = load_share_store()
        store[snapshot_id] = {
            "created_at": created_at,
            "snapshot": snapshot,
        }

        if len(store) > SHARE_STORE_MAX_ENTRIES:
            oldest_ids = sorted(
                store,
                key=lambda key: store[key].get("created_at", "")
            )[:-SHARE_STORE_MAX_ENTRIES]
            for old_id in oldest_ids:
                store.pop(old_id, None)

        save_share_store(store)

    return {"share_id": snapshot_id}


@router.get("/share/{snapshot_id}")
def get_share_snapshot(snapshot_id: str):
    with share_store_lock:
        store = load_share_store()

    record = store.get(snapshot_id)
    if not record:
        raise HTTPException(status_code=404, detail="Shared result not found.")

    return record.get("snapshot", {})
