from fastapi import APIRouter, HTTPException, Request
from server.auth.key_handler import validate_key
from server.fl_core.aggregator import aggregate_model
from server.auth.pqcrypto_utils import generate_quantum_key
from server.ledger.audit_logger import log_audit_event
from server.registry import registered_devices
import time

router = APIRouter()

# In-memory one-time quantum key store: {qkey: {device_id, action, expiry, used}}
one_time_qkey_store = {}
QKEY_EXPIRY_SECONDS = 60  # Keys valid for 60 seconds

def store_one_time_qkey(device_id, action, qkey):
    one_time_qkey_store[qkey] = {
        "device_id": device_id,
        "action": action,
        "expiry": time.time() + QKEY_EXPIRY_SECONDS,
        "used": False
    }

def validate_one_time_qkey(device_id, action, qkey):
    entry = one_time_qkey_store.get(qkey)
    if not entry:
        return False
    if entry["used"] or entry["device_id"] != device_id or entry["action"] != action:
        return False
    if time.time() > entry["expiry"]:
        return False
    # Mark as used
    entry["used"] = True
    return True

@router.post("/request_qkey")
async def request_qkey(request: Request):
    payload = await request.json()
    device_id = payload.get("device_id")
    action = payload.get("action")
    if not device_id or not action:
        raise HTTPException(status_code=400, detail="Missing device_id or action")
    if device_id not in registered_devices:
        raise HTTPException(status_code=403, detail="Device not registered")
    qkey = generate_quantum_key(f"{device_id}:{action}:{time.time()}")
    store_one_time_qkey(device_id, action, qkey)
    log_audit_event({
        "event": "qkey_issued",
        "device_id": device_id,
        "action": action,
        "qkey": qkey,
        "timestamp": time.time()
    })
    return {"qkey": qkey, "expires_in": QKEY_EXPIRY_SECONDS}

@router.post("/validate")
async def validate_device(request: Request):
    payload = await request.json()
    device_id = payload.get("device_id")
    key = payload.get("key")
    
    if validate_key(device_id, key):
        return {"status": "granted"}
    raise HTTPException(status_code=403, detail="Invalid Key")

@router.post("/submit_model")
async def submit_model(request: Request):
    payload = await request.json()
    model_update = payload.get("weights")
    device_id = payload.get("device_id")
    qkey = payload.get("qkey")
    if device_id not in registered_devices:
        raise HTTPException(status_code=403, detail="Device not registered")
    if not validate_one_time_qkey(device_id, "submit_model", qkey):
        log_audit_event({
            "event": "qkey_invalid",
            "device_id": device_id,
            "action": "submit_model",
            "qkey": qkey,
            "timestamp": time.time()
        })
        raise HTTPException(status_code=403, detail="Invalid or expired quantum key")
    aggregate_model(device_id, model_update)
    log_audit_event({
        "event": "qkey_used",
        "device_id": device_id,
        "action": "submit_model",
        "qkey": qkey,
        "timestamp": time.time()
    })
    return {"status": "model received"}
