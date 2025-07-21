from fastapi import APIRouter, HTTPException, Request
from server.auth.key_handler import validate_key
from server.fl_core.aggregator import aggregate_model

router = APIRouter()

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

    if not validate_key(device_id, payload.get("key")):
        raise HTTPException(status_code=403, detail="Key mismatch")

    aggregate_model(device_id, model_update)
    return {"status": "model received"}
