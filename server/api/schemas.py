from pydantic import BaseModel

class AuthRequest(BaseModel):
    device_id: str
    qkey: str  # Simulated quantum key from edge

class AuthResponse(BaseModel):
    status: str
    device_id: str
