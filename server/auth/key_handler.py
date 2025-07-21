import hashlib

# Mock store (In real world: use DB)
device_keys = {}

def register_device(device_id, key):
    device_keys[device_id] = hashlib.sha256(key.encode()).hexdigest()

def validate_key(device_id, key):
    stored_hash = device_keys.get(device_id)
    return stored_hash == hashlib.sha256(key.encode()).hexdigest()
