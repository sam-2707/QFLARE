# Simulated quantum key store ledger
# In production, replace with a secure, persistent DB (e.g., PostgreSQL, Redis)

key_store = {}  # device_id -> quantum key

def store_key(device_id: str, qkey: str):
    key_store[device_id] = qkey

def get_key(device_id: str):
    return key_store.get(device_id)

def revoke_key(device_id: str):
    if device_id in key_store:
        del key_store[device_id]
