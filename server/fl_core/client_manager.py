import time

# In-memory registry of connected edge clients
registered_clients = {}

def register_client(device_id: str):
    if device_id not in registered_clients:
        registered_clients[device_id] = {
            "status": "active",
            "last_seen": time.time(),
            "tasks_completed": 0
        }

def update_client_activity(device_id: str):
    if device_id in registered_clients:
        registered_clients[device_id]["last_seen"] = time.time()

def mark_task_complete(device_id: str):
    if device_id in registered_clients:
        registered_clients[device_id]["tasks_completed"] += 1

def get_client_status(device_id: str):
    return registered_clients.get(device_id)

def list_active_clients():
    return [cid for cid, data in registered_clients.items() if data["status"] == "active"]
