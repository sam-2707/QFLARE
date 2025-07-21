from edge_node.secure_comm import authenticate_with_server
from edge_node.trainer import train_local_model
import time
import requests

DEVICE_ID = "edge-node-01"


def main():
    print("[*] Starting QFLARE Edge Node...")

    # Step 1: Request Quantum Key for model submission
    qkey_resp = requests.post("http://localhost:8000/request_qkey", json={
        "device_id": DEVICE_ID,
        "action": "submit_model"
    })
    if qkey_resp.status_code != 200:
        print("[-] Failed to obtain quantum key:", qkey_resp.text)
        return
    session_key = qkey_resp.json()["qkey"]

    # Step 2: Authenticate with Central Server (optional, legacy)
    # auth = authenticate_with_server(DEVICE_ID, session_key)
    # if not auth:
    #     print("[-] Authentication Failed. Exiting.")
    #     return

    # Step 3: Perform Local Training
    model_weights = train_local_model()

    # Step 4: Send model to server with quantum key
    response = requests.post("http://localhost:8000/submit_model", json={
        "device_id": DEVICE_ID,
        "qkey": session_key,
        "weights": model_weights
    })

    if response.status_code == 200:
        print("[+] Model update sent successfully!")
    else:
        print("[-] Failed to send model:", response.text)

if __name__ == "__main__":
    main()
