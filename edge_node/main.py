from edge_node.secure_comm import authenticate_with_server
from edge_node.trainer import train_local_model
import time
from server.auth.pqcrypto_utils import generate_quantum_key

DEVICE_ID = "edge-node-01"

def main():
    print("[*] Starting QFLARE Edge Node...")

    # Step 1: Generate Quantum Key (simulated)
    session_key = generate_quantum_key(DEVICE_ID)

    # Step 2: Authenticate with Central Server
    auth = authenticate_with_server(DEVICE_ID, session_key)
    if not auth:
        print("[-] Authentication Failed. Exiting.")
        return

    # Step 3: Perform Local Training
    model_weights = train_local_model()

    # Step 4: Send model to server
    from requests import post
    response = post("http://localhost:8000/submit_model", json={
        "device_id": DEVICE_ID,
        "key": session_key,
        "weights": model_weights
    })

    if response.status_code == 200:
        print("[+] Model update sent successfully!")
    else:
        print("[-] Failed to send model:", response.text)

if __name__ == "__main__":
    main()
