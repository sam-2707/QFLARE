import time
import requests
from secure_comm import authenticate_with_server
from trainer import train_local_model

DEVICE_ID = "ID_01"

def wait_for_server(url, timeout=60):
    start = time.time()
    while True:
        try:
            r = requests.get(url)
            if r.status_code == 200:
                print("[*] Server is up!")
                return
        except Exception:
            pass
        if time.time() - start > timeout:
            print("[-] Server did not become available in time.")
            exit(1)
        print("[*] Waiting for server...")
        time.sleep(2)

def main():
    print("[*] Starting QFLARE Edge Node...")
    wait_for_server("http://server:8000/")

    # Step 1: Request Quantum Key for model submission
    qkey_resp = requests.post("http://server:8000/request_qkey", json={
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
    response = requests.post("http://server:8000/submit_model", json={
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
