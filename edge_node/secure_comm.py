from requests import post

# secure_comm.py
# Handles authentication with the server using a simulated quantum-safe key.
def authenticate_with_server(device_id, key):
    print("[*] Authenticating with server...")
    response = post("http://server:8000/validate", json={
        "device_id": device_id,
        "key": key
    })

    if response.status_code == 200:
        print("[+] Authenticated with central server.")
        return True
    else:
        print("[-] Authentication failed:", response.text)
        return False
