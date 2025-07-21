import numpy as np
import pickle

# Aggregates model weights from clients using FedAvg
def fed_avg(client_weights: list[bytes]) -> bytes:
    deserialized = [pickle.loads(w) for w in client_weights]
    avg_weights = np.mean(deserialized, axis=0)
    return pickle.dumps(avg_weights)

# Weighted average (optional enhancement)
def weighted_avg(client_weights: list[bytes], weights: list[float]) -> bytes:
    deserialized = [pickle.loads(w) for w in client_weights]
    weighted = np.average(deserialized, axis=0, weights=weights)
    return pickle.dumps(weighted)

# Save aggregated model
def save_aggregated_model(model_bytes: bytes, path: str = "models/global_model.pkl"):
    with open(path, "wb") as f:
        f.write(model_bytes)
