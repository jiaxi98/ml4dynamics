import pickle
import os
import numpy as np

aposteriori_path = "results/aposteriori_metrics.pkl"

if not os.path.exists(aposteriori_path):
    print(f"File not found: {aposteriori_path}")
    exit(1)

with open(aposteriori_path, "rb") as f:
    aposteriori_metrics = pickle.load(f)

print(f"Found {len(aposteriori_metrics)} parameter sweeps in {aposteriori_path}.\n")

for key in aposteriori_metrics:
    print(f"=== {key} ===")
    metrics = aposteriori_metrics[key]
    for metric_name, value in metrics.items():
        arr = None
        if isinstance(value, (list, tuple)):
            try:
                arr = np.array(value)
                print(f"  {metric_name}: type={type(value).__name__}, shape={arr.shape}")
            except Exception as e:
                print(f"  {metric_name}: type={type(value).__name__}, could not get shape ({e})")
        elif hasattr(value, 'shape'):
            print(f"  {metric_name}: type={type(value).__name__}, shape={value.shape}")
        else:
            print(f"  {metric_name}: type={type(value).__name__}, value={value}")
    print()
