"""
Script pengujian cepat untuk endpoint /predict FastAPI.
Jalankan:
    python src/api/test_api.py
"""

import json
import urllib.request

BASE_URL = "http://127.0.0.1:8001"

print("=" * 60)
print("PENGUJIAN ENDPOINT /predict")
print("=" * 60)

# Test 1: Tanpa input (Otomatis data terbaru)
try:
    with urllib.request.urlopen(f"{BASE_URL}/predict", timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        print(f"\n1. Default (Tanpa Input) - {len(data)} Sensor:")
        print(f"   Daftar Sensor: {list(data.keys())}")
        print("\nContoh Output Plaza Seremoni:")
        print(json.dumps(data.get("Plaza Seremoni", {}), indent=2))
except Exception as e:
    print(f"Error Test 1: {e}")

# Test 2: Dengan input jam origin (contoh: 02:00)
try:
    with urllib.request.urlopen(f"{BASE_URL}/predict?origin=02:00", timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        print(f"\n2. Input origin=02:00 - {len(data)} Sensor:")
        print("\nContoh Output Plaza Seremoni (Origin 02:00):")
        print(json.dumps(data.get("Plaza Seremoni", {}), indent=2))
except Exception as e:
    print(f"Error Test 2: {e}")

print("\n" + "=" * 60)
