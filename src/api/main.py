"""
API Forecast ISPU - Inferensi Model Training Machine Learning
Hanya 1 endpoint (/predict) dengan 1 input parameter (origin / jam).
Output: Tabel matriks prediksi (Target Jam x Polutan) untuk seluruh lokasi sensor.
"""

from enum import Enum
from pathlib import Path
import sys
import json
from typing import Optional, Any, Dict

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# 1. Setup Path ke Root Project & Folder src
ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pipeline as pl


# 2. Pilihan Dropdown Jam ORIGIN (Acuan t=0)
class JamOrigin(str, Enum):
    JAM_02 = "02:00"
    JAM_05 = "05:00"
    JAM_08 = "08:00"
    JAM_11 = "11:00"
    JAM_14 = "14:00"
    JAM_17 = "17:00"
    JAM_20 = "20:00"
    JAM_23 = "23:00"


# 3. Inisialisasi FastAPI
app = FastAPI(
    title="Forecast ISPU API",
    description="API Inferensi Machine Learning untuk peramalan ISPU di seluruh sensor.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory cache data & model
_CACHE: Dict[str, Any] = {}


def get_pipeline_resources():
    """Memuat model dan fitur ke memori agar respon API instan (< 50ms)."""
    if "feat" not in _CACHE:
        with open(ROOT / "data" / "processed" / "model_config.json", encoding="utf-8") as f:
            config = json.load(f)
        models, metadata = pl.load_models(ROOT / "models" / "rf_multioutput")
        ispu_raw, weather_raw = pl.load_raw_tables(
            ROOT / "data" / "raw" / "ispu.csv",
            ROOT / "data" / "raw" / "ispu_weather.csv"
        )
        hourly = pl.build_hourly(ispu_raw, weather_raw)
        feat = pl.build_features(hourly, config, metadata["horizons"])

        _CACHE["config"] = config
        _CACHE["models"] = models
        _CACHE["metadata"] = metadata
        _CACHE["feat"] = feat

    return _CACHE["models"], _CACHE["metadata"], _CACHE["feat"]


def clean_num(val: Any) -> Optional[float]:
    """Membulatkan angka ke 2 desimal atau None jika NaN/Inf."""
    if val is None or pd.isna(val) or np.isnan(val) or np.isinf(val):
        return None
    return round(float(val), 2)


# 4. SATU-SATUNYA ENDPOINT: /predict
@app.get("/predict", summary="Forecast ISPU Semua Sensor")
@app.get("/", include_in_schema=False)
def predict(
    origin: Optional[JamOrigin] = Query(
        None,
        description="Pilih jam acuan ORIGIN (02:00, 05:00, 08:00, 11:00, 14:00, 17:00, 20:00, 23:00). Kosongkan untuk memakai waktu data sensor terbaru."
    )
):
    """
    Menjalankan inferensi model:
        forecast, skipped = pl.predict_forecast(feat, ORIGIN, models, metadata)

    Mengembalikan tabel matriks (Target Jam x Polutan) untuk seluruh sensor.
    """
    try:
        models, metadata, feat = get_pipeline_resources()

        # Menentukan waktu ORIGIN acuan
        origin_val = origin.value if isinstance(origin, JamOrigin) else (str(origin) if origin else None)

        if origin_val:
            target_hour = int(origin_val.split(":")[0])
            complete = feat[metadata["feature_cols"]].notna().all(axis=1)
            candidates = feat.loc[feat["is_grid_origin"] & complete, "origin_time"]
            matches = candidates[candidates.dt.hour == target_hour]
            if matches.empty:
                raise HTTPException(
                    status_code=400,
                    detail=f"Tidak ada data lengkap untuk origin jam {origin_val}."
                )
            selected_origin = matches.max()
        else:
            selected_origin = pl.latest_grid_origin(feat, metadata)

        # Inferensi model Machine Learning
        forecast, skipped = pl.predict_forecast(feat, selected_origin, models, metadata)

        # Bentuk tabel matriks per sensor: Waktu Target x Polutan
        tabel_per_sensor = {}
        for sensor_id, fc_s in forecast.groupby("sensor"):
            lokasi = pl.SENSOR_LOCATIONS.get(sensor_id, sensor_id)
            fc_sorted = fc_s.sort_values("horizon_jam")

            matriks_sensor = {}
            for t, grp in fc_sorted.groupby("forecast_time", sort=False):
                key_waktu = t.strftime("%H.%M")
                grp_reindexed = grp.set_index("polutan").reindex(pl.POLLUTANT_COLS)
                matriks_sensor[key_waktu] = {
                    p: clean_num(v)
                    for p, v in zip(grp_reindexed.index, grp_reindexed["konsentrasi_3jam"])
                }
            tabel_per_sensor[lokasi] = matriks_sensor

        return tabel_per_sensor

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal mengeksekusi model forecast: {type(exc).__name__}: {exc}"
        ) from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)
