"""
Pipeline forecast ISPU — dipakai oleh NB05 dan src/forecast.py.

Alur: data mentah (ispu + ispu_weather) → data per jam per sensor (= NB00)
      → fitur (= NB03) → forecast 6 model RF (= NB04) → konversi ISPU.

Logika cleaning dan fitur di file ini harus sama dengan NB00 dan NB03.
NB05-C03 dan NB05-C04 mengecek kesamaannya dengan file hasil NB00 dan NB03.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


# ============================================================
# KONSTANTA (sama dengan NB00-C01 dan NB03-C01)
# ============================================================

DEVICE_COL = "ispu_device_id"

POLLUTANT_COLS = ["PM25", "PM10", "CO", "O3", "SO2", "NO2"]
ISPU_EXTRA_COLS = ["HEDRO", "AQI"]
WEATHER_COLS = ["TMP", "RAIN", "HUM", "WS", "WD", "PR", "UV", "LIGHT", "NOISE"]
WEATHER_VALUE_COLS = [c for c in WEATHER_COLS if c != "WD"] + ["WD_sin", "WD_cos"]

VALID_RANGES = {
    "PM25": (0, 500),
    "PM10": (0, 500),
    "CO": (0, 45000),
    "O3": (0, 1000),
    "SO2": (0, 1200),
    "NO2": (0, 3000),
    "TMP": (10, 45),
    "HUM": (1, 100),
    "PR": (900, 1100),
    "WS": (0, 30),
    "WD": (0, 360),
    "RAIN": (0, 100),
    "UV": (0, 20),
    "LIGHT": (0, 200000),
    "NOISE": (20, 140),
}
MIN_SLOTS_PER_HOUR = 6

POLLUTANT_LAGS = [1, 2, 3, 6, 12, 24]
POLLUTANT_ROLLING = [3, 6, 24]
PROFILE_HOURS_AGO = [3, 6, 9, 12, 15, 18, 21]
WEATHER_FEATURE_COLS = ["TMP", "HUM", "RAIN", "WS", "WD_sin", "WD_cos"]
BLOCK_HOURS = 3
MIN_HOURS_IN_BLOCK = 2

SENSOR_LOCATIONS = {
    "PH10P0001": "Plaza Seremoni",
    "PH14P0002": "Wonotirto",
    "PH14P0003": "Senipah",
    "PH14P0004": "Muara Jawa Tengah",
    "PH20P0001": "Plaza Bhineka",
}

# Batas konsentrasi ISPU (Permen LHK No. 14/2020), µg/m³.
# HC tidak di-forecast; nilainya diambil dari rata-rata 24 jam terakhir (HEDRO).
ISPU_LEVELS = [0, 50, 100, 200, 300, 500]
ISPU_BREAKPOINTS = {
    "PM10": [0, 50, 150, 350, 420, 500],
    "PM25": [0, 15.5, 55.4, 150.4, 250.4, 500],
    "SO2": [0, 52, 180, 400, 800, 1200],
    "CO": [0, 4000, 8000, 15000, 30000, 45000],
    "O3": [0, 120, 235, 400, 800, 1000],
    "NO2": [0, 80, 200, 1130, 2260, 3000],
    "HC": [0, 45, 100, 215, 432, 648],
}
ISPU_CATEGORIES = [
    (50, "Baik"),
    (100, "Sedang"),
    (200, "Tidak Sehat"),
    (300, "Sangat Tidak Sehat"),
    (np.inf, "Berbahaya"),
]
# Sistem di database menghitung ISPU dari rata-rata 24 jam terakhir (dicek di NB05-C09)
ISPU_WINDOW_HOURS = 24


# ============================================================
# DATA MENTAH → DATA PER JAM (sama dengan NB00)
# ============================================================

def load_raw_tables(ispu_path, weather_path):
    """Baca tabel mentah. Di sistem produksi, ganti fungsi ini dengan query ke database."""
    ispu = pd.read_csv(ispu_path, sep=";", na_values=["NULL"], parse_dates=["DATETIME"])
    weather = pd.read_csv(weather_path, sep=";", na_values=["NULL"], parse_dates=["datetime"])
    ispu = ispu.rename(columns={"DATETIME": "datetime_db"})
    weather = weather.rename(columns={"datetime": "datetime_db"})
    return ispu, weather


def _apply_valid_ranges(df):
    df = df.copy()
    for col, (low, high) in VALID_RANGES.items():
        if col in df.columns:
            invalid = df[col].notna() & ~df[col].between(low, high)
            df.loc[invalid, col] = np.nan
    return df


def _to_5min_slots(df, value_cols):
    df = df.assign(slot_db=df["datetime_db"].dt.floor("5min"))
    return df.groupby([DEVICE_COL, "slot_db"])[value_cols].mean().reset_index()


def build_hourly(ispu_raw, weather_raw):
    """Cleaning, merge, dan agregasi per jam per sensor (sama dengan NB00-C05 s.d. C13)."""
    ispu = ispu_raw[ispu_raw[DEVICE_COL].notna()]
    ispu = ispu[ispu[POLLUTANT_COLS].notna().any(axis=1)]
    ispu = ispu.drop_duplicates([DEVICE_COL, "datetime_db"], keep="first")
    ispu = ispu[[DEVICE_COL, "datetime_db"] + POLLUTANT_COLS + ISPU_EXTRA_COLS]

    weather = weather_raw.drop_duplicates([DEVICE_COL, "datetime_db"], keep="first")
    weather = weather[[DEVICE_COL, "datetime_db"] + WEATHER_COLS]

    ispu = _apply_valid_ranges(ispu)
    weather = _apply_valid_ranges(weather)
    ispu = ispu[ispu[POLLUTANT_COLS].notna().any(axis=1)]

    weather = weather.assign(
        WD_sin=np.sin(np.deg2rad(weather["WD"])),
        WD_cos=np.cos(np.deg2rad(weather["WD"])),
    ).drop(columns="WD")

    ispu_5 = _to_5min_slots(ispu, POLLUTANT_COLS + ISPU_EXTRA_COLS)
    weather_5 = _to_5min_slots(weather, WEATHER_VALUE_COLS)

    merged = ispu_5.merge(
        weather_5, on=[DEVICE_COL, "slot_db"], how="left", validate="one_to_one", indicator=True
    )
    merged["has_weather"] = merged["_merge"] == "both"
    merged = merged.drop(columns="_merge")
    merged["hour_db"] = merged["slot_db"].dt.floor("1h")
    merged["ispu_ok"] = merged[POLLUTANT_COLS].notna().any(axis=1)

    mean_cols = POLLUTANT_COLS + ISPU_EXTRA_COLS + WEATHER_VALUE_COLS
    hourly = (
        merged.groupby([DEVICE_COL, "hour_db"])
        .agg(
            **{col: (col, "mean") for col in mean_cols},
            n_slots_ispu=("ispu_ok", "sum"),
            n_slots_weather=("has_weather", "sum"),
        )
        .reset_index()
    )
    hourly["WD"] = np.rad2deg(np.arctan2(hourly["WD_sin"], hourly["WD_cos"])) % 360
    hourly.loc[hourly["n_slots_ispu"] < MIN_SLOTS_PER_HOUR, POLLUTANT_COLS + ISPU_EXTRA_COLS] = np.nan
    hourly.loc[hourly["n_slots_weather"] < MIN_SLOTS_PER_HOUR, WEATHER_VALUE_COLS + ["WD"]] = np.nan

    parts = []
    for device, d in hourly.groupby(DEVICE_COL):
        d = d.set_index("hour_db").sort_index()
        d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="1h", name="hour_db"))
        d[DEVICE_COL] = device
        d[["n_slots_ispu", "n_slots_weather"]] = (
            d[["n_slots_ispu", "n_slots_weather"]].fillna(0).astype(int)
        )
        parts.append(d.reset_index())

    out_cols = (
        [DEVICE_COL, "hour_db"]
        + POLLUTANT_COLS
        + ISPU_EXTRA_COLS
        + ["TMP", "RAIN", "HUM", "WS", "WD", "WD_sin", "WD_cos", "PR", "UV", "LIGHT", "NOISE"]
        + ["n_slots_ispu", "n_slots_weather"]
    )
    return (
        pd.concat(parts, ignore_index=True)[out_cols]
        .rename(columns={"hour_db": "datetime_db"})
        .sort_values([DEVICE_COL, "datetime_db"])
        .reset_index(drop=True)
    )


# ============================================================
# DATA PER JAM → FITUR (sama dengan NB03)
# ============================================================

def _rolling_mean(grouped, window, min_periods):
    return grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).mean())


def build_features(hourly, config, horizons):
    """Fitur, titik grafik (block3h), dan nilai 'kemarin jam sama' per sensor (sama dengan NB03-C02 s.d. C09)."""
    df = hourly.copy()
    df["hour_wita"] = df["datetime_db"] + pd.Timedelta(hours=config["db_to_wita_hours"])
    df = df[df["hour_wita"] >= pd.Timestamp(config["start_wita"])]
    df = df.sort_values([DEVICE_COL, "hour_wita"]).reset_index(drop=True)

    df["origin_time"] = df["hour_wita"] + pd.Timedelta(hours=1)
    df["is_grid_origin"] = df["origin_time"].dt.hour.isin(config["grid_hours_wita"])

    g = df.groupby(DEVICE_COL)
    new = {}

    for col in POLLUTANT_COLS:
        for lag in POLLUTANT_LAGS:
            new[f"{col}_lag{lag}"] = g[col].shift(lag - 1)
        for window in POLLUTANT_ROLLING:
            new[f"{col}_mean{window}h"] = _rolling_mean(g[col], window, int(np.ceil(window / 2)))

    for col in POLLUTANT_COLS:
        mean3h = new[f"{col}_mean3h"].groupby(df[DEVICE_COL])
        for ago in PROFILE_HOURS_AGO:
            new[f"{col}_mean3h_ago{ago}"] = mean3h.shift(ago)

    for col in WEATHER_FEATURE_COLS:
        new[f"{col}_lag1"] = df[col]
    for col in ["TMP", "HUM", "RAIN", "WS"]:
        new[f"{col}_mean3h"] = _rolling_mean(g[col], 3, 2)
    new["RAIN_mean24h"] = _rolling_mean(g["RAIN"], 24, 12)

    origin_hour = df["origin_time"].dt.hour
    new["hour_sin"] = np.sin(2 * np.pi * origin_hour / 24)
    new["hour_cos"] = np.cos(2 * np.pi * origin_hour / 24)
    new["day_of_week"] = df["origin_time"].dt.dayofweek

    for sensor in SENSOR_LOCATIONS:
        new[f"sensor_{sensor}"] = (df[DEVICE_COL] == sensor).astype(int)

    # Titik grafik (rata-rata 3 jam terakhir) dan nilai "kemarin jam sama" per horizon
    for col in POLLUTANT_COLS:
        block = _rolling_mean(g[col], BLOCK_HOURS, MIN_HOURS_IN_BLOCK)
        new[f"{col}_block3h"] = block
        for h in horizons:
            new[f"{col}_seasonal_{h}h"] = block.groupby(df[DEVICE_COL]).shift(24 - h)

    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


# ============================================================
# MODEL & FORECAST
# ============================================================

def load_models(model_dir):
    model_dir = Path(model_dir)
    with open(model_dir / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    models = {p: joblib.load(model_dir / f"RF_{p}.joblib") for p in metadata["pollutants"]}
    return models, metadata


def latest_grid_origin(feat, metadata):
    """Origin jam grid terbaru yang fiturnya lengkap untuk minimal satu sensor."""
    complete = feat[metadata["feature_cols"]].notna().all(axis=1)
    candidates = feat.loc[feat["is_grid_origin"] & complete, "origin_time"]
    if candidates.empty:
        raise ValueError("Tidak ada origin jam grid dengan fitur lengkap.")
    return candidates.max()


def predict_forecast(feat, origin, models, metadata):
    """Forecast 8 titik (rata-rata 3 jam) untuk setiap sensor dan polutan pada satu origin."""
    features = metadata["feature_cols"]
    horizons = metadata["horizons"]
    rows = feat[feat["origin_time"] == origin].set_index(DEVICE_COL)

    records = []
    skipped = []

    complete = rows[features].notna().all(axis=1)
    for sensor in rows.index[~complete]:
        skipped.append({"sensor": sensor, "alasan": "fitur tidak lengkap (data sensor terputus)"})
    rows = rows[complete]

    for p in metadata["pollutants"]:
        mode = metadata["target_mode"][p]
        seasonal_cols = [f"{p}_seasonal_{h}h" for h in horizons]
        ok = rows.index
        if mode == "koreksi":
            ok = rows.index[rows[seasonal_cols].notna().all(axis=1)]
            for sensor in rows.index.difference(ok):
                skipped.append({"sensor": sensor, "alasan": f"{p}: nilai 'kemarin jam sama' tidak tersedia"})
        if len(ok) == 0:
            continue

        pred = models[p].predict(rows.loc[ok, features])
        if mode == "koreksi":
            pred = pred + rows.loc[ok, seasonal_cols].values
        pred = np.clip(pred, 0, None)   # konsentrasi tidak mungkin negatif

        for i, sensor in enumerate(ok):
            for j, h in enumerate(horizons):
                records.append({
                    "forecast_origin": origin,
                    "forecast_time": origin + pd.Timedelta(hours=h),
                    "horizon_jam": h,
                    "sensor": sensor,
                    "lokasi": SENSOR_LOCATIONS.get(sensor, sensor),
                    "polutan": p,
                    "cara": mode,
                    "konsentrasi_3jam": pred[i, j],
                })

    return pd.DataFrame(records), pd.DataFrame(skipped, columns=["sensor", "alasan"])


# ============================================================
# ISPU
# ============================================================

def ispu_subindex(concentration, parameter):
    bp = ISPU_BREAKPOINTS[parameter]
    return np.interp(np.clip(concentration, 0, bp[-1]), bp, ISPU_LEVELS)


def ispu_category(value):
    if pd.isna(value):
        return None
    value = round(value)
    for upper, name in ISPU_CATEGORIES:
        if value <= upper:
            return name


def historical_ispu(feat):
    """ISPU aktual per jam: rata-rata 24 jam terakhir setiap parameter (cara sistem di database)."""
    g = feat.groupby(DEVICE_COL)
    sub = {}
    for p in POLLUTANT_COLS + ["HEDRO"]:
        param = "HC" if p == "HEDRO" else p
        mean24 = _rolling_mean(g[p], ISPU_WINDOW_HOURS, int(ISPU_WINDOW_HOURS * 0.75))
        sub[f"ISPU_{param}"] = ispu_subindex(mean24, param)
    sub = pd.DataFrame(sub, index=feat.index)
    out = feat[[DEVICE_COL, "hour_wita", "origin_time", "AQI"]].copy()
    out["ISPU"] = sub.max(axis=1)
    has_value = sub.notna().any(axis=1)
    out["polutan_dominan"] = None
    out.loc[has_value, "polutan_dominan"] = (
        sub[has_value].idxmax(axis=1).str.replace("ISPU_", "", regex=False)
    )
    return pd.concat([out, sub], axis=1)


def forecast_ispu(feat, forecast, origin, horizons):
    """
    ISPU forecast per sensor per titik.

    Rata-rata 24 jam yang berakhir di titik forecast = gabungan data aktual
    (jam sebelum origin) dan nilai forecast (jam setelah origin).
    HC tidak di-forecast: bagian setelah origin memakai rata-rata 24 jam terakhir.
    """
    records = []
    for sensor, fc_s in forecast.groupby("sensor"):
        hist = feat[feat[DEVICE_COL] == sensor].set_index("hour_wita")
        last24 = pd.date_range(origin - pd.Timedelta(hours=24), origin - pd.Timedelta(hours=1), freq="1h")
        hc_recent = hist["HEDRO"].reindex(last24).mean()

        for h in horizons:
            target = origin + pd.Timedelta(hours=h)
            actual_hours = pd.date_range(target - pd.Timedelta(hours=24), origin - pd.Timedelta(hours=1), freq="1h")
            n_actual = len(actual_hours)
            sub = {}

            for p in POLLUTANT_COLS:
                blocks = fc_s[(fc_s["polutan"] == p) & (fc_s["horizon_jam"] <= h)]["konsentrasi_3jam"]
                if blocks.empty:
                    continue
                actual = hist[p].reindex(actual_hours)
                if n_actual and actual.notna().sum() < n_actual / 2:
                    continue
                actual_sum = actual.mean() * n_actual if n_actual else 0.0
                mean24 = (actual_sum + BLOCK_HOURS * blocks.sum()) / ISPU_WINDOW_HOURS
                sub[f"ISPU_{p}"] = float(ispu_subindex(mean24, p))

            hc_actual = hist["HEDRO"].reindex(actual_hours)
            if pd.notna(hc_recent) and (not n_actual or hc_actual.notna().sum() >= n_actual / 2):
                hc_sum = hc_actual.mean() * n_actual if n_actual else 0.0
                sub["ISPU_HC"] = float(ispu_subindex((hc_sum + h * hc_recent) / ISPU_WINDOW_HOURS, "HC"))

            if not sub:
                continue
            dominant = max(sub, key=sub.get)
            value = sub[dominant]
            records.append({
                "forecast_origin": origin,
                "forecast_time": target,
                "horizon_jam": h,
                "sensor": sensor,
                "lokasi": SENSOR_LOCATIONS.get(sensor, sensor),
                "ISPU": round(value),
                "kategori": ispu_category(value),
                "polutan_dominan": dominant.replace("ISPU_", ""),
                **{k: round(v, 1) for k, v in sub.items()},
            })
    return pd.DataFrame(records)


# ============================================================
# JALANKAN & SIMPAN
# ============================================================

def run_forecast(root, origin=None):
    """Jalankan seluruh pipeline dari data mentah sampai forecast ISPU."""
    root = Path(root)
    with open(root / "data" / "processed" / "model_config.json", encoding="utf-8") as f:
        config = json.load(f)
    models, metadata = load_models(root / "models" / "rf_multioutput")

    ispu_raw, weather_raw = load_raw_tables(root / "data" / "raw" / "ispu.csv", root / "data" / "raw" / "ispu_weather.csv")
    hourly = build_hourly(ispu_raw, weather_raw)
    feat = build_features(hourly, config, metadata["horizons"])

    if origin is None:
        origin = latest_grid_origin(feat, metadata)
    else:
        origin = pd.Timestamp(origin)
        if origin.hour not in metadata["grid_hours_wita"] or origin.minute != 0:
            raise ValueError(f"Origin harus tepat di jam grid {metadata['grid_hours_wita']} WITA.")

    forecast, skipped = predict_forecast(feat, origin, models, metadata)
    ispu = forecast_ispu(feat, forecast, origin, metadata["horizons"])

    return {
        "origin": origin,
        "forecast": forecast,
        "ispu": ispu,
        "skipped": skipped,
        "feat": feat,
        "hourly": hourly,
        "metadata": metadata,
        "config": config,
    }


def _append_history(new, path, keys):
    if path.exists():
        old = pd.read_csv(path, parse_dates=["forecast_origin", "forecast_time"])
        new = pd.concat([old, new], ignore_index=True)
    new = new.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)
    new.to_csv(path, index=False)
    return len(new)


def save_outputs(result, out_dir):
    """Simpan forecast terbaru (ditimpa) dan riwayat forecast (ditambah, tanpa duplikat)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "forecast_latest": out_dir / "forecast_latest.csv",
        "ispu_latest": out_dir / "forecast_ispu_latest.csv",
        "forecast_history": out_dir / "forecast_history.csv",
        "ispu_history": out_dir / "forecast_ispu_history.csv",
    }
    result["forecast"].to_csv(paths["forecast_latest"], index=False)
    result["ispu"].to_csv(paths["ispu_latest"], index=False)

    n_fc = _append_history(result["forecast"], paths["forecast_history"],
                           ["forecast_origin", "sensor", "polutan", "horizon_jam"])
    n_ispu = _append_history(result["ispu"], paths["ispu_history"],
                             ["forecast_origin", "sensor", "horizon_jam"])
    return paths, n_fc, n_ispu
