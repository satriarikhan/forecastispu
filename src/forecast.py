"""
Forecast ISPU 24 jam ke depan (8 titik, setiap 3 jam) untuk semua sensor.

Dijadwalkan setiap 3 jam, 10 menit setelah jam grid (02:10, 05:10, ..., 23:10 WITA),
agar data jam sebelumnya sudah lengkap masuk ke database.

Contoh:
    .venv\\Scripts\\python.exe src\\forecast.py
    .venv\\Scripts\\python.exe src\\forecast.py --origin "2026-09-21 08:00"

Output (folder outputs/forecast/):
    forecast_latest.csv          forecast terbaru, per sensor x polutan x horizon
    forecast_ispu_latest.csv     ISPU forecast terbaru, per sensor x horizon
    forecast_history.csv         riwayat semua forecast (untuk monitoring akurasi)
    forecast_ispu_history.csv    riwayat semua ISPU forecast
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pipeline as pl  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Forecast ISPU 24 jam ke depan untuk semua sensor.")
    parser.add_argument(
        "--origin",
        help='Waktu origin (WITA), contoh "2026-09-21 08:00". Default: origin jam grid terbaru.',
    )
    args = parser.parse_args()

    result = pl.run_forecast(ROOT, origin=args.origin)
    paths, n_fc, n_ispu = pl.save_outputs(result, ROOT / "outputs" / "forecast")

    print(f"Origin forecast : {result['origin']} WITA")
    print(f"Sensor          : {result['forecast']['sensor'].nunique()} sensor, "
          f"{len(result['forecast'])} nilai forecast")
    if not result["skipped"].empty:
        print("Dilewati        :")
        print(result["skipped"].to_string(index=False))

    print("\nISPU forecast:")
    table = result["ispu"].pivot_table(index="lokasi", columns="forecast_time", values="ISPU")
    table.columns = [t.strftime("%d/%m %H:%M") for t in table.columns]
    print(table.to_string())

    print(f"\nDisimpan ke     : {paths['forecast_latest'].parent}")
    print(f"Riwayat         : {n_fc} baris forecast, {n_ispu} baris ISPU")


if __name__ == "__main__":
    main()
