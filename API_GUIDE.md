# Panduan API Forecast ISPU (Inferensi Model Training)

API ini hanya memiliki **1 endpoint** dan **1 input parameter** (`origin` jam acuan t=0):

```
Endpoint: GET /predict
Parameter: origin (Pilihan: 02:00, 05:00, 08:00, 11:00, 14:00, 17:00, 20:00, 23:00)
```

---

## 1. Menjalankan Server API

Jalankan perintah ini di PowerShell (port 8001):
```powershell
uvicorn src.api.main:app --port 8001 --reload
```

Dokumentasi Swagger UI:  
👉 **[http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)**

---

## 2. Cara Pakai di Swagger UI

1. Buka endpoint **`GET /predict`**.
2. Klik tombol **Try it out**.
3. Di kolom **`origin`**:
   - **Bisa dikosongkan:** Server otomatis memakai jam data sensor terbaru.
   - **Atau pilih salah satu jam:** Misal `08:00`, `02:00`, dll.
4. Klik **Execute**.

---

## 3. Hasil Output

Langsung berupa tabel matriks (Target Jam × 6 Polutan) untuk seluruh sensor:

```json
{
  "Plaza Seremoni": {
    "11.00": { "PM25": 38.37, "PM10": 69.75, "CO": 584.89, "O3": 3.14, "SO2": 5.30, "NO2": 41.53 },
    "14.00": { "PM25": 32.72, "PM10": 66.93, "CO": 511.39, "O3": 7.99, "SO2": 4.83, "NO2": 45.07 },
    "17.00": { "PM25": 27.14, "PM10": 51.47, "CO": 503.58, "O3": 13.68, "SO2": 4.75, "NO2": 49.38 },
    "20.00": { "PM25": 16.12, "PM10": 30.39, "CO": 494.05, "O3": 11.53, "SO2": 5.59, "NO2": 56.48 },
    "23.00": { "PM25": 17.27, "PM10": 29.39, "CO": 489.28, "O3": 4.24, "SO2": 6.09, "NO2": 61.79 },
    "02.00": { "PM25": 18.29, "PM10": 33.28, "CO": 440.58, "O3": 2.80, "SO2": 5.89, "NO2": 51.79 },
    "05.00": { "PM25": 41.95, "PM10": 71.83, "CO": 708.22, "O3": 2.77, "SO2": 6.42, "NO2": 44.33 },
    "08.00": { "PM25": 41.97, "PM10": 75.79, "CO": 758.70, "O3": 2.58, "SO2": 6.79, "NO2": 41.38 }
  },
  "Wonotirto": { ... },
  "Senipah": { ... },
  "Muara Jawa Tengah": { ... },
  "Plaza Bhineka": { ... }
}
```
