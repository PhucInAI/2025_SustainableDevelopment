"""
Hòn Mun Environmental Data Module
=======================================================

* This module standardizes **daily** data retrieval around Hòn Mun, Nha Trang
(approx lat=12.16764, lon=109.30581) across **16+ Weather/Marine & Satellite APIs**.
* It focuses on **robust engineering**: chunked time windows, retries with
exponential backoff, optional rate-limit pacing, and consistent outputs
(pandas DataFrame for numeric series; bytes/metadata for imagery).

* Notes
- Some providers need API keys or paid tiers for historical data. Insert your
  credentials via each class' constructor.
- This environment can't call the internet. Use locally with Python 3.9+.
- Date strings accepted: "YYYYMMDD" or "YYYY-MM-DD". Outputs use ISO dates.

* Stacks
------
Weather & Marine
    1) NASA POWER (daily since 1981)
    2) Open-Meteo Archive (ERA5-based)
    3) OpenWeather One Call (per-day timemachine)
    4) Weatherbit (history/daily)
    5) WeatherAPI.com (history)
    6) Meteostat (via RapidAPI) - daily aggregates
    7) MET Norway Locationforecast - mainly forecast (filter to range)
    8) Visual Crossing Timeline
    9) NOAA CDO (NCEI) - requires station or location identifiers
    10) Stormglass (marine point, limited free tier)

Satellite / Ocean Products
    11) NOAA OISST (SST daily via ERDDAP)
    12) NOAA Coral Reef Watch (DHW/SST via ERDDAP)
    13) NASA GIBS (imagery tiles - daily)
    14) Sentinel-2 (stub - Process API; chunked scaffolding)
    15) Landsat (stub)
    16) Copernicus Marine (CMEMS) (stub)
    17) MODIS/VIIRS Ocean Color (stub)
    18) Agromonitoring (stub)
"""
