"""
Hòn Mun Environmental Data Module (Chunked API Fetchers)
=======================================================

This module standardizes **daily** data retrieval around Hòn Mun, Nha Trang
(approx lat=12.16764, lon=109.30581) across **16+ Weather/Marine & Satellite APIs**.
It focuses on **robust engineering**: chunked time windows, retries with
exponential backoff, optional rate‑limit pacing, and consistent outputs
(pandas DataFrame for numeric series; bytes/metadata for imagery).

⚠️ Notes
- Some providers need API keys or paid tiers for historical data. Insert your
  credentials via each class' constructor.
- This environment can’t call the internet. Use locally with Python 3.9+.
- Date strings accepted: "YYYYMMDD" or "YYYY-MM-DD". Outputs use ISO dates.

Stacks
------
Weather & Marine
  1) NASA POWER (daily since 1981)
  2) Open‑Meteo Archive (ERA5-based)
  3) OpenWeather One Call (per‑day timemachine)
  4) Weatherbit (history/daily)
  5) WeatherAPI.com (history)
  6) Meteostat (via RapidAPI) – daily aggregates
  7) MET Norway Locationforecast – mainly forecast (filter to range)
  8) Visual Crossing Timeline
  9) NOAA CDO (NCEI) – requires station or location identifiers
 10) Stormglass (marine point, limited free tier)

Satellite / Ocean Products
 11) NOAA OISST (SST daily via ERDDAP)
 12) NOAA Coral Reef Watch (DHW/SST via ERDDAP)
 13) NASA GIBS (imagery tiles – daily)
 14) Sentinel‑2 (stub – Process API; chunked scaffolding)
 15) Landsat (stub)
 16) Copernicus Marine (CMEMS) (stub)
 17) MODIS/VIIRS Ocean Color (stub)
 18) Agromonitoring (stub)

You can extend/enable any stub by filling auth and endpoint specifics.
"""
from __future__ import annotations

import abc
import io
import math
import time
import json
import hashlib
import pathlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ----------------------------
# Utility helpers
# ----------------------------
ISO = "%Y-%m-%d"
YMD = "%Y%m%d"


def _to_iso(d: Union[str, datetime]) -> str:
    if isinstance(d, datetime):
        return d.strftime(ISO)
    s = str(d)
    if len(s) == 8 and s.isdigit():
        return datetime.strptime(s, YMD).strftime(ISO)
    return datetime.strptime(s[:10], ISO).strftime(ISO)


def _to_ymd(d: Union[str, datetime]) -> str:
    return datetime.strptime(_to_iso(d), ISO).strftime(YMD)


def _iter_date_chunks(start: Union[str, datetime], end: Union[str, datetime], days: int) -> Iterable[Tuple[str, str]]:
    """Yield [start, end] chunks in ISO date strings inclusive.
    """
    s = datetime.strptime(_to_iso(start), ISO)
    e = datetime.strptime(_to_iso(end), ISO)
    cur = s
    while cur <= e:
        nxt = min(cur + timedelta(days=days - 1), e)
        yield cur.strftime(ISO), nxt.strftime(ISO)
        cur = nxt + timedelta(days=1)


class _Pacer:
    """Simple rate pacer: ensures a minimum delay between requests."""
    def __init__(self, min_interval_sec: float = 0.0):
        self.min_interval = max(0.0, float(min_interval_sec))
        self._last = 0.0

    def wait(self):
        if self.min_interval <= 0:
            return
        now = time.time()
        delta = now - self._last
        sleep_for = self.min_interval - delta
        if sleep_for > 0:
            time.sleep(sleep_for)
        self._last = time.time()


def _make_session(total_retries: int = 5, backoff: float = 0.6) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        status=total_retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


# ----------------------------
# Abstract base classes
# ----------------------------
class EnvironmentalDataAPI(abc.ABC):
    def __init__(self, name: str, min_interval_sec: float = 0.0):
        self.name = name
        self.session = _make_session()
        self.pacer = _Pacer(min_interval_sec=min_interval_sec)

    @abc.abstractmethod
    def fetch_daily(self, location: Tuple[float, float], start_date: str, end_date: str) -> pd.DataFrame:
        """Return a daily DataFrame with an ISO `date` column and variables."""
        ...

    def _get(self, url: str, **kwargs) -> requests.Response:
        self.pacer.wait()
        r = self.session.get(url, timeout=60, **kwargs)
        if r.status_code == 429:
            # exponential backoff with jitter
            for i in range(5):
                sleep = (2 ** i) + (0.1 * i)
                time.sleep(sleep)
                self.pacer.wait()
                r = self.session.get(url, timeout=60, **kwargs)
                if r.ok:
                    break
        r.raise_for_status()
        return r

    def _post(self, url: str, **kwargs) -> requests.Response:
        self.pacer.wait()
        r = self.session.post(url, timeout=90, **kwargs)
        if r.status_code == 429:
            for i in range(5):
                sleep = (2 ** i) + (0.1 * i)
                time.sleep(sleep)
                self.pacer.wait()
                r = self.session.post(url, timeout=90, **kwargs)
                if r.ok:
                    break
        r.raise_for_status()
        return r


# ----------------------------
# Weather & Marine APIs
# ----------------------------
class PowerWeatherAPI(EnvironmentalDataAPI):
    """NASA POWER daily – chunk by year to keep payloads manageable."""
    BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"

    def __init__(self, min_interval_sec: float = 0.2, parameters: str = "T2M_MAX,T2M_MIN,PRECTOT"):
        super().__init__("NASA POWER", min_interval_sec)
        self.parameters = parameters

    def fetch_daily(self, location: Tuple[float, float], start_date: str, end_date: str) -> pd.DataFrame:
        lat, lon = location
        frames: List[pd.DataFrame] = []
        for a, b in _iter_date_chunks(start_date, end_date, 365):
            params = {
                "parameters": self.parameters,
                "community": "AG",
                "latitude": lat,
                "longitude": lon,
                "start": _to_ymd(a),
                "end": _to_ymd(b),
                "format": "CSV",
            }
            r = self._get(self.BASE, params=params)
            df = pd.read_csv(io.StringIO(r.text))
            # Standardize date
            if {"YEAR","MO","DA"}.issubset(df.columns):
                df["date"] = pd.to_datetime(df[["YEAR","MO","DA"]].astype(int).astype(str).agg("-".join, axis=1))
                df.drop(columns=[c for c in ["YEAR","MO","DA","DOY"] if c in df.columns], inplace=True)
            frames.append(df)
        out = pd.concat(frames, ignore_index=True)
        out["date"] = pd.to_datetime(out["date"]).dt.strftime(ISO)
        return out.sort_values("date").reset_index(drop=True)


class OpenMeteoAPI(EnvironmentalDataAPI):
    """Open‑Meteo archive – chunk by 365 days, return daily variables."""
    BASE = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self, min_interval_sec: float = 0.1,
                 daily_vars: str = "temperature_2m_max,temperature_2m_min,precipitation_sum"):
        super().__init__("Open-Meteo", min_interval_sec)
        self.daily_vars = daily_vars

    def fetch_daily(self, location: Tuple[float, float], start_date: str, end_date: str) -> pd.DataFrame:
        lat, lon = location
        frames = []
        for a, b in _iter_date_chunks(start_date, end_date, 365):
            params = {
                "latitude": lat, "longitude": lon,
                "start_date": a, "end_date": b,
                "daily": self.daily_vars,
                "timezone": "UTC", "format": "csv",
            }
            r = self._get(self.BASE, params=params)
            frames.append(pd.read_csv(io.StringIO(r.text)))
        out = pd.concat(frames, ignore_index=True)
        out.rename(columns={"time": "date"}, inplace=True)
        out["date"] = pd.to_datetime(out["date"]).dt.strftime(ISO)
        return out.sort_values("date").reset_index(drop=True)


class OpenWeatherMapAPI(EnvironmentalDataAPI):
    """OpenWeather One Call 3.0 timemachine – 1 day per call, chunk per day.
    Aggregates hourly to daily averages/sums.
    """
    BASE = "https://api.openweathermap.org/data/3.0/onecall/timemachine"

    def __init__(self, api_key: str, min_interval_sec: float = 0.25):
        super().__init__("OpenWeatherMap", min_interval_sec)
        self.api_key = api_key

    def fetch_daily(self, location: Tuple[float, float], start_date: str, end_date: str) -> pd.DataFrame:
        lat, lon = location
        days = pd.date_range(_to_iso(start_date), _to_iso(end_date), freq="D")
        recs: List[Dict[str, Any]] = []
        for d in days:
            params = {
                "lat": lat, "lon": lon,
                "dt": int(pd.Timestamp(d).timestamp()),
                "appid": self.api_key, "units": "metric",
            }
            r = self._get(self.BASE, params=params)
            js = r.json()
            hourly = js.get("hourly", [])
            if not hourly:
                continue
            hdf = pd.DataFrame(hourly)
            recs.append({
                "date": d.strftime(ISO),
                "temp_mean": hdf["temp"].mean(),
                "humidity_mean": hdf["humidity"].mean(),
                "wind_speed_mean": hdf.get("wind_speed", pd.Series(dtype=float)).mean(),
                "pressure_mean": hdf.get("pressure", pd.Series(dtype=float)).mean(),
                "precip_sum": hdf.get("rain", pd.Series([0]*len(hdf))).apply(lambda x: x.get("1h", 0) if isinstance(x, dict) else 0).sum()
            })
        return pd.DataFrame.from_records(recs).sort_values("date").reset_index(drop=True)


class WeatherbitAPI(EnvironmentalDataAPI):
    """Weatherbit history/daily – supports date windows; chunk by 180 days."""
    BASE = "https://api.weatherbit.io/v2.0/history/daily"

    def __init__(self, api_key: str, min_interval_sec: float = 0.2):
        super().__init__("Weatherbit", min_interval_sec)
        self.api_key = api_key

    def fetch_daily(self, location: Tuple[float, float], start_date: str, end_date: str) -> pd.DataFrame:
        lat, lon = location
        frames = []
        for a, b in _iter_date_chunks(start_date, end_date, 180):
            params = {"lat": lat, "lon": lon, "start_date": a, "end_date": b, "units": "M", "key": self.api_key}
            r = self._get(self.BASE, params=params)
            data = r.json().get("data", [])
            if data:
                df = pd.DataFrame(data)
                df.rename(columns={"datetime": "date"}, inplace=True)
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


class WeatherAPIService(EnvironmentalDataAPI):
    """WeatherAPI.com history – chunk by day (free) or larger windows (paid)."""
    BASE = "http://api.weatherapi.com/v1/history.json"

    def __init__(self, api_key: str, min_interval_sec: float = 0.2):
        super().__init__("WeatherAPI.com", min_interval_sec)
        self.api_key = api_key

    def fetch_daily(self, location: Tuple[float, float], start_date: str, end_date: str) -> pd.DataFrame:
        lat, lon = location
        days = pd.date_range(_to_iso(start_date), _to_iso(end_date), freq="D")
        recs: List[Dict[str, Any]] = []
        for d in days:
            params = {"key": self.api_key, "q": f"{lat},{lon}", "dt": d.strftime(ISO)}
            r = self._get(self.BASE, params=params)
            js = r.json()
            for fd in js.get("forecast", {}).get("forecastday", []):
                day = fd.get("day", {})
                recs.append({
                    "date": fd.get("date"),
                    "max_temp_c": day.get("maxtemp_c"),
                    "min_temp_c": day.get("mintemp_c"),
                    "avg_temp_c": day.get("avgtemp_c"),
                    "total_precip_mm": day.get("totalprecip_mm"),
                    "avg_humidity": day.get("avghumidity"),
                })
        return pd.DataFrame.from_records(recs).sort_values("date").reset_index(drop=True)


class MeteostatAPI(EnvironmentalDataAPI):
    """Meteostat daily – chunk by ≤10 years per Meteostat guidance."""
    BASE = "https://meteostat.p.rapidapi.com/point/daily"

    def __init__(self, rapidapi_key: str, min_interval_sec: float = 0.2):
        super().__init__("Meteostat", min_interval_sec)
        self.rapidapi_key = rapidapi_key

    def fetch_daily(self, location: Tuple[float, float], start_date: str, end_date: str) -> pd.DataFrame:
        lat, lon = location
        frames: List[pd.DataFrame] = []
        for a, b in _iter_date_chunks(start_date, end_date, 3650):  # 10 years
            headers = {"X-RapidAPI-Key": self.rapidapi_key, "X-RapidAPI-Host": "meteostat.p.rapidapi.com"}
            params = {"lat": lat, "lon": lon, "start": a, "end": b}
            r = self._get(self.BASE, params=params, headers=headers)
            data = r.json().get("data", [])
            if data:
                frames.append(pd.DataFrame(data))
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not out.empty and "date" in out.columns:
            out["date"] = pd.to_datetime(out["date"]).dt.strftime(ISO)
            out.sort_values("date", inplace=True)
        return out.reset_index(drop=True)


class MetNorwayAPI(EnvironmentalDataAPI):
    """MET Norway – forecast/nowcast; fetch then filter to requested range."""
    BASE = "https://api.met.no/weatherapi/locationforecast/2.0/compact"

    def __init__(self, user_agent: str = "hon-mun-data-fetcher", min_interval_sec: float = 0.2):
        super().__init__("MET Norway", min_interval_sec)
        self.user_agent = user_agent

    def fetch_daily(self, location: Tuple[float, float], start_date: str, end_date: str) -> pd.DataFrame:
        lat, lon = location
        headers = {"User-Agent": self.user_agent}
        r = self._get(self.BASE, params={"lat": lat, "lon": lon}, headers=headers)
        series = r.json().get("properties", {}).get("timeseries", [])
        if not series:
            return pd.DataFrame()
        df = pd.DataFrame([{**s.get("data", {}).get("instant", {}).get("details", {}), "time": s.get("time")} for s in series])
        df["date"] = pd.to_datetime(df["time"]).dt.strftime(ISO)
        # group to daily means
        out = df.groupby("date").mean(numeric_only=True).reset_index()
        # filter to requested window
        mask = (out["date"] >= _to_iso(start_date)) & (out["date"] <= _to_iso(end_date))
        return out.loc[mask].reset_index(drop=True)


class VisualCrossingAPI(EnvironmentalDataAPI):
    """Visual Crossing Timeline – chunk by 365 days, return daily."""
    BASE = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"

    def __init__(self, api_key: str, min_interval_sec: float = 0.2):
        super().__init__("Visual Crossing", min_interval_sec)
        self.api_key = api_key

    def fetch_daily(self, location: Tuple[float, float], start_date: str, end_date: str) -> pd.DataFrame:
        lat, lon = location
        frames = []
        for a, b in _iter_date_chunks(start_date, end_date, 365):
            url = f"{self.BASE}/{lat},{lon}/{a}/{b}"
            params = {"unitGroup": "metric", "include": "days", "key": self.api_key, "contentType": "json"}
            r = self._get(url, params=params)
            days = r.json().get("days", [])
            if days:
                frames.append(pd.DataFrame(days))
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not out.empty and "datetime" in out.columns:
            out.rename(columns={"datetime": "date"}, inplace=True)
        return out


class NoaaCDOAPI(EnvironmentalDataAPI):
    """NOAA Climate Data Online – chunk by month, handle pagination.
    Requires a `token` and ideally a `stationid` (e.g., 'GHCND:XXXX').
    """
    BASE = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"

    def __init__(self, token: str, datasetid: str = "GHCND", stationid: Optional[str] = None,
                 locationid: Optional[str] = None, min_interval_sec: float = 0.25):
        super().__init__("NOAA CDO", min_interval_sec)
        self.token = token
        self.datasetid = datasetid
        self.stationid = stationid
        self.locationid = locationid

    def fetch_daily(self, location: Tuple[float, float], start_date: str, end_date: str) -> pd.DataFrame:
        headers = {"token": self.token}
        frames = []
        for a, b in _iter_date_chunks(start_date, end_date, 31):
            params = {"datasetid": self.datasetid, "startdate": a, "enddate": b, "limit": 1000}
            if self.stationid:
                params["stationid"] = self.stationid
            if self.locationid:
                params["locationid"] = self.locationid
            offset = 1
            while True:
                params["offset"] = offset
                r = self._get(self.BASE, params=params, headers=headers)
                js = r.json()
                data = js.get("results", [])
                if not data:
                    break
                frames.append(pd.DataFrame(data))
                if len(data) < params["limit"]:
                    break
                offset += params["limit"]
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not out.empty:
            if "date" in out.columns:
                out["date"] = pd.to_datetime(out["date"]).dt.strftime(ISO)
            out.sort_values("date", inplace=True, errors="ignore")
        return out.reset_index(drop=True)


class StormglassAPI(EnvironmentalDataAPI):
    """Stormglass marine – chunk by 7 days, specify params & source."""
    BASE = "https://api.stormglass.io/v2/weather/point"

    def __init__(self, api_key: str, params: str = "waveHeight,windSpeed,windDirection,waterTemperature,humidity",
                 source: str = "noaa", min_interval_sec: float = 0.3):
        super().__init__("Stormglass", min_interval_sec)
        self.api_key = api_key
        self.params = params
        self.source = source

    def fetch_daily(self, location: Tuple[float, float], start_date: str, end_date: str) -> pd.DataFrame:
        lat, lon = location
        frames = []
        for a, b in _iter_date_chunks(start_date, end_date, 7):
            params = {
                "lat": lat, "lng": lon, "params": self.params, "source": self.source,
                "start": int(datetime.strptime(a, ISO).timestamp()),
                "end": int((datetime.strptime(b, ISO) + timedelta(days=1)).timestamp())
            }
            r = self._get(self.BASE, params=params, headers={"Authorization": self.api_key})
            js = r.json()
            hours = js.get("hours", [])
            if not hours:
                continue
            hdf = pd.DataFrame(hours)
            # Flatten param dicts (e.g., windSpeed: {noaa: value}) to direct columns
            for col in list(hdf.columns):
                if isinstance(hdf[col].dropna().iloc[0], dict):
                    hdf[col] = hdf[col].apply(lambda d: list(d.values())[0] if isinstance(d, dict) and d else None)
            hdf["date"] = pd.to_datetime(hdf["time"]).dt.strftime(ISO)
            frames.append(hdf.groupby("date").mean(numeric_only=True).reset_index())
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ----------------------------
# Satellite / Ocean Products
# ----------------------------
class OisstAPI(EnvironmentalDataAPI):
    """NOAA OISST via ERDDAP – chunk by month/year to avoid timeouts."""
    BASE = "https://www.ncei.noaa.gov/erddap/griddap/ncdc_oisst_v2_avhrr_by_time_zlev_lat_lon.csv"

    def __init__(self, min_interval_sec: float = 0.25):
        super().__init__("NOAA OISST", min_interval_sec)

    def fetch_daily(self, location: Tuple[float, float], start_date: str, end_date: str) -> pd.DataFrame:
        lat, lon = location
        frames = []
        for a, b in _iter_date_chunks(start_date, end_date, 31):
            query = f"sst[{a}T00:00:00Z:1:{b}T00:00:00Z][0.0][({lat}):({lat})][({lon}):({lon})]"
            url = f"{self.BASE}?{query}"
            r = self._get(url)
            df = pd.read_csv(io.StringIO(r.text))
            frames.append(df)
        out = pd.concat(frames, ignore_index=True)
        if "time" in out.columns:
            out.rename(columns={"time": "date"}, inplace=True)
            out["date"] = pd.to_datetime(out["date"]).dt.strftime(ISO)
        return out.sort_values("date").reset_index(drop=True)


class CoralReefWatchAPI(EnvironmentalDataAPI):
    """NOAA CRW via ERDDAP – chunk by month; variables like CRW_SST, CRW_DHW."""
    BASE = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/noaa_crw_v3_1_daily.csv"

    def __init__(self, variables: List[str] = None, min_interval_sec: float = 0.25):
        super().__init__("NOAA Coral Reef Watch", min_interval_sec)
        self.variables = variables or ["CRW_SST", "CRW_DHW"]

    def fetch_daily(self, location: Tuple[float, float], start_date: str, end_date: str) -> pd.DataFrame:
        lat, lon = location
        frames = []
        for a, b in _iter_date_chunks(start_date, end_date, 31):
            varq = ",".join(f"{v}[{a}:1:{b}][({lat}):({lat})][({lon}):({lon})]" for v in self.variables)
            url = f"{self.BASE}?{varq}"
            r = self._get(url)
            df = pd.read_csv(io.StringIO(r.text))
            frames.append(df)
        out = pd.concat(frames, ignore_index=True)
        if "time" in out.columns:
            out.rename(columns={"time": "date"}, inplace=True)
            out["date"] = pd.to_datetime(out["date"]).dt.strftime(ISO)
        return out.sort_values("date").reset_index(drop=True)


class GibsAPI(EnvironmentalDataAPI):
    """NASA GIBS – fetch daily tiles (PNG) for given dates; returns list of blobs."""
    TILE = "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/1.0.0/BlueMarble_ShadedRelief/{date}/250m/4/12/7.png"

    def __init__(self, min_interval_sec: float = 0.2):
        super().__init__("NASA GIBS", min_interval_sec)

    def fetch_daily(self, location: Tuple[float, float], start_date: str, end_date: str) -> pd.DataFrame:
        # For imagery tiles we return a table with date and bytes length (you can save bytes to files as needed)
        days = pd.date_range(_to_iso(start_date), _to_iso(end_date), freq="D")
        recs = []
        for d in days:
            url = self.TILE.format(date=d.strftime(ISO))
            try:
                r = self._get(url)
                recs.append({"date": d.strftime(ISO), "bytes": len(r.content), "url": url})
            except Exception as e:
                recs.append({"date": d.strftime(ISO), "bytes": 0, "url": url, "error": str(e)})
        return pd.DataFrame(recs)


# ---- Satellite stubs with chunk scaffolding (fill in auth & endpoints) ----
class Sentinel2API(EnvironmentalDataAPI):
    def __init__(self, oauth_token: Optional[str] = None, min_interval_sec: float = 0.3):
        super().__init__("Sentinel-2", min_interval_sec)
        self.token = oauth_token

    def fetch_daily(self, location: Tuple[float, float], start_date: str, end_date: str) -> pd.DataFrame:
        # Implement Process API calls per day/batch if token provided
        if not self.token:
            raise RuntimeError("Sentinel-2 requires an OAuth token. Provide `oauth_token`.")
        # Pseudocode: iterate days -> POST /api/v1/process with bbox/time -> collect stats/png bytes
        days = pd.date_range(_to_iso(start_date), _to_iso(end_date), freq="D")
        return pd.DataFrame({"date": [d.strftime(ISO) for d in days], "status": "requested"})


class LandsatAPI(EnvironmentalDataAPI):
    def __init__(self, username: Optional[str] = None, password: Optional[str] = None, min_interval_sec: float = 0.3):
        super().__init__("USGS Landsat", min_interval_sec)
        self.username = username
        self.password = password

    def fetch_daily(self, location: Tuple[float, float], start_date: str, end_date: str) -> pd.DataFrame:
        raise RuntimeError("Implement Landsat search/download via USGS M2M/EarthExplorer.")


class CopernicusMarineAPI(EnvironmentalDataAPI):
    def __init__(self, username: Optional[str] = None, password: Optional[str] = None, min_interval_sec: float = 0.3):
        super().__init__("Copernicus Marine", min_interval_sec)
        self.username = username
        self.password = password

    def fetch_daily(self, location: Tuple[float, float], start_date: str, end_date: str) -> pd.DataFrame:
        raise RuntimeError("Implement CMEMS product download (NetCDF) and aggregate daily.")


class ModisOceanColorAPI(EnvironmentalDataAPI):
    def __init__(self, earthdata_token: Optional[str] = None, min_interval_sec: float = 0.3):
        super().__init__("MODIS/VIIRS Ocean Color", min_interval_sec)
        self.token = earthdata_token

    def fetch_daily(self, location: Tuple[float, float], start_date: str, end_date: str) -> pd.DataFrame:
        raise RuntimeError("Implement OB.DAAC download and compute daily chlorophyll.")


class AgromonitoringAPI(EnvironmentalDataAPI):
    def __init__(self, api_key: str, polygon_id: Optional[str] = None, min_interval_sec: float = 0.3):
        super().__init__("Agromonitoring", min_interval_sec)
        self.api_key = api_key
        self.polygon_id = polygon_id

    def fetch_daily(self, location: Tuple[float, float], start_date: str, end_date: str) -> pd.DataFrame:
        if not self.polygon_id:
            raise RuntimeError("Agromonitoring requires a polygon_id representing the AOI.")
        # Example: image/search with start/end (unix), chunk by 30 days, then post-process to daily coverage
        frames = []
        for a, b in _iter_date_chunks(start_date, end_date, 30):
            url = "http://api.agromonitoring.com/agro/1.0/image/search"
            params = {
                "start": int(datetime.strptime(a, ISO).timestamp()),
                "end": int((datetime.strptime(b, ISO) + timedelta(days=1)).timestamp()),
                "polyid": self.polygon_id,
                "appid": self.api_key,
            }
            r = self._get(url, params=params)
            js = r.json()
            if js:
                frames.append(pd.DataFrame(js))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ----------------------------
# Example usage (run locally)
# ----------------------------
if __name__ == "__main__":
    HON_MUN = (12.16764, 109.30581)
    START = "1981-01-01"
    END = "2025-01-01"

    # NASA POWER
    power = PowerWeatherAPI()
    try:
        df_power = power.fetch_daily(HON_MUN, START, END)
        print("POWER", df_power.head())
    except Exception as e:
        print("POWER error:", e)

    # Open-Meteo
    om = OpenMeteoAPI()
    try:
        df_om = om.fetch_daily(HON_MUN, START, END)
        print("Open-Meteo", df_om.head())
    except Exception as e:
        print("Open-Meteo error:", e)

    # OISST (SST)
    oisst = OisstAPI()
    try:
        df_sst = oisst.fetch_daily(HON_MUN, "1981-09-01", END)
        print("OISST", df_sst.head())
    except Exception as e:
        print("OISST error:", e)

    # CRW (coral stress)
    crw = CoralReefWatchAPI()
    try:
        df_crw = crw.fetch_daily(HON_MUN, "1985-01-01", END)
        print("CRW", df_crw.head())
    except Exception as e:
        print("CRW error:", e)

    # GIBS imagery tile bytes per day
    gibs = GibsAPI()
    try:
        df_tile = gibs.fetch_daily(HON_MUN, "2023-08-01", "2023-08-05")
        print("GIBS", df_tile.head())
    except Exception as e:
        print("GIBS error:", e)
