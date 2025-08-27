"""Config for whole project"""

# pylint: disable=W1201

from __future__ import annotations

import os
import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional, Tuple

from dotenv import load_dotenv

from src.utils.ai_logger import aiLogger


# -----------------------------------------------------------------------------
# Load .env from project root
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
env_path = ROOT / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=False)


# -----------------------------------------------------------------------------
# Helpers for parsing & validation
# -----------------------------------------------------------------------------
def _get_str(name: str, default: Optional[str] = None) -> str:
    raw = os.getenv(name)
    if raw is None or raw == "":
        if default is None:
            msg = f"Missing required env var: {name}"
            aiLogger.error(msg)
            raise RuntimeError()
        return default
    return raw

def _get_float(name: str, default: Optional[float] = None) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        if default is None:
            msg = f"Missing required float env var: {name}"
            aiLogger.error(msg)
            raise RuntimeError()
        return default
    try:
        return float(raw.strip().strip("'").strip('"'))
    except ValueError as e:
        msg = f"Invalid float for {name}: {raw!r}"
        aiLogger.error(msg)
        raise RuntimeError() from e

def _get_int(name: str, default: Optional[int] = None) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        if default is None:
            msg = f"Missing required int env var: {name}"
            aiLogger.error(msg)
            raise RuntimeError()
        return default
    try:
        return int(raw.strip().strip("'").strip('"'))
    except ValueError as e:
        msg = f"Invalid int for {name}: {raw!r}"
        aiLogger.error(msg)
        raise RuntimeError() from e

def _parse_date(name: str) -> date:
    raw = _get_str(name)
    s = raw.strip().strip("'").strip('"')
    try:
        y, m, d = map(int, s.split("-"))
        return date(y, m, d)
    except Exception as e:
        msg = f"Invalid date for {name}: {raw!r} (expected YYYY-MM-DD)"
        aiLogger.error(msg)
        raise RuntimeError() from e

def _log_level_to_std(name: str) -> str:
    # Accept things like DEBUG, Info, "warning", etc.
    lvl = _get_str(name, "INFO").strip().strip("'").strip('"').upper()
    valid = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
    if lvl not in valid:
        msg = f"Invalid LOG_LEVEL {lvl!r}. Choose one of {sorted(valid)}."
        aiLogger.error(msg)
        raise RuntimeError()
    return lvl

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    """Config for whole project"""
    # System
    log_level: str

    # Region of interest
    roi_lat: float
    roi_lon: float
    degree_offset: float

    # Date range
    date_start: date
    date_end: date

    # Output resolution
    out_res_x: int
    out_res_y: int

    # Sentinel Hub
    sentinel_id_key: str = field(repr=False)
    sentinel_id_secret: str = field(repr=False)
    sentinel_url_token: str
    sentinel_url_catalog: str
    sentinel_url_process: str
    sentinel_output_folder: str

    # ---- Convenience computed properties ----
    @property
    def bbox_deg(self) -> Tuple[float, float, float, float]:
        """(lat_min, lon_min, lat_max, lon_max) in degrees."""
        return (
            self.roi_lat - self.degree_offset,
            self.roi_lon - self.degree_offset,
            self.roi_lat + self.degree_offset,
            self.roi_lon + self.degree_offset,
        )

    @property
    def km_per_deg(self) -> Tuple[float, float]:
        """(km per degree latitude, km per degree longitude at roi_lat)."""
        km_lat = 111.13
        km_lon = 111.32 * math.cos(math.radians(self.roi_lat))
        return km_lat, km_lon

    @property
    def half_extents_km(self) -> Tuple[float, float]:
        """Half-size from center to edge of the bbox in km: (north-south, east-west)."""
        km_lat, km_lon = self.km_per_deg
        return self.degree_offset * km_lat, self.degree_offset * km_lon

    @property
    def corner_radius_km(self) -> float:
        """Distance from center to a bbox corner in km (small-area approximation)."""
        ns, ew = self.half_extents_km
        return math.hypot(ns, ew)

def load_config() -> Config:
    """Load config"""
    # System
    log_level = _log_level_to_std("LOG_LEVEL")

    # ROI & date range
    roi_lat = _get_float("ROI_LAT")
    roi_lon = _get_float("ROI_LON")
    degree_offset = _get_float("DEGREE_OFFSET")
    date_start = _parse_date("DATE_START")
    date_end = _parse_date("DATE_END")
    if date_end < date_start:
        msg = "DATE_END must be on or after DATE_START"
        aiLogger.error(msg)
        raise RuntimeError()

    # Output
    out_res_x = _get_int("OUT_RES_X")
    out_res_y = _get_int("OUT_RES_Y")

    # Sentinel (handle common SECRET typo: SENTINAL_ID_SECRET)
    sentinel_id_key = _get_str("SENTINEL_ID_KEY")
    sentinel_id_secret = os.getenv("SENTINEL_ID_SECRET") or os.getenv("SENTINAL_ID_SECRET")
    if not sentinel_id_secret:
        msg = "Missing SENTINEL_ID_SECRET (or legacy SENTINAL_ID_SECRET)"
        aiLogger.error(msg)
        raise RuntimeError()

    sentinel_url_token = _get_str("SENTINEL_URL_TOKEN")
    sentinel_url_catalog = _get_str("SENTINEL_URL_CATALOG")
    sentinel_url_process = _get_str("SENTINEL_URL_PROCESS")
    sentinel_output_folder = _get_str("SENTINEL_OUTPUT_FOLDER", "Sentinel")

    return Config(
        log_level=log_level,
        roi_lat=roi_lat,
        roi_lon=roi_lon,
        degree_offset=degree_offset,
        date_start=date_start,
        date_end=date_end,
        out_res_x=out_res_x,
        out_res_y=out_res_y,
        sentinel_id_key=sentinel_id_key,
        sentinel_id_secret=sentinel_id_secret,
        sentinel_url_token=sentinel_url_token,
        sentinel_url_catalog=sentinel_url_catalog,
        sentinel_url_process=sentinel_url_process,
        sentinel_output_folder=sentinel_output_folder,
    )


config = load_config()

aiLogger.info("LOG_LEVEL: " + config.log_level)
aiLogger.info("ROI (lat, lon): (" + str(config.roi_lat) + "," + str(config.roi_lon) + ")")
aiLogger.info("BBOX deg (lat_min, lon_min, lat_max, lon_max): "+ str(config.bbox_deg))
aiLogger.info("Half-extents (NS km, EW km):"+ str(config.half_extents_km))
aiLogger.info("Corner radius (km):"+ str(round(config.corner_radius_km, 3)))
aiLogger.info("Date range:"+ str(config.date_start)+ "→"+ str(config.date_end))
aiLogger.info("Out resolution: (" + str(config.out_res_x) + "," + str(config.out_res_y) +")")
aiLogger.info("Output folder:"+ str(config.sentinel_output_folder))
