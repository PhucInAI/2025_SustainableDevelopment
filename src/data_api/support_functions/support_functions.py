"""Support functions for data processing"""

# pylint: disable=W0718

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Iterable, Tuple, Union, Optional, List
try:
    from pyproj import Transformer
except Exception:
    Transformer = None # pylint:disable=C0103


import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ISO = "%Y-%m-%d"
YMD = "%Y%m%d"


def to_iso(d: Union[str, datetime]) -> str:
    """Conver to ISO format"""
    if isinstance(d, datetime):
        return d.strftime(ISO)
    s = str(d)
    if len(s) == 8 and s.isdigit():
        return datetime.strptime(s, YMD).strftime(ISO)
    return datetime.strptime(s[:10], ISO).strftime(ISO)


def to_ymd(d: Union[str, datetime]) -> str:
    """Convert to YMD format"""
    return datetime.strptime(to_iso(d), ISO).strftime(YMD)


def iter_date_chunks(
                    start: Union[str, datetime],
                    end: Union[str, datetime],
                    days: int,
                    )-> Iterable[Tuple[str, str]]:
    """Yield [start, end] chunks in ISO date strings inclusive."""
    s = datetime.strptime(to_iso(start), ISO)
    e = datetime.strptime(to_iso(end), ISO)
    cur = s
    while cur <= e:
        nxt = min(cur + timedelta(days=days - 1), e)
        yield cur.strftime(ISO), nxt.strftime(ISO)
        cur = nxt + timedelta(days=1)


class Pacer:
    """Simple rate pacer: ensures a minimum delay between requests."""
    def __init__(self, min_interval_sec: float = 0.0):
        self.min_interval = max(0.0, float(min_interval_sec))
        self._last = 0.0

    def wait(self):
        """Wait between request"""
        if self.min_interval <= 0:
            return
        now = time.time()
        delta = now - self._last
        sleep_for = self.min_interval - delta
        if sleep_for > 0:
            time.sleep(sleep_for)
        self._last = time.time()


def make_session(
                total_retries: int = 5,
                backoff: float = 0.6
                ) -> requests.Session:
    """Make request session"""
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


def make_bbox_from_point(lon: float, lat: float, buffer_deg: float) -> List[float]:
    """Return [minx,miny,maxx,maxy] in EPSG:4326 from center + buffer in degrees."""
    return [lon - buffer_deg, lat - buffer_deg, lon + buffer_deg, lat + buffer_deg]


def bbox4326_to_epsg32649(bbox4326: List[float]) -> Optional[List[float]]:
    """
    Reproject 4326 bbox to EPSG:32649 (UTM zone for Hòn Mun).
    Returns None if pyproj is unavailable.
    """
    if Transformer is None:
        return None
    t = Transformer.from_crs(4326, 32649, always_xy=True)
    x1, y1 = t.transform(bbox4326[0], bbox4326[1])
    x2, y2 = t.transform(bbox4326[2], bbox4326[3])
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def daterange_inclusive(start_iso: str, end_iso: str):
    """Yield ISO YYYY-MM-DD for every calendar day from start..end inclusive."""
    s = datetime.strptime(to_iso(start_iso), ISO).date()
    e = datetime.strptime(to_iso(end_iso), ISO).date()
    cur = s
    one = timedelta(days=1)
    while cur <= e:
        yield cur.strftime(ISO)
        cur += one


def split_by_2017(start_iso: str, end_iso: str):
    """
    Split [start,end] into [(s,e,collection,subfolder), ...] by the S2 L2A global cutoff (2017-01-01).
    - pre-2017 → sentinel-2-l1c, subfolder 's2-l1c'
    - 2017+    → sentinel-2-l2a, subfolder 's2-l2a'
    """
    cutoff = datetime(2017,1,1)
    s = datetime.strptime(to_iso(start_iso), ISO)
    e = datetime.strptime(to_iso(end_iso), ISO)
    if e < cutoff:
        return [(s.strftime(ISO), e.strftime(ISO), "sentinel-2-l1c", "s2-l1c")]
    if s >= cutoff:
        return [(s.strftime(ISO), e.strftime(ISO), "sentinel-2-l2a", "s2-l2a")]
    # crosses the boundary
    left_end = (cutoff - timedelta(days=1)).strftime(ISO)
    return [
        (s.strftime(ISO), left_end, "sentinel-2-l1c", "s2-l1c"),
        (cutoff.strftime(ISO), e.strftime(ISO), "sentinel-2-l2a", "s2-l2a"),
    ]
