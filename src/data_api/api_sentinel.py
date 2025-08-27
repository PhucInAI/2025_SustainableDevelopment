"""Daily Sentinel-2 RGB imagery around Hòn Mun (or a user AOI)."""

# pylint:disable=W0718

import io
import os
import time
import requests
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
from PIL import Image
import pandas as pd

from src.data_api.api_abstract import EnvironmentalDataAPI
from src.data_api.support_functions.support_functions import (
    make_bbox_from_point,
    bbox4326_to_epsg32649,
    to_iso,
    split_by_2017,
    daterange_inclusive,
)
from src.utils.config import config

class SentinelHubS2ImageryAPI(EnvironmentalDataAPI):
    """
    Daily Sentinel-2 RGB imagery around Hòn Mun (or a user AOI).
    - Auto splits date range at 2017-01-01: L1C (pre-2017) / L2A (2017+)
    - Daily modes:
        * "available_only": one image per day that actually has data (Catalog distinct=date)
        * "daily_all": strict daily cadence; optional lookback window + mosaickingOrder to fill
    - Returns: pandas.DataFrame with columns:
        date (ISO), collection, mime, bytes, path (if saved), crs, bbox, width, height, status
    """
    PROCESS_URL = config.sentinel_url_process
    CATALOG_URL = config.sentinel_url_catalog

    def __init__(
                    self,
                    name: str = "sentinel-hub-s2",
                    min_interval_sec: float = 0.0,

                    # AOI config
                    center_lat: float = config.roi_lat,
                    center_lon: float = config.roi_lon,
                    buffer_deg: float = config.degree_offset,
                    bbox_4326: Optional[List[float]] = None,                    # [minx,miny,maxx,maxy] in EPSG:4326 overrides center/buffer

                    # resolution / CRS
                    prefer_utm: bool = True,                                    # use UTM + resx/resy (meters) if possible, fallback when prefer_utm=False or pyproj unavailable
                    res_m: int = 10,                                            # native S2 RGB
                    width: int = 512,
                    height: int = 512,

                    # fetch behavior
                    mode: str = "available_only",                               # or "daily_all"
                    lookback_days: int = 3,                                     # expand time window backward in daily_all mode
                    max_cloud: int = 30,
                    mosaicking_order: str = "leastCC",                          # mostRecent | leastRecent | leastCC
                    gain: float = 2.5, gamma: float = 1.6,
                ):
        super().__init__(name=name, min_interval_sec=min_interval_sec)
        client = BackendApplicationClient(client_id=config.sentinel_id_key)
        oauth = OAuth2Session(client=client)
        token = oauth.fetch_token(
                                    token_url = config.sentinel_url_token,
                                    client_secret=config.sentinel_id_secret,
                                    include_client_id=config.sentinel_id_key
                                )
        self.token = token['access_token']
        self.headers_json = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        self.mode = mode
        self.lookback_days = max(0, int(lookback_days))
        self.max_cloud = int(max_cloud)
        self.mosaicking_order = mosaicking_order
        self.save_dir = config.sentinel_output_folder
        self.prefer_utm = prefer_utm
        self.res_m = int(res_m)
        self.width = int(width)
        self.height = int(height)
        self.gain = float(gain)
        self.gamma = float(gamma)

        # AOI
        if bbox_4326 is None:
            self.bbox_4326 = make_bbox_from_point(center_lon, center_lat, buffer_deg)
        else:
            self.bbox_4326 = list(map(float, bbox_4326))

        # CRS/bbox for Process API output
        self.crs_epsg = "http://www.opengis.net/def/crs/OGC/1.3/CRS84"  # default
        self.bbox_process = self.bbox_4326
        self.use_res_m = False

        if self.prefer_utm:
            utm = bbox4326_to_epsg32649(self.bbox_4326)
            if utm is not None:
                self.crs_epsg = "http://www.opengis.net/def/crs/EPSG/0/32649"
                self.bbox_process = utm
                self.use_res_m = True  # resx/resy in meters

        # Evalscript (RGB + AUTO scaling + mild stretch)
        self.evalscript = f"""//VERSION=3
            function setup() {{
            return {{ input: ["B02","B03","B04","dataMask"], output: {{ bands: 3, sampleType: "AUTO" }} }};
            }}
            const GAIN = {self.gain}, GAMMA = {self.gamma};
            function evaluatePixel(s) {{
            let r = Math.pow(Math.min(1, GAIN * s.B04), 1.0 / GAMMA);
            let g = Math.pow(Math.min(1, GAIN * s.B03), 1.0 / GAMMA);
            let b = Math.pow(Math.min(1, GAIN * s.B02), 1.0 / GAMMA);
            return [r,g,b];
            }}
        """

    # --------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------
    def fetch_daily(
                    self,
                    location: Tuple[float, float],
                    start_date: str = config.date_start,
                    end_date: str = config.date_end
                ) -> pd.DataFrame:
        """
        Returns a DataFrame:
          ['date','collection','mime','bytes','path','crs','bbox','width','height','status']
        """
        a = to_iso(start_date)
        b = to_iso(end_date)
        segments = split_by_2017(a, b)  # [(seg_start, seg_end, collection, subfolder), ...]
        rows: List[Dict] = []

        for seg_start, seg_end, collection, sub in segments:
            if self.mode == "available_only":
                # Only days with actual scenes (Catalog distinct=date)
                days = self._catalog_distinct_dates(seg_start, seg_end, collection, self.max_cloud)
            else:
                # Strict daily cadence
                days = list(daterange_inclusive(seg_start, seg_end))

            for d in days:
                try:
                    out = self._process_day(d, collection, sub)
                    rows.append(out)
                except Exception as e:
                    rows.append({
                        "date": d, "collection": collection, "mime": None, "bytes": None,
                        "path": None, "crs": self.crs_epsg, "bbox": self.bbox_process,
                        "width": None, "height": None, "status": f"error: {e}"
                    })

        df = pd.DataFrame(rows)
        # ensure sorted by date
        if not df.empty and "date" in df.columns:
            df = df.sort_values("date").reset_index(drop=True)
        return df

    # --------------------------------------------------------------------
    # Internal functions
    # --------------------------------------------------------------------
    def _process_day(self, day_iso: str, collection: str, subfolder: str) -> Dict:
        # widen window for daily_all mode
        start_iso = f"{day_iso}T00:00:00Z"
        if self.mode == "daily_all" and self.lookback_days > 0:
            d0 = (datetime.fromisoformat(day_iso) - timedelta(days=self.lookback_days)).strftime("%Y-%m-%dT00:00:00Z")
            start_iso = d0
        end_iso = f"{day_iso}T23:59:59Z"

        input_bounds = {
            "bbox": self.bbox_process,
            "properties": {"crs": self.crs_epsg}
        }
        data_block = [{
            "type": collection,
            "dataFilter": {
                "timeRange": {"from": start_iso, "to": end_iso},
                "maxCloudCoverage": self.max_cloud,
                "mosaickingOrder": self.mosaicking_order
            },
            "processing": { "upsampling": "BILINEAR", "downsampling": "BILINEAR" }
        }]

        output_block = {
            "responses": [{"identifier": "default", "format": {"type": "image/png"}}]
        }
        if self.use_res_m:
            output_block.update({"resx": self.res_m, "resy": self.res_m})
        else:
            output_block.update({"width": self.width, "height": self.height})

        req = {
            "input": {"bounds": input_bounds, "data": data_block},
            "output": output_block,
            "evalscript": self.evalscript
        }

        r = self._post(self.PROCESS_URL, headers={"Authorization": self.headers_json["Authorization"]}, json=req)
        mime = r.headers.get("Content-Type", "")
        content = r.content

        # Optional “all black” guard if PIL is present
        w = h = None
        if Image is not None and mime.startswith("image/"):
            try:
                im = Image.open(io.BytesIO(content))
                w, h = im.size
                if all(m[1] <= 3 for m in im.getextrema()):
                    # try alternate collection once (flip L1C/L2A) within this day, else keep bytes
                    alt = "sentinel-2-l1c" if collection.endswith("l2a") else "sentinel-2-l2a"
                    if alt != collection:
                        content, mime, w, h = self._try_alt_collection(day_iso, alt)
            except Exception:
                pass

        out_path = None
        if self.save_dir:
            # nest by collection subfolder
            folder = os.path.join(self.save_dir, subfolder)
            os.makedirs(folder, exist_ok=True)
            out_path = os.path.join(folder, f"{day_iso}.png")
            with open(out_path, "wb") as f:
                f.write(content)

        return {
            "date": day_iso,
            "collection": collection,
            "mime": mime,
            "bytes": content,
            "path": out_path,
            "crs": self.crs_epsg,
            "bbox": self.bbox_process,
            "width": w,
            "height": h,
            "status": "ok"
        }

    def _try_alt_collection(self, day_iso: str, collection: str):
        start_iso = f"{day_iso}T00:00:00Z"
        end_iso = f"{day_iso}T23:59:59Z"
        req = {
            "input": {
                "bounds": {"bbox": self.bbox_process, "properties": {"crs": self.crs_epsg}},
                "data": [{
                    "type": collection,
                    "dataFilter": {
                        "timeRange": {"from": start_iso, "to": end_iso},
                        "maxCloudCoverage": self.max_cloud,
                        "mosaickingOrder": self.mosaicking_order
                    },
                    "processing": {"upsampling": "BILINEAR", "downsampling": "BILINEAR"}
                }]
            },
            "output": (
                {"resx": self.res_m, "resy": self.res_m, "responses": [{"identifier":"default","format":{"type":"image/png"}}]}
                if self.use_res_m else
                {"width": self.width, "height": self.height, "responses": [{"identifier":"default","format":{"type":"image/png"}}]}
            ),
            "evalscript": self.evalscript
        }
        r = self._post(self.PROCESS_URL, headers={"Authorization": self.headers_json["Authorization"]}, json=req)
        mime = r.headers.get("Content-Type", "")
        content = r.content
        w = h = None
        if Image is not None and mime.startswith("image/"):
            try:
                im = Image.open(io.BytesIO(content))
                w, h = im.size
            except Exception:
                pass
        return content, mime, w, h

    def _catalog_distinct_dates(self, start_date: str, end_date: str, collection: str, max_cloud: Optional[int]) -> List[str]:
        """Return list of ISO YYYY-MM-DD strings with actual acquisitions."""
        payload = {
            "bbox": self.bbox_4326,  # Catalog accepts 4326 bbox
            "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
            "collections": [collection],
            "limit": 500,
            "distinct": "date",
        }
        if max_cloud is not None:
            payload["filter"] = f"eo:cloud_cover <= {int(max_cloud)}"
            payload["filter-lang"] = "cql2-text"

        r = requests.post(self.CATALOG_URL, headers=self.headers_json, json=payload, timeout=120)
        dates: List[str] = []
        try:
            j = r.json()
        except Exception:
            return dates

        feats = j.get("features", [])
        if feats and isinstance(feats[0], str):
            return sorted(feats)

        # fallback: page with fields include
        dates_set = set()
        next_tok = None
        while True:
            p2 = {
                "bbox": self.bbox_4326,
                "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
                "collections": [collection],
                "limit": 100,
                "fields": {"include": ["properties.datetime","properties.eo:cloud_cover"],
                           "exclude": ["assets","links","geometry"]},
            }
            if next_tok is not None:
                p2["next"] = next_tok
            r2 = self._post(self.CATALOG_URL, headers=self.headers_json, json=p2)
            j2 = r2.json()
            for feat in j2.get("features", []):
                pr = feat.get("properties", {})
                dt = pr.get("datetime")
                cloud = pr.get("eo:cloud_cover")
                if dt:
                    day = dt[:10]
                    if (max_cloud is None) or (cloud is None) or (cloud <= int(max_cloud)):
                        dates_set.add(day)
            ctx = j2.get("context", {})
            if "next" in ctx:
                next_tok = ctx["next"]
                time.sleep(0.2)
            else:
                break

        return sorted(dates_set)
