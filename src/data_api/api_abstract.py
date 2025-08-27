"""Abstract class"""


from __future__ import annotations

import abc
import time
from typing import Tuple
import pandas as pd
import requests

from src.data_api.support_functions.support_functions import (
    Pacer,
    make_session
)


class EnvironmentalDataAPI(abc.ABC):
    """Abstract base classes"""
    def __init__(self, name: str, min_interval_sec: float = 0.0):
        self.name = name
        self.session = make_session()
        self.pacer = Pacer(min_interval_sec=min_interval_sec)

    @abc.abstractmethod
    def fetch_daily(self, location: Tuple[float, float], start_date: str, end_date: str) -> pd.DataFrame:
        """Return a daily DataFrame with an ISO `date` column and variables."""

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
