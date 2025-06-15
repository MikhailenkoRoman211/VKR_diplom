from __future__ import annotations

import contextlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from io import StringIO
from typing import Dict, List, Sequence

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from urllib3.util.retry import Retry

_PRINT = threading.Lock()
_thread = threading.local()


def _sess() -> requests.Session:
    if not hasattr(_thread, "s"):
        s = requests.Session()
        s.mount("https://", HTTPAdapter(max_retries=Retry(total=0), pool_maxsize=10))
        _thread.s = s
    return _thread.s


def _get_json(url: str, params: dict, tries: int = 3) -> dict | None:
    for i in range(tries):
        try:
            r = _sess().get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            time.sleep(2 ** i)
    return None

@dataclass
class DataCollector:
    start_date: str
    end_date: str
    listing_levels: Sequence[int] = (1, 2)
    custom_secids: Sequence[str] | None = None
    rate_limit: float = 0.15
    max_workers: int = 4
    _raw_df: pd.DataFrame | None = field(default=None, init=False)

    @staticmethod
    def _fetch_table(url: str) -> List[str]:
        co = Options()
        co.add_argument("--headless")
        co.add_argument("--disable-gpu")
        co.add_argument("--no-sandbox")
        drv = webdriver.Chrome(options=co)
        drv.get(url)
        try:
            WebDriverWait(drv, 20).until(EC.presence_of_element_located((By.TAG_NAME, "table")))
            html = drv.page_source
        finally:
            drv.quit()

        for tbl in pd.read_html(StringIO(html)):
            if {"Инструмент", "ISIN"}.issubset(tbl.columns):
                return tbl["Инструмент"].tolist()
        return []

    def _get_secids(self) -> List[str]:
        if self.custom_secids:
            return list(dict.fromkeys([s.upper() for s in self.custom_secids]))
        base = (
            "https://www.moex.com/s3122#/?"
            "bg%5B%5D='stock_tplus','stock_d_tplus'&"
            "sec_type%5B%5D='stock_common_share','stock_preferred_share',"
            "'stock_russian_depositary_receipt','stock_foreign_share','stock_foreign_share_dr'&"
            "listname%5B%5D='{}'"
        )
        out: List[str] = []
        for lvl in self.listing_levels:
            with _PRINT:
                print(f"[Listing {lvl}] tickers…")
            out.extend(self._fetch_table(base.format(lvl)))
        return out

    @staticmethod
    def _board_info(secid: str) -> tuple[str | None, str | None]:
        url = f"https://iss.moex.com/iss/securities/{secid}.json"
        try:
            jd = _sess().get(url, timeout=30).json()
            df = pd.DataFrame(jd["boards"]["data"], columns=jd["boards"]["columns"])
            prior = ["TQBR", "TQTF", "TQOB", "TQCB"]
            board = next((b for b in prior if b in df["boardid"].values), None)
            if board is None and not df.empty:
                board = df.iloc[0]["boardid"]
            market = df.iloc[0]["market"] if not df.empty else None
            return market, board
        except Exception:
            return None, None

    def _hist(self, secid: str) -> List[Dict]:
        mkt, brd = self._board_info(secid)
        if not mkt or not brd:
            with _PRINT:
                print(f"→ {secid}: skip (board)")
            return []
        base = f"https://iss.moex.com/iss/history/engines/stock/markets/{mkt}/boards/{brd}/securities/{secid}.json"
        params = {"from": self.start_date, "till": self.end_date, "start": 0}
        out: List[Dict] = []

        while True:
            jd = _get_json(base, params)
            if jd is None:
                with _PRINT:
                    print(f"→ {secid}: partial ({len(out)})")
                break
            rows, cols = jd["history"]["data"], jd["history"]["columns"]
            if not rows:
                break
            for r in rows:
                d = dict(zip(cols, r))
                out.append(
                    {
                        "TRADEDATE": d["TRADEDATE"],
                        "SECID": secid,
                        "MARKETPRICE3": d["MARKETPRICE3"],
                        "CLOSE": d["CLOSE"],
                        "VOLUME": d["VOLUME"],
                    }
                )
            if len(rows) < 100:
                break
            params["start"] += len(rows)
            time.sleep(self.rate_limit)
        with _PRINT:
            print(f"→ {secid}: ok ({len(out)})")
        return out

    def collect(self) -> pd.DataFrame:
        bag: List[Dict] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self._hist, s): s for s in self._get_secids()}
            for f in as_completed(futures):
                bag.extend(f.result())
        self._raw_df = pd.DataFrame(bag)
        return self._raw_df

    @staticmethod
    def _wrap_exc(s: pd.Series) -> pd.Series:
        return s.apply(lambda x: f'="{x}"' if x != "" else "")

    def save_csv(self, path="history.csv"):
        if self._raw_df is None:
            raise RuntimeError("collect() first")
        df = self._raw_df.copy()
        for c in ["MARKETPRICE3", "CLOSE"]:
            df[c] = self._wrap_exc(df[c])
        df.to_csv(path, sep=";", index=False, encoding="utf-8-sig")
        with _PRINT:
            print(f"CSV → {path}")
