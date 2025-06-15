from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class DataProcessor:
    n_lags: int = 5
    ma_windows: Sequence[int] = (5, 10)
    vol_windows: Sequence[int] = (5,)
    price_col: str = "CLOSE"
    volume_col: str = "VOLUME"

    def transform(self, df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        df = df_raw.copy()
        df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
        df.sort_values(["SECID", "TRADEDATE"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        df["return"] = (
            df.groupby("SECID")[self.price_col]
            .pct_change(fill_method=None)
        )
        df["log_volume"] = np.log1p(df[self.volume_col])
        df["price_lag1"] = df.groupby("SECID")[self.price_col].shift(1)

        for lag in range(1, self.n_lags + 1):
            df[f"return_lag{lag}"] = df.groupby("SECID")["return"].shift(lag)
            df[f"vol_lag{lag}"] = df.groupby("SECID")[self.volume_col].shift(lag)

        for w in self.ma_windows:
            df[f"ret_ma{w}"] = (
                df.groupby("SECID")["return"].rolling(w).mean().reset_index(level=0, drop=True)
            )
        for w in self.vol_windows:
            df[f"ret_std{w}"] = (
                df.groupby("SECID")["return"].rolling(w).std().reset_index(level=0, drop=True)
            )

        df["dow"] = df["TRADEDATE"].dt.dayofweek
        df["month"] = df["TRADEDATE"].dt.month
        df["target"] = df.groupby("SECID")["return"].shift(-1)
        df["SECID_code"] = df["SECID"].astype("category").cat.codes.astype("int16")

        feature_cols = [
            c for c in df.columns
            if c not in {"TRADEDATE", "SECID", "MARKETPRICE3", "target"}
        ]
        df_feat = df[feature_cols + ["target"]]
        df_feat = df_feat.dropna(subset=["target"])
        df_feat = df_feat.dropna()

        X = df_feat.drop(columns=["target"]).astype("float32")
        X["SECID_code"] = df_feat["SECID_code"]
        y = df_feat["target"].astype("float32")

        return X, y
