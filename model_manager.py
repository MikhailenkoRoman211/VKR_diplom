from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


class ModelManager:
    def __init__(self, params: Dict | None = None):
        if params is None and Path("best_params.pkl").exists():
            params = joblib.load("best_params.pkl")
            print("[ModelManager] loaded best_params.pkl")

        self.params = params or {}
        self.model: lgb.LGBMRegressor | None = None
        self.r2_: float | None = None
        self.rmse_: float | None = None

    def _default_params(self) -> Dict:
        base = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 40,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "n_estimators": 2000,
            "verbosity": -1,
        }
        base.update(self.params)
        return base

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        self.model = lgb.LGBMRegressor(**self._default_params())
        self.model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[
                lgb.early_stopping(200),
                lgb.log_evaluation(100),
            ],
        )

        preds = self.model.predict(X_val, num_iteration=self.model.best_iteration_)
        self.r2_ = r2_score(y_val, preds)
        self.rmse_ = float(np.sqrt(((y_val - preds) ** 2).mean()))

        print(
            f"[Model] best_iter={self.model.best_iteration_} "
            f"R²={self.r2_:.4f}  RMSE={self.rmse_:.5f}"
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Модель не обучена")
        return self.model.predict(X, num_iteration=self.model.best_iteration_)

    def save(self, path: str | Path = "model.pkl") -> None:
        if self.model is None:
            raise RuntimeError("Сначала обучите модель")
        joblib.dump({"model": self.model, "r2": self.r2_, "rmse": self.rmse_}, path)
        with open(Path(path).with_suffix(".metrics.json"), "w", encoding="utf-8") as f:
            json.dump({"r2": self.r2_, "rmse": self.rmse_}, f, indent=2)
        print(f"[Model] saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "ModelManager":
        data = joblib.load(path)
        mgr = cls()
        mgr.model = data["model"]
        mgr.r2_ = data.get("r2")
        mgr.rmse_ = data.get("rmse")
        print(f"[Model] loaded ← {path}")
        return mgr
