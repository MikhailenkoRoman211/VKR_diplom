import argparse
from pathlib import Path

import joblib
import optuna
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from data_collector import DataCollector
from data_processor import DataProcessor

DEFAULT_TRIALS = 500


def build_dataset(start="2015-01-01", end="2021-12-31"):
    dc = DataCollector(start, end, listing_levels=[1])
    raw = dc.collect()

    proc = DataProcessor(n_lags=5, ma_windows=(5, 10), vol_windows=(5,))
    X, y = proc.transform(raw)
    return train_test_split(X, y, test_size=0.2, shuffle=False)


def objective(trial, X_tr, X_val, y_tr, y_val):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": trial.suggest_float("lr", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("leaves", 31, 255, step=32),
        "min_data_in_leaf": trial.suggest_int("min_leaf", 20, 150),
        "feature_fraction": trial.suggest_float("ff", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bf", 0.6, 1.0),
        "bagging_freq": 1,
        "verbosity": -1,
    }
    model = lgb.LGBMRegressor(**params, n_estimators=2000)

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    preds = model.predict(X_val, num_iteration=model.best_iteration_)
    r2 = r2_score(y_val, preds)
    return -r2


def tune(trials: int):
    print("[Tuner] строим датасет… (1-2 мин)")
    X_tr, X_val, y_tr, y_val = build_dataset()

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda t: objective(t, X_tr, X_val, y_tr, y_val),
        n_trials=trials,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_r2 = -study.best_value

    print("\nЛучшие параметры:", best_params)
    print("Лучший R²:", round(best_r2, 4))

    joblib.dump(best_params, "best_params.pkl")
    print("Сохранено → best_params.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS,
                        help=f"Число попыток Optuna (по умолчанию {DEFAULT_TRIALS})")
    args = parser.parse_args()

    tune(args.trials)
