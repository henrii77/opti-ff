import math
import statistics
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional
import pandas as pd


def _module_dir() -> Path:
    """Return the directory this file lives in.

    This lets us load data files with paths relative to the library file,
    regardless of the current working directory when students run the script.
    """
    return Path(__file__).resolve().parent


def load_data(base_dir: Optional[Path] = None, filename: str = "vol_lab.parquet") -> pd.DataFrame:
    """Load the stock microstructure data and return a sorted DataFrame.

    Columns:
    - stock_id: identifier for the stock
    - time_id: which half-hour window the row belongs to
    - seconds_in_bucket: seconds since the start of the half-hour (0..1800)
    - bid_price, ask_price, bid_size, ask_size, weighted_average_price
    - log_return: log-price change from previous row (NaN for first row)
    """
    base = _module_dir() if base_dir is None else Path(base_dir)
    df = pd.read_parquet(base / filename)
    return df.sort_values(["stock_id", "time_id", "seconds_in_bucket"]).reset_index(drop=True)


def load_targets(base_dir: Optional[Path] = None, filename: str = "target_vars.csv") -> pd.Series:
    """Load known target variances and return as a Series indexed by (stock_id, time_id)."""
    base = _module_dir() if base_dir is None else Path(base_dir)
    tdf = pd.read_csv(base / filename)
    tdf["target"] = tdf["target"].astype(float)
    return tdf.set_index(["stock_id", "time_id"])["target"].sort_index()



# ----- Evaluation helpers -----

def rmse(actual: List[float], predicted: List[float]) -> float:
    """Root Mean Squared Error between two equally sized lists."""
    return math.sqrt(statistics.mean((a - p) ** 2 for a, p in zip(actual, predicted)))


def score(
    predicted: Mapping[int, float] | pd.Series,
    targets: pd.Series,
    plot: bool = False,
    log_axes: bool = False,
):
    if not isinstance(predicted, pd.Series):
        predicted = pd.Series(predicted, dtype=float)
    predicted = predicted.sort_index()
    actual = targets.sort_index()
    common = actual.index.intersection(predicted.index)
    actual_vals = actual.loc[common].to_list()
    predicted_vals = predicted.loc[common].to_list()
    score = rmse([x for x in actual_vals], [x for x in predicted_vals])
    return score


def evaluate_predictions(
    predicted: Mapping[int, float] | pd.Series,
    targets: pd.Series,
    plot: bool = False,
    log_axes: bool = False,
) -> float:
    """Compare predictions against known targets. Returns RMSE (lower is better)."""
    if not isinstance(predicted, pd.Series):
        predicted = pd.Series(predicted, dtype=float)
    predicted = predicted.sort_index()
    actual = targets.sort_index()
    common = actual.index.intersection(predicted.index)
    actual_vals = actual.loc[common].to_list()
    predicted_vals = predicted.loc[common].to_list()
    score = rmse([x for x in actual_vals], [x for x in predicted_vals])
    if plot:
        try:
            import matplotlib.pyplot as plt
            dfp = pd.DataFrame({"actual": actual_vals, "predicted": predicted_vals})
            plt.figure()
            plt.scatter(dfp["predicted"], dfp["actual"], alpha=0.6, s=plt.rcParams["lines.markersize"] ** 2 / 4)
            min_val = min(dfp["predicted"].min(), dfp["actual"].min())
            max_val = max(dfp["predicted"].max(), dfp["actual"].max())
            plt.plot([min_val, max_val], [min_val, max_val], "r--", label="y=x")
            if log_axes:
                plt.xscale("log")
                plt.yscale("log")
            plt.title(f"RMSE = {score:.7f}")
            plt.xlabel("predicted variance")
            plt.ylabel("actual variance")
            plt.legend()
        except Exception:
            pass
    return round(score, 7)

