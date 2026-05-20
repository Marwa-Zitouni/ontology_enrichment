from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# Physical-range scaling, identical convention to data.real_data.
RANGES = np.array(
    [[0.0, 100.0],     # temperature [degC]
     [-150.0, 150.0],  # current     [A]
     [2.5, 4.5]],      # voltage     [V]
    dtype=np.float32,
)



_ID_TO_TYPE = {
    "A1": "overdischarge",     # voltage drop
    "A2": "thermal_runaway",   # large T ramp
    "A3": "overcurrent",       # current spike
    "A4": "overcharge",        # noisy V (placeholder; expected to be filtered)
    "A5": "overheating",       # slow T drift
    "A6": "thermal_runaway",   # missing T => recon error spike
    "A7": "overdischarge",     # capacity inconsistency
}


def _scale(window_phys: np.ndarray) -> np.ndarray:
    lo = RANGES[:, 0]
    span = RANGES[:, 1] - RANGES[:, 0]
    out = (window_phys - lo) / span
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _impute_nan_linear(arr: np.ndarray) -> np.ndarray:
    """Replace NaNs with linear interpolation along the time axis."""
    if not np.isnan(arr).any():
        return arr
    mask = np.isfinite(arr)
    if not mask.any():
        return np.zeros_like(arr)
    idx = np.arange(len(arr))
    return np.interp(idx, idx[mask], arr[mask]).astype(np.float32)


def _resample_to(window: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly resample a (T, C) window to (target_len, C)."""
    T, C = window.shape
    if T == target_len:
        return window.astype(np.float32)
    xs_old = np.linspace(0.0, 1.0, T)
    xs_new = np.linspace(0.0, 1.0, target_len)
    out = np.stack(
        [np.interp(xs_new, xs_old, window[:, c]) for c in range(C)],
        axis=-1,
    )
    return out.astype(np.float32)


def load_mit_dataset(
    final_dir: os.PathLike | str | None = None,
    seq_len: int = 50,
    standardize: bool = True,
) -> dict:
    
    if final_dir is None:
        # Default: parquet bundled alongside the repository.
        here = Path(__file__).resolve().parent
        final_dir = here / "mit_batch1_final"
    final_dir = Path(final_dir)

    ts_path = final_dir / "timeseries_final.parquet"
    cyc_path = final_dir / "cycles_final.parquet"
    if not ts_path.exists() or not cyc_path.exists():
        raise FileNotFoundError(
            f"MIT case-study parquet missing under {final_dir}. "
            f"Expected timeseries_final.parquet and cycles_final.parquet."
        )

    ts_df = pd.read_parquet(ts_path)
    cyc_df = pd.read_parquet(cyc_path)

    grouped = ts_df.sort_values(["cell_id", "cycle", "step"]).groupby(
        ["cell_id", "cycle"], sort=False
    )

    samples_phys: list[np.ndarray] = []
    labels: list[int] = []
    anomaly_types: list[str] = []
    anomaly_ids: list[str] = []
    cell_ids: list[str] = []
    cycles_meta: list[int] = []

    for (cell_id, cyc), g in grouped:
        g = g.sort_values("step")
        T = _impute_nan_linear(g["T"].to_numpy(dtype=np.float64))
        I = _impute_nan_linear(g["I"].to_numpy(dtype=np.float64))
        V = _impute_nan_linear(g["V"].to_numpy(dtype=np.float64))
        window_phys = np.stack([T, I, V], axis=-1).astype(np.float32)
        window_phys = _resample_to(window_phys, seq_len)
        samples_phys.append(window_phys)

        is_anom = bool(g["is_anomaly"].any())
        a_id = next((s for s in g["anomaly_id"].unique() if s), "")
        labels.append(1 if is_anom else 0)
        anomaly_ids.append(a_id)
        anomaly_types.append(_ID_TO_TYPE.get(a_id, "normal") if is_anom else "normal")
        cell_ids.append(cell_id)
        cycles_meta.append(int(cyc))

    ts_phys = np.stack(samples_phys, axis=0)
    ts_data = _scale(ts_phys) if standardize else ts_phys.astype(np.float32)

    n_normal = sum(1 for y in labels if y == 0)
    n_anom = sum(1 for y in labels if y == 1)
    print(
        f"Loaded MIT case-study dataset: ts={ts_data.shape}, "
        f"normal={n_normal}, anomalous={n_anom}"
    )

    return {
        "ts_data":       ts_data,
        "labels":        np.asarray(labels, dtype=np.int64),
        "anomaly_types": anomaly_types,
        "anomaly_ids":   anomaly_ids,
        "cell_ids":      cell_ids,
        "cycles":        cycles_meta,
        "ts_raw":        ts_phys,
    }


if __name__ == "__main__":
    ds = load_mit_dataset()
    print("anomaly types:", set(ds["anomaly_types"]))
    print("first window shape:", ds["ts_data"][0].shape)
