from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(slots=True)
class WindowedDataset:
    features: np.ndarray
    targets: np.ndarray
    unit_ids: np.ndarray
    end_cycles: np.ndarray


def create_sliding_windows(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str = "RUL",
    window_size: int = 30,
    stride: int = 1,
    unit_column: str = "unit_id",
    cycle_column: str = "cycle",
) -> WindowedDataset:
    """Convert an engine-level sequence table into fixed-length windows."""
    features: list[np.ndarray] = []
    targets: list[float] = []
    unit_ids: list[int] = []
    end_cycles: list[int] = []

    for unit_id, unit_frame in frame.groupby(unit_column):
        ordered = unit_frame.sort_values(cycle_column)
        feature_values = ordered.loc[:, feature_columns].to_numpy(dtype=np.float32)
        target_values = ordered.loc[:, target_column].to_numpy(dtype=np.float32)
        cycles = ordered.loc[:, cycle_column].to_numpy(dtype=np.int32)

        if len(ordered) < window_size:
            continue

        for start in range(0, len(ordered) - window_size + 1, stride):
            stop = start + window_size
            features.append(feature_values[start:stop])
            targets.append(float(target_values[stop - 1]))
            unit_ids.append(int(unit_id))
            end_cycles.append(int(cycles[stop - 1]))

    if not features:
        empty_features = np.empty((0, window_size, len(feature_columns)), dtype=np.float32)
        empty_vector = np.empty((0,), dtype=np.float32)
        empty_ids = np.empty((0,), dtype=np.int32)
        return WindowedDataset(empty_features, empty_vector, empty_ids, empty_ids)

    return WindowedDataset(
        features=np.stack(features).astype(np.float32),
        targets=np.asarray(targets, dtype=np.float32),
        unit_ids=np.asarray(unit_ids, dtype=np.int32),
        end_cycles=np.asarray(end_cycles, dtype=np.int32),
    )
