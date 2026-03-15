# src/data_loader.py

import pandas as pd
import numpy as np

CMAPSS_COLUMNS = (
    ['unit_id', 'cycle'] +
    [f'op_setting_{i}' for i in range(1, 4)] +
    [f'sensor_{i}' for i in range(1, 22)]
)

def load_cmapss(dataset_id: str, split: str = 'train', data_dir: str = 'data/raw') -> pd.DataFrame:
    """
    Load a C-MAPSS dataset by ID (e.g. 'FD001') and split ('train'/'test').
    Assigns standardised column names.
    """
    path = f"{data_dir}/{split}_{dataset_id}.txt"
    df = pd.read_csv(path, sep=r'\s+', header=None, names=CMAPSS_COLUMNS)
    df['dataset_id'] = dataset_id
    return df

def load_rul_labels(dataset_id: str, data_dir: str = 'data/raw') -> np.ndarray:
    """Load ground truth RUL values for the test split."""
    path = f"{data_dir}/RUL_{dataset_id}.txt"
    return pd.read_csv(path, header=None).values.flatten()