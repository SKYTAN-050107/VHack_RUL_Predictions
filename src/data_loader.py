# src/data_loader.py

import pandas as pd
import numpy as np

# Standard column names for all C-MAPSS datasets
CMAPSS_COLUMNS = (
    ['unit_id', 'cycle'] +
    [f'op_setting_{i}' for i in range(1, 4)] +
    [f'sensor_{i}' for i in range(1, 22)]
)

# All feature columns used as model input
FEATURE_COLS = (
    [f'op_setting_{i}' for i in range(1, 4)] +
    [f'sensor_{i}' for i in range(1, 22)]
)

SENSOR_COLS = [f'sensor_{i}' for i in range(1, 22)]


def load_cmapss(dataset_id: str, split: str = 'train',
                data_dir: str = 'data/raw') -> pd.DataFrame:
    """
    Load a C-MAPSS dataset by ID (e.g. 'FD001') and split ('train' or 'test').

    Args:
        dataset_id : One of 'FD001', 'FD002', 'FD003', 'FD004'
        split      : 'train' or 'test'
        data_dir   : Path to the directory containing the raw .txt files

    Returns:
        pd.DataFrame with standardised column names and a 'dataset_id' column
    """
    path = f"{data_dir}/{split}_{dataset_id}.txt"
    df = pd.read_csv(
        path,
        sep=r'\s+',
        header=None,
        names=CMAPSS_COLUMNS,
        engine='python'
    )
    df['dataset_id'] = dataset_id
    return df


def load_rul_labels(dataset_id: str, data_dir: str = 'data/raw') -> np.ndarray:
    """
    Load ground-truth RUL values for the test split of a C-MAPSS dataset.

    Returns:
        1D numpy array of RUL values, one per test engine unit
    """
    path = f"{data_dir}/RUL_{dataset_id}.txt"
    return pd.read_csv(path, header=None).values.flatten()


def load_all_datasets(data_dir: str = 'data/raw') -> dict:
    """
    Convenience function: load all four C-MAPSS datasets at once.

    Returns:
        dict with keys 'FD001' ... 'FD004', each containing:
            'train' : DataFrame
            'test'  : DataFrame
            'rul'   : np.ndarray of test RUL labels
    """
    datasets = {}
    for ds_id in ['FD001', 'FD002', 'FD003', 'FD004']:
        datasets[ds_id] = {
            'train': load_cmapss(ds_id, 'train', data_dir),
            'test':  load_cmapss(ds_id, 'test',  data_dir),
            'rul':   load_rul_labels(ds_id, data_dir)
        }
    return datasets