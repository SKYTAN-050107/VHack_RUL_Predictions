import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_pred - y_true)))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    NASA Prognostics Scoring Function (Equation 16 in the paper).

    Asymmetric penalty where OVER-predictions are penalised more harshly
    than UNDER-predictions, reflecting the real-world cost of falsely
    assuring operators that an engine has more life than it does.

        s = sum( exp(-c/13) - 1 )  if c < 0  (early prediction)
        s = sum( exp( c/10) - 1 )  if c >= 0 (late prediction)

    where c = predicted_RUL - true_RUL

    Args:
        y_true : Ground-truth RUL values
        y_pred : Model-predicted RUL values

    Returns:
        Scalar score (lower is better; 0 = perfect)
    """
    c  = y_pred - y_true
    a1, a2 = 13.0, 10.0
    scores = np.where(
        c < 0,
        np.exp(-c / a1) - 1,
        np.exp( c / a2) - 1
    )
    return float(np.sum(scores))


def evaluate_model(model,
                    X_test: np.ndarray,
                    y_true: np.ndarray,
                    model_name: str = 'Model',
                    batch_size: int = 256) -> dict:
    """
    Run inference and compute all evaluation metrics for a given model.

    Args:
        model      : Any model with a .predict(X) method
        X_test     : Input windows of shape (n_engines, window_size, n_features)
        y_true     : Ground-truth RUL values, shape (n_engines,)
        model_name : Label for the results dict
        batch_size : Prediction batch size

    Returns:
        dict with keys: model, RMSE, NASA_Score, MAE, y_pred
    """
    y_pred = model.predict(X_test, verbose=0).flatten()
    return {
        'model':       model_name,
        'RMSE':        round(rmse(y_true, y_pred), 2),
        'MAE':         round(mae(y_true, y_pred), 2),
        'NASA_Score':  round(nasa_score(y_true, y_pred), 1),
        'y_pred':      y_pred
    }


def compare_models(results_list: list) -> pd.DataFrame:
    """
    Combine a list of evaluate_model result dicts into a summary DataFrame.

    Args:
        results_list : List of dicts from evaluate_model()

    Returns:
        DataFrame sorted by RMSE ascending, without the y_pred column
    """
    df = pd.DataFrame(results_list)
    df = df.drop(columns='y_pred', errors='ignore')
    return df.sort_values('RMSE').reset_index(drop=True)
