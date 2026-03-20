import tensorflow as tf
import keras
from keras import layers, Model, Input
from .grl import GradientReversalLayer


def build_cnn_lstm_dann(window_size: int = 30,
                         n_features: int = 24,
                         filters: list = None,
                         kernel_size: int = 3,
                         lstm_units: int = 128,
                         feature_dim: int = 64,
                         reg_units: list = None,
                         domain_units: list = None,
                         dropout_rate: float = 0.3,
                         reg_dropout: float = 0.2,
                         dom_dropout: float = 0.2,
                         alpha: float = 1.0) -> tuple:
    """
    CNN-LSTM Domain Adversarial Neural Network.

    Upgrade from pure LSTM-DANN:
      CNN layers extract spatial sensor correlations (local patterns)
      LSTM models temporal evolution of those patterns
      GRL forces features to be domain-invariant (source vs target)

    Architecture:
        ┌─ Input ──────────────────────────────────────────────┐
        │  Conv1D(64) → Conv1D(64) → MaxPool → Dropout        │  Feature
        │  LSTM(128) → Dropout                                 │  Extractor g_f
        │  Dense(feature_dim, ReLU)  ← shared feature space f │
        └──────────────────────────────────────────────────────┘
                 │                          │
                 ▼                          ▼
        ┌─ RUL Regressor g_y ─┐   ┌─ GRL ─────────────────────┐
        │  Dense(reg_units)   │   │  Domain Classifier g_d     │
        │  Dense(1)           │   │  Dense(domain_units)       │
        │  RUL output         │   │  Dense(1, Sigmoid)         │
        └─────────────────────┘   └────────────────────────────┘

    Args:
        window_size   : Must match windowing step
        n_features    : After FeatureAligner — always 24 for this project
        filters       : Conv1D filters per layer
        kernel_size   : Conv1D kernel size
        lstm_units    : LSTM hidden size
        feature_dim   : Shared embedding dimension
        reg_units     : RUL regressor hidden layers
        domain_units  : Domain classifier hidden layers
        dropout_rate  : Dropout in feature extractor
        reg_dropout   : Dropout in RUL head
        dom_dropout   : Dropout in domain classifier head
        alpha         : GRL reversal strength

    Returns:
        (regression_model, adversarial_model)
        regression_model  → used for inference (Input → RUL)
        adversarial_model → used for training (Input → [RUL, domain])
    """
    if filters is None:
        filters = [64, 64]
    if reg_units is None:
        reg_units = [64, 32]
    if domain_units is None:
        domain_units = [32]

    sensor_input = Input(shape=(window_size, n_features), name='sensor_input')

    # ── Feature Extractor g_f ─────────────────────────────────────────────────
    x = sensor_input
    for i, f in enumerate(filters):
        x = layers.Conv1D(f, kernel_size=kernel_size,
                           activation='relu', padding='same',
                           name=f'conv1d_{i+1}')(x)
    x = layers.MaxPooling1D(pool_size=2, name='maxpool')(x)
    x = layers.Dropout(dropout_rate, name='cnn_dropout')(x)
    x = layers.LSTM(lstm_units, return_sequences=False, name='lstm_1')(x)
    x = layers.Dropout(dropout_rate, name='lstm_dropout')(x)
    features = layers.Dense(feature_dim, activation='relu',
                             name='feature_layer')(x)

    # ── RUL Regressor g_y ─────────────────────────────────────────────────────
    ry = features
    for i, units in enumerate(reg_units):
        ry = layers.Dense(units, activation='relu', name=f'reg_dense_{i+1}')(ry)
        ry = layers.Dropout(reg_dropout, name=f'reg_drop_{i+1}')(ry)
    rul_output = layers.Dense(1, name='rul_output')(ry)

    # ── Domain Classifier g_d via GRL ─────────────────────────────────────────
    grl_out = GradientReversalLayer(alpha=alpha, name='grl')(features)
    dy = grl_out
    for i, units in enumerate(domain_units):
        dy = layers.Dense(units, activation='relu', name=f'dom_dense_{i+1}')(dy)
        dy = layers.Dropout(dom_dropout, name=f'dom_drop_{i+1}')(dy)
    domain_output = layers.Dense(1, activation='sigmoid',
                                   name='domain_output')(dy)

    regression_model  = Model(inputs=sensor_input, outputs=rul_output,
                               name='CNN_LSTM_DANN_Regressor')
    adversarial_model = Model(inputs=sensor_input,
                               outputs=[rul_output, domain_output],
                               name='CNN_LSTM_DANN_Full')

    return regression_model, adversarial_model


def get_cnn_lstm_feature_extractor(adversarial_model: Model) -> Model:
    """Extract the feature extractor sub-model for t-SNE / SHAP analysis."""
    return Model(
        inputs=adversarial_model.input,
        outputs=adversarial_model.get_layer('feature_layer').output,
        name='CNN_LSTM_Feature_Extractor'
    )