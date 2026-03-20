import tensorflow as tf
import keras
from keras import layers, Model, Input
from .grl import GradientReversalLayer


def build_lstm_dann(window_size: int = 30,
                     n_features: int = 24,
                     lstm_units: int = 128,
                     lstm_layers: int = 1,
                     feature_dim: int = 64,
                     reg_units: list = None,
                     domain_units: list = None,
                     lstm_dropout: float = 0.5,
                     reg_dropout: float = 0.3,
                     dom_dropout: float = 0.3,
                     alpha: float = 0.8) -> tuple:
    """
    Build the LSTM-DANN architecture as described in Section 3.4 of the paper.

    Architecture (Figure 2 in paper):
        Shared Feature Extractor g_f:
            Input → LSTM(lstm_units) × lstm_layers → Dense(feature_dim, ReLU)

        RUL Regressor g_y  (θ_y):
            features → Dense(reg_units[0], ReLU) → ... → Dense(1)

        Domain Classifier g_d  (θ_d):
            features → GRL(alpha) → Dense(domain_units[0], ReLU) → ... → Dense(1, Sigmoid)

    Training behaviour:
        - g_f + g_y minimise the RUL regression loss (MAE) on SOURCE domain
        - g_f + GRL + g_d minimise domain binary cross-entropy on SOURCE + TARGET
        - GRL reverses gradients so g_f learns to MAXIMISE domain confusion
          while g_d continues to MINIMISE its own classification loss

    Args:
        window_size   : T_w (must match windowing step)
        n_features    : Number of input features
        lstm_units    : LSTM hidden size
        lstm_layers   : Number of stacked LSTM layers (1 or 2)
        feature_dim   : Size of the shared feature embedding layer
        reg_units     : Hidden layer sizes for the RUL regressor
        domain_units  : Hidden layer sizes for the domain classifier
        lstm_dropout  : Dropout fraction applied after each LSTM layer
        reg_dropout   : Dropout fraction in the regressor head
        dom_dropout   : Dropout fraction in the domain classifier head
        alpha         : GRL scaling factor (domain confusion strength)

    Returns:
        Tuple of two Keras Models sharing the same feature extractor weights:
            regression_model  : Input → RUL output  (used for inference)
            adversarial_model : Input → (RUL output, domain output)  (used for training)
    """
    if reg_units    is None: reg_units    = [32]
    if domain_units is None: domain_units = [32]

    # ── Shared Input ───────────────────────────────────────────────────────────
    sensor_input = Input(shape=(window_size, n_features), name='sensor_input')

    # ── Feature Extractor g_f ──────────────────────────────────────────────────
    x = sensor_input
    for i in range(lstm_layers):
        return_seq = (i < lstm_layers - 1)  # only last LSTM returns single vector
        x = layers.LSTM(
            lstm_units,
            return_sequences=return_seq,
            name=f'lstm_{i+1}'
        )(x)
        x = layers.Dropout(lstm_dropout, name=f'lstm_drop_{i+1}')(x)

    features = layers.Dense(feature_dim, activation='relu',
                             name='feature_layer')(x)   # shared embedding space f

    # ── RUL Regressor g_y ──────────────────────────────────────────────────────
    ry = features
    for i, units in enumerate(reg_units):
        ry = layers.Dense(units, activation='relu', name=f'reg_dense_{i+1}')(ry)
        ry = layers.Dropout(reg_dropout, name=f'reg_drop_{i+1}')(ry)
    rul_output = layers.Dense(1, name='rul_output')(ry)

    # ── Domain Classifier g_d (via GRL) ───────────────────────────────────────
    grl_out = GradientReversalLayer(alpha=alpha, name='grl')(features)
    dy = grl_out
    for i, units in enumerate(domain_units):
        dy = layers.Dense(units, activation='relu', name=f'dom_dense_{i+1}')(dy)
        dy = layers.Dropout(dom_dropout, name=f'dom_drop_{i+1}')(dy)
    # Sigmoid output → probability of being from TARGET domain (label=1)
    domain_output = layers.Dense(1, activation='sigmoid', name='domain_output')(dy)

    # ── Two Views of the Same Network ──────────────────────────────────────────
    # regression_model: used for prediction / evaluation
    regression_model = Model(
        inputs=sensor_input,
        outputs=rul_output,
        name='LSTM_DANN_Regressor'
    )

    # adversarial_model: used during training (both heads active)
    adversarial_model = Model(
        inputs=sensor_input,
        outputs=[rul_output, domain_output],
        name='LSTM_DANN_Full'
    )

    return regression_model, adversarial_model


def get_feature_extractor(adversarial_model: Model) -> Model:
    """
    Extract the feature extractor sub-model from a trained LSTM-DANN.
    Useful for t-SNE visualisation and SHAP analysis.

    Returns:
        Model: Input → feature_layer output (the shared embedding)
    """
    return Model(
        inputs=adversarial_model.input,
        outputs=adversarial_model.get_layer('feature_layer').output,
        name='Feature_Extractor'
    )
