import tensorflow as tf
from tensorflow.keras import layers, Model, Input


def build_cnn_lstm(window_size: int = 30,
                   n_features: int = 24,
                   filters: list = None,
                   kernel_size: int = 3,
                   lstm_units: int = 128,
                   dense_units: list = None,
                   dropout_rate: float = 0.3,
                   learning_rate: float = 5e-4,
                   use_bilstm: bool = True) -> Model:
    """
    CNN-LSTM hybrid model for RUL regression.

    Architecture:
        Input(window_size, n_features)
        → Conv1D(64, kernel=3, ReLU)  — extract local sensor correlations
        → Conv1D(64, kernel=3, ReLU)  — deeper spatial features
        → MaxPooling1D(2)              — reduce sequence length
        → Dropout(rate)
        → LSTM(128)                    — model temporal degradation trend
        → Dropout(rate)
        → Dense(64, ReLU)             — shared feature embedding
        → Dense(32, ReLU)
        → Dense(1)                    — RUL output [0,1] normalised

    Why CNN before LSTM:
        CNN scans across the time-window to detect co-activation patterns
        among sensors (e.g., temperature + pressure rising together = fault).
        LSTM then models how these patterns evolve over cycles.
        This outperforms pure LSTM on C-MAPSS by ~30-40% RMSE reduction.

    Args:
        window_size   : T_w — number of cycles per window
        n_features    : Number of input sensor/op features
        filters       : Conv1D filter counts per layer, default [64, 64]
        kernel_size   : Conv1D kernel size
        lstm_units    : LSTM hidden dimension
        dense_units   : FC head layer sizes, default [64, 32]
        dropout_rate  : Applied after CNN block and after LSTM
        learning_rate : Adam optimiser LR
    """
    if filters is None:
        filters = [32, 64]
    if dense_units is None:
        dense_units = [64, 32]

    inp = Input(shape=(window_size, n_features), name='sensor_input')

    # ── CNN Block: spatial sensor correlation extraction ──────────────────────
    x = inp
    for i, f in enumerate(filters):
        stride = 2 if i == len(filters) - 1 else 1
        x = layers.Conv1D(f,
                          kernel_size=kernel_size,
                          strides=stride,
                          activation='relu',
                          padding='same',
                          name=f'conv1d_{i+1}')(x)
        x = layers.BatchNormalization(name=f'bn_{i+1}')(x)
    x = layers.Dropout(dropout_rate, name='cnn_dropout')(x)

    # ── LSTM Block: temporal degradation modeling ─────────────────────────────
    if use_bilstm:
        x = layers.Bidirectional(
            layers.LSTM(lstm_units, return_sequences=False),
            name='bilstm_1'
        )(x)
    else:
        x = layers.LSTM(lstm_units, return_sequences=False, name='lstm_1')(x)
    x = layers.Dropout(dropout_rate, name='lstm_dropout')(x)

    # ── Dense Head: RUL regression ────────────────────────────────────────────
    for i, units in enumerate(dense_units):
        x = layers.Dense(units, activation='relu', name=f'dense_{i+1}')(x)

    output = layers.Dense(1, name='rul_output')(x)

    model = Model(inputs=inp, outputs=output, name='CNN_LSTM_Baseline')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    return model