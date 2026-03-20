import tensorflow as tf
import keras
from keras import layers, Model, Input


def build_lstm_baseline(window_size: int = 30,
                         n_features: int = 24,
                         lstm_units: int = 100,
                         dense_units: list = None,
                         dropout_rate: float = 0.5,
                         learning_rate: float = 1e-3) -> Model:
    """
    Baseline LSTM model for single-domain RUL regression.

    This is the SOURCE-ONLY / TARGET-ONLY architecture referenced throughout
    the paper. It establishes the performance ceiling (TARGET-ONLY, trained
    on the same domain) and the unadapted baseline (SOURCE-ONLY, applied
    cross-domain without any adaptation).

    Architecture:
        Input(window_size, n_features)
        → LSTM(100)
        → Dropout(0.5)
        → Dense(30, ReLU)
        → Dropout(0.1)
        → Dense(20, ReLU)
        → Dense(1)           ← RUL scalar output

    Args:
        window_size    : Length of input time window (T_w)
        n_features     : Number of input features (sensors + op_settings)
        lstm_units     : Number of LSTM cells
        dense_units    : List of hidden layer sizes after LSTM
        dropout_rate   : Dropout fraction after LSTM
        learning_rate  : Adam optimiser learning rate

    Returns:
        Compiled Keras Model
    """
    if dense_units is None:
        dense_units = [30, 20]

    inp = Input(shape=(window_size, n_features), name='sensor_input')

    x = layers.LSTM(lstm_units, return_sequences=False, name='lstm_1')(inp)
    x = layers.Dropout(dropout_rate, name='dropout_lstm')(x)

    for i, units in enumerate(dense_units):
        x = layers.Dense(units, activation='relu', name=f'dense_{i+1}')(x)
        drop = 0.1 if i < len(dense_units) - 1 else 0.0
        if drop > 0:
            x = layers.Dropout(drop, name=f'dropout_dense_{i+1}')(x)

    output = layers.Dense(1, name='rul_output')(x)

    model = Model(inputs=inp, outputs=output, name='LSTM_Baseline')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    return model
