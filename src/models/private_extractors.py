import tensorflow as tf
import keras
from keras import layers, Model, Input
from typing import Dict


def build_private_extractor(n_input_sensors: int,
                            window_size: int = 30,
                            hidden_dim: int = 128,
                            output_dim: int = 64,
                            dropout_rate: float = 0.2,
                            name: str = 'private_extractor') -> Model:
    inp = Input(shape=(window_size, n_input_sensors), name=f'{name}_input')
    x = layers.Conv1D(hidden_dim, kernel_size=3,
                      activation='relu', padding='same',
                      name=f'{name}_conv')(inp)
    x = layers.Dropout(dropout_rate, name=f'{name}_dropout')(x)
    x = layers.GlobalAveragePooling1D(name=f'{name}_gap')(x)
    out = layers.Dense(output_dim, activation='relu',
                       name=f'{name}_output')(x)

    return Model(inputs=inp, outputs=out, name=name)


def build_hda_extractor_registry(machine_configs: Dict[str, int],
                                 window_size: int = 30,
                                 output_dim: int = 64) -> Dict[str, Model]:
    registry = {}
    for machine_name, n_sensors in machine_configs.items():
        registry[machine_name] = build_private_extractor(
            n_input_sensors=n_sensors,
            window_size=window_size,
            output_dim=output_dim,
            name=f'{machine_name}_extractor'
        )
        print(f"  ✅ Private extractor: {machine_name} ({n_sensors} sensors → {output_dim} features)")
    return registry
