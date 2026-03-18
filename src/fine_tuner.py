import numpy as np
import tensorflow as tf
import os


class FewShotFineTuner:
    """
    Fine-tunes a pretrained LSTM model on a small labelled sample
    from a new machine type.

    Strategy (standard transfer learning):
      Phase 1 — Freeze everything except the final regressor head.
                 Train only the Dense output layers for a few epochs.
                 This prevents catastrophic forgetting on tiny datasets.

      Phase 2 — Unfreeze all layers and train end-to-end with a
                 very low learning rate. This lets the LSTM adapt its
                 temporal patterns to the new machine's degradation signature.

    When to use each phase:
      - If you have < 20 labelled cycles  → Phase 1 only (set phase2_epochs=0)
      - If you have 20–100 labelled cycles → Phase 1 + Phase 2
      - If you have > 100 labelled cycles  → Full retraining (run Notebook 04)

    Args:
        base_model    : Pretrained Keras LSTM model (from build_lstm_baseline)
        freeze_layers : Layer name prefixes to freeze in Phase 1
                        Default freezes LSTM layers, keeps Dense trainable
    """

    def __init__(self, base_model, freeze_layers: list = None):
        self.model         = base_model
        self.freeze_layers = freeze_layers or ['lstm']
        self._orig_weights = base_model.get_weights()

    def _freeze(self, freeze: bool):
        """Freeze or unfreeze layers whose names start with any prefix in self.freeze_layers."""
        for layer in self.model.layers:
            if any(layer.name.startswith(p) for p in self.freeze_layers):
                layer.trainable = not freeze

    def fine_tune(self,
                   X_new: np.ndarray,
                   y_new: np.ndarray,
                   X_val: np.ndarray = None,
                   y_val: np.ndarray = None,
                   phase1_epochs: int   = 30,
                   phase2_epochs: int   = 20,
                   phase1_lr:     float = 1e-3,
                   phase2_lr:     float = 1e-4,
                   batch_size:    int   = 16) -> dict:
        """
        Run two-phase fine-tuning.

        Args:
            X_new         : New machine windows  (n, window_size, n_features)
            y_new         : RUL labels            (n,)
            X_val         : Optional validation windows
            y_val         : Optional validation labels
            phase1_epochs : Epochs for head-only training
            phase2_epochs : Epochs for full fine-tune (0 to skip Phase 2)
            phase1_lr     : Learning rate Phase 1
            phase2_lr     : Learning rate Phase 2 (should be 10x smaller than phase1_lr)
            batch_size    : Keep small for few-shot data (8–32 recommended)

        Returns:
            history dict with keys: phase1_loss, phase2_loss, val_mae
        """
        history  = {'phase1_loss': [], 'phase2_loss': [], 'val_mae': []}
        val_data = (X_val, y_val) if X_val is not None and len(X_val) > 0 else None

        # ── Phase 1: Head-only training ───────────────────────────────────────
        print(f"Phase 1: head-only ({phase1_epochs} epochs, lr={phase1_lr})")
        self._freeze(freeze=True)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=phase1_lr),
            loss='mse',
            metrics=['mae']
        )
        h1 = self.model.fit(
            X_new, y_new,
            validation_data=val_data,
            epochs=phase1_epochs,
            batch_size=batch_size,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if val_data else 'loss',
                patience=10,
                restore_best_weights=True
            )],
            verbose=0
        )
        history['phase1_loss'] = h1.history['loss']
        if 'val_mae' in h1.history:
            history['val_mae'] += h1.history['val_mae']
        print(f"  Phase 1 done. Final loss: {h1.history['loss'][-1]:.4f}")

        # ── Phase 2: Full fine-tune ───────────────────────────────────────────
        if phase2_epochs > 0:
            print(f"Phase 2: full fine-tune ({phase2_epochs} epochs, lr={phase2_lr})")
            self._freeze(freeze=False)
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=phase2_lr),
                loss='mse',
                metrics=['mae']
            )
            h2 = self.model.fit(
                X_new, y_new,
                validation_data=val_data,
                epochs=phase2_epochs,
                batch_size=batch_size,
                callbacks=[tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss' if val_data else 'loss',
                    patience=8,
                    restore_best_weights=True
                )],
                verbose=0
            )
            history['phase2_loss'] = h2.history['loss']
            if 'val_mae' in h2.history:
                history['val_mae'] += h2.history['val_mae']
            print(f"  Phase 2 done. Final loss: {h2.history['loss'][-1]:.4f}")

        # Re-enable all layers after training
        self._freeze(freeze=False)
        return history

    def save_adapted_model(self, path: str):
        """Save fine-tuned weights to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_weights(path)
        print(f"Adapted weights saved: {path}")

    def reset_to_base(self):
        """Restore the original pretrained weights (undo fine-tuning)."""
        self.model.set_weights(self._orig_weights)
        print("Model reset to original pretrained weights.")
