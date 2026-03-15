import numpy as np
import tensorflow as tf
from tqdm import tqdm


class LSTMDANNTrainer:
    """
    Two-pass adversarial training loop for the LSTM-DANN model.

    Implements the gradient update equations (12)–(14) from the paper:

        θ_f  ←  θ_f  −  λ ( ∂L_y/∂θ_f  −  α · ∂L_d/∂θ_f )
        θ_y  ←  θ_y  −  λ ( ∂L_y/∂θ_y )
        θ_d  ←  θ_d  −  λ ( α · ∂L_d/∂θ_d )

    Each training step:
      Pass 1 — Compute RUL regression loss on source-domain mini-batch.
               Update feature extractor (θ_f) and regressor (θ_y).
      Pass 2 — Compute domain classification loss on concatenated
               source + target mini-batch. GRL handles the sign flip in θ_f.
               Update domain classifier (θ_d).

    Stopping criterion: early stopping on source-domain validation MAE,
    as described in Section 5.2 of the paper.

    Args:
        adversarial_model : The full LSTM-DANN Keras model
        alpha             : Domain loss weighting coefficient
        lr_reg            : Learning rate for the regression pass (SGD)
        lr_dom            : Learning rate for the domain classification pass (SGD)
    """

    def __init__(self,
                  adversarial_model,
                  alpha: float = 1.0,
                  lr_reg: float = 0.01,
                  lr_dom: float = 0.01):
        self.model   = adversarial_model
        self.alpha   = alpha
        # SGD with gradient clipping (clipnorm=1) to avoid exploding gradients
        self.reg_opt = tf.keras.optimizers.SGD(learning_rate=lr_reg,
                                                clipnorm=1.0)
        self.dom_opt = tf.keras.optimizers.SGD(learning_rate=lr_dom,
                                                clipnorm=1.0)
        self.rul_loss_fn = tf.keras.losses.MeanAbsoluteError()
        self.dom_loss_fn = tf.keras.losses.BinaryCrossentropy()

    def _get_feature_vars(self):
        """All trainable variables belonging to the feature extractor (θ_f)."""
        feature_layer_names = ['feature_layer'] + \
            [l.name for l in self.model.layers if 'lstm' in l.name]
        return [v for l in self.model.layers
                  if l.name in feature_layer_names
                  for v in l.trainable_variables]

    def _get_regressor_vars(self):
        """All trainable variables belonging to the RUL regressor (θ_y)."""
        return [v for l in self.model.layers
                  if 'reg_' in l.name or l.name == 'rul_output'
                  for v in l.trainable_variables]

    def _get_domain_vars(self):
        """All trainable variables belonging to the domain classifier (θ_d)."""
        return [v for l in self.model.layers
                  if 'dom_' in l.name or l.name == 'domain_output'
                  for v in l.trainable_variables]

    @tf.function
    def train_step(self, X_src, y_src, X_tgt):
        """
        One mini-batch adversarial update.

        Args:
            X_src : Source-domain sensor windows  (batch, Tw, F)
            y_src : Source-domain RUL labels       (batch,)
            X_tgt : Target-domain sensor windows   (batch, Tw, F)

        Returns:
            rul_loss : Scalar MAE on source RUL task
            dom_loss : Scalar BCE on source+target domain classification
        """
        batch = tf.shape(X_src)[0]
        # Domain labels: 0 = source, 1 = target
        d_src  = tf.zeros((batch, 1), dtype=tf.float32)
        d_tgt  = tf.ones((tf.shape(X_tgt)[0], 1), dtype=tf.float32)
        X_both = tf.concat([X_src, X_tgt], axis=0)
        d_both = tf.concat([d_src,  d_tgt],  axis=0)

        # ── Pass 1: Regression ─────────────────────────────────────────────
        with tf.GradientTape() as tape_reg:
            rul_pred, _ = self.model(X_src, training=True)
            rul_loss    = self.rul_loss_fn(y_src[:, tf.newaxis], rul_pred)

        reg_vars  = self._get_feature_vars() + self._get_regressor_vars()
        grads_reg = tape_reg.gradient(rul_loss, reg_vars)
        grads_reg = [g if g is not None else tf.zeros_like(v)
                     for g, v in zip(grads_reg, reg_vars)]
        self.reg_opt.apply_gradients(zip(grads_reg, reg_vars))

        # ── Pass 2: Adversarial Domain Classification ──────────────────────
        with tf.GradientTape() as tape_dom:
            _, dom_pred = self.model(X_both, training=True)
            dom_loss    = self.dom_loss_fn(d_both, dom_pred)

        dom_vars  = self._get_domain_vars()
        grads_dom = tape_dom.gradient(dom_loss, dom_vars)
        grads_dom = [g if g is not None else tf.zeros_like(v)
                     for g, v in zip(grads_dom, dom_vars)]
        self.dom_opt.apply_gradients(zip(grads_dom, dom_vars))

        return rul_loss, dom_loss

    def fit(self,
             X_src: np.ndarray,
             y_src: np.ndarray,
             X_tgt: np.ndarray,
             X_val_src: np.ndarray = None,
             y_val_src: np.ndarray = None,
             epochs: int = 200,
             batch_size: int = 256,
             patience: int = 20,
             lr_decay_epoch: int = 100) -> dict:
        """
        Full training loop.

        Key design decisions following the paper:
          - Over-sample the smaller dataset to equalise mini-batch counts
          - Apply ×0.1 LR decay at lr_decay_epoch (default: epoch 100)
          - Early stopping on source validation MAE (not target, since
            target labels are not available during adaptation)
          - Gradient clipping (clipnorm=1) via the SGD optimiser

        Args:
            X_src         : Source train windows  (N_s, Tw, F)
            y_src         : Source RUL labels      (N_s,)
            X_tgt         : Target train windows  (N_t, Tw, F)
            X_val_src     : Source validation windows (optional)
            y_val_src     : Source validation RUL (optional)
            epochs        : Maximum training epochs
            batch_size    : Mini-batch size
            patience      : Early stopping patience (epochs without improvement)
            lr_decay_epoch: Epoch at which to multiply LR by 0.1

        Returns:
            history dict with keys: rul_loss, dom_loss, val_mae
        """
        history = {'rul_loss': [], 'dom_loss': [], 'val_mae': []}
        best_val_mae  = np.inf
        no_improve    = 0
        best_weights  = None

        # Over-sample smaller domain to match mini-batch count
        n_src = len(X_src)
        if len(X_tgt) < n_src:
            repeat = int(np.ceil(n_src / len(X_tgt)))
            X_tgt  = np.tile(X_tgt, (repeat, 1, 1))[:n_src]

        n_batches = int(np.ceil(n_src / batch_size))

        for epoch in range(epochs):

            # Learning rate decay at specified epoch
            if epoch == lr_decay_epoch:
                new_lr_reg = float(self.reg_opt.learning_rate) * 0.1
                new_lr_dom = float(self.dom_opt.learning_rate) * 0.1
                self.reg_opt.learning_rate.assign(new_lr_reg)
                self.dom_opt.learning_rate.assign(new_lr_dom)
                print(f"  [LR decay at epoch {epoch}: "
                      f"reg={new_lr_reg:.6f}, dom={new_lr_dom:.6f}]")

            # Shuffle both domains independently each epoch
            src_idx = np.random.permutation(n_src)
            tgt_idx = np.random.permutation(len(X_tgt))

            epoch_rul, epoch_dom = [], []
            for b in range(n_batches):
                sl = src_idx[b * batch_size:(b + 1) * batch_size]
                tl = tgt_idx[b * batch_size:(b + 1) * batch_size]
                rl, dl = self.train_step(
                    X_src[sl].astype(np.float32),
                    y_src[sl].astype(np.float32),
                    X_tgt[tl].astype(np.float32)
                )
                epoch_rul.append(float(rl))
                epoch_dom.append(float(dl))

            mean_rul = np.mean(epoch_rul)
            mean_dom = np.mean(epoch_dom)
            history['rul_loss'].append(mean_rul)
            history['dom_loss'].append(mean_dom)

            # Validation and early stopping
            if X_val_src is not None:
                val_pred, _ = self.model(X_val_src.astype(np.float32),
                                          training=False)
                val_mae = float(tf.reduce_mean(
                    tf.abs(y_val_src[:, np.newaxis].astype(np.float32) - val_pred)
                ))
                history['val_mae'].append(val_mae)

                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    best_weights = self.model.get_weights()
                    no_improve   = 0
                else:
                    no_improve += 1

                if epoch % 10 == 0:
                    print(f"Epoch {epoch:03d} | "
                          f"RUL Loss: {mean_rul:.4f} | "
                          f"Dom Loss: {mean_dom:.4f} | "
                          f"Val MAE: {val_mae:.4f} | "
                          f"No improve: {no_improve}/{patience}")

                if no_improve >= patience:
                    print(f"\nEarly stopping at epoch {epoch}. "
                          f"Best Val MAE: {best_val_mae:.4f}")
                    if best_weights is not None:
                        self.model.set_weights(best_weights)
                    break

        return history
