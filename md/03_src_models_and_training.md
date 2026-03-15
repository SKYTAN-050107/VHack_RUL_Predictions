# 03 — Source Modules: Models & Training

> **IDE Agent Instructions:** Create each file at the exact path shown under `### Create File:`. Files `grl.py`, `lstm_baseline.py`, and `lstm_dann.py` go inside `src/models/`. File `train.py` goes inside `src/`.

---

## 3.1 — Gradient Reversal Layer

### Create File: `src/models/grl.py`

```python
import tensorflow as tf


@tf.custom_gradient
def _gradient_reversal_op(x, alpha):
    """
    Custom TensorFlow op that is an identity function in the forward pass
    and negates + scales gradients in the backward pass.

    This is the core of the Domain Adversarial Neural Network mechanism:
    when gradients flow back through this op, the feature extractor receives
    a signal to MAXIMISE the domain classification loss (i.e., confuse the
    domain classifier) rather than minimise it.
    """
    def grad(dy):
        return -alpha * dy, None
    return x, grad


class GradientReversalLayer(tf.keras.layers.Layer):
    """
    Gradient Reversal Layer (GRL) as described in Ganin & Lempitsky (2015)
    and used in the LSTM-DANN paper (Section 3.4).

    Forward pass  : f(x) = x  (pure identity, no transformation)
    Backward pass : dL/dx  →  -alpha * dL/dx

    Effect during training:
      - The domain classifier (downstream of GRL) learns to discriminate
        between source and target domains, minimising its own loss.
      - The feature extractor (upstream of GRL) receives reversed gradients,
        so it learns to CONFUSE the domain classifier, producing features
        that are indistinguishable across domains (domain-invariant features).

    Args:
        alpha : Scaling factor for the reversed gradient.
                Higher alpha = stronger domain confusion pressure.
                Typical range: 0.8 – 3.0 (see Table 3 in paper).
    """

    def __init__(self, alpha: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = tf.cast(alpha, dtype=tf.float32)

    def call(self, x):
        return _gradient_reversal_op(x, self.alpha)

    def get_config(self):
        config = super().get_config()
        config.update({'alpha': float(self.alpha)})
        return config
```

---

## 3.2 — Baseline LSTM Model

### Create File: `src/models/lstm_baseline.py`

```python
import tensorflow as tf
from tensorflow.keras import layers, Model, Input


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
```

---

## 3.3 — LSTM-DANN Model

### Create File: `src/models/lstm_dann.py`

```python
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
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
```

---

## 3.4 — LSTM-DANN Training Loop

### Create File: `src/train.py`

```python
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
```
