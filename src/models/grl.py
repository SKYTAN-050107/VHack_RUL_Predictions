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
    # Original: just reverses gradient sign
    # New: reverses AND clips magnitude to prevent over-pushing encoder
        @tf.custom_gradient
        def _reverse_clip(x):
            def grad(dy):
            # Reverse sign, clip norm to 0.1
                reversed_grad = -self.alpha * dy
                return tf.clip_by_norm(reversed_grad, clip_norm=0.1)
            return x, grad
        return _reverse_clip(x)

    def get_config(self):
        config = super().get_config()
        config.update({'alpha': float(self.alpha)})
        return config
