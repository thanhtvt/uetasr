import tensorflow as tf

from typing import Tuple


class PositionalEncoding(tf.keras.layers.Layer):

    def __init__(
        self,
        d_model: int,
        dropout_rate: float = 0.0,
        max_len: int = 5000,
        **kwargs
    ):
        super(PositionalEncoding, self).__init__(trainable=False, **kwargs)
        self.d_model = d_model
        self.xscale = tf.math.sqrt(self.d_model * 1.0)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        position = tf.expand_dims(
            tf.range(0, max_len, dtype=tf.float32),
            axis=1
        )
        div_term = tf.math.exp(
            tf.repeat(tf.range(0, d_model, 2, dtype=tf.float32), repeats=2) *
            -(tf.math.log(10000.0) / d_model))
        sin = tf.math.sin(position * div_term)
        cos = tf.math.cos(position * div_term)
        mask = tf.tile(
            tf.expand_dims(tf.range(d_model), axis=0),
            [max_len, 1]
        ) % 2 == 0
        self.pe = tf.where(mask, sin, cos)
        self.pe = tf.expand_dims(self.pe, axis=0)

    def call(self, inputs, training=False):
        outputs = inputs * self.xscale + self.pe[:, tf.shape(inputs)[1]]
        return self.dropout(outputs, training=training)


class ScaledPositionalEncoding(PositionalEncoding):
    """Scaled positional encoding module.
    See Sec. 3.2  https://arxiv.org/abs/1809.08895
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Initialize class."""
        super().__init__(d_model=d_model,
                         dropout_rate=dropout_rate,
                         max_len=max_len)
        self.alpha = tf.Variable(1.0, trainable=True, dtype=tf.float32)

    def call(self, inputs, training=False):
        """Add positional encoding.
        Args:
            x (tf.Tensor): Input tensor (batch, time, `*`).
        Returns:
            tf.Tensor: Encoded tensor (batch, time, `*`).
        """
        outputs = inputs + self.alpha * self.pe[:, tf.shape(inputs)[1]]
        return self.dropout(outputs, training=training)


class RelPositionalEncoding(tf.keras.layers.Layer):
    """Relative positional encoding module (new implementation).
    Details can be found in https://github.com/espnet/espnet/pull/2816.
    See : Appendix B in https://arxiv.org/abs/1901.02860
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
    """

    def __init__(self, d_model: int, dropout_rate: float, max_len=5000):
        """Construct an PositionalEncoding object."""
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.xscale = tf.math.sqrt(self.d_model * 1.0)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        position = tf.expand_dims(tf.range(0, max_len, dtype=tf.float32),
                                  axis=1)
        div_term = tf.exp(
            tf.repeat(tf.range(0, d_model, 2, dtype=tf.float32), repeats=2) *
            -(tf.math.log(10000.0) / d_model))

        sin_pos = tf.math.sin(position * div_term)
        cos_pos = tf.math.cos(position * div_term)
        sin_neg = tf.math.sin(-1 * position * div_term)
        cos_neg = tf.math.cos(-1 * position * div_term)
        mask = tf.tile(
            tf.expand_dims(tf.range(d_model), axis=0),
            [max_len, 1]
        ) % 2 == 0
        pe_positive = tf.where(mask, sin_pos, cos_pos)
        pe_negative = tf.where(mask, sin_neg, cos_neg)

        pe_positive = tf.expand_dims(tf.reverse(pe_positive, axis=[0]), axis=0)
        pe_negative = tf.expand_dims(pe_negative[1:], axis=0)
        self.pe = tf.concat([pe_positive, pe_negative], axis=1)

    def call(self,
             inputs: tf.Tensor,
             left_context: int = 0,
             training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        """Add positional encoding.
        Args:
            x (tf.Tensor): Input tensor (batch, time, `*`).
        Returns:
            tf.Tensor: Encoded tensor (batch, time, `*`).
        """
        inputs = inputs * self.xscale
        time1 = tf.shape(inputs)[1] + left_context
        pos_emb = self.pe[:, self.max_len // 2 - time1 + 1:self.max_len // 2 +
                          tf.shape(inputs)[1]]
        return self.dropout(inputs,
                            training=training), self.dropout(pos_emb,
                                                             training=training)
