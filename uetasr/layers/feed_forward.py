import math
import tensorflow as tf
from tensorflow.keras.layers import Layer

from .normalization import RMSLayerNormalization


class PositionwiseFeedForward(Layer):
    """Positionwise feed forward layer.
    Args:
        input_dim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_units: int,
                 dropout_rate: float = 0.2,
                 activation: Layer = tf.keras.layers.ReLU()):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        initializer1 = tf.keras.initializers.RandomUniform(
            minval=-math.sqrt(1.0 / input_dim),
            maxval=math.sqrt(1.0 / input_dim),
            seed=None
        )
        initializer2 = tf.keras.initializers.RandomUniform(
            minval=-math.sqrt(1.0 / hidden_units),
            maxval=math.sqrt(1.0 / hidden_units),
            seed=None
        )
        self.w_1 = tf.keras.layers.Dense(hidden_units,
                                         kernel_initializer=initializer1,
                                         bias_initializer=initializer1)
        self.w_2 = tf.keras.layers.Dense(input_dim,
                                         kernel_initializer=initializer2,
                                         bias_initializer=initializer2)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.activation = activation

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward function."""
        x = self.w_1(x, training=training)
        x = self.activation(x, training=training)
        x = self.dropout(x, training=training)
        x = self.w_2(x, training=training)
        return x


class PointwiseFeedForward(tf.keras.layers.Layer):

    def __init__(self,
                 input_dim: int,
                 dropout_rate: float = 0.1,
                 ffn_dim: int = 1024,
                 activation: Layer = tf.keras.layers.ReLU(),
                 use_rmsnorm: bool = False,
                 name: str = 'pointwise_feed_forward',
                 **kwargs):
        super(PointwiseFeedForward, self).__init__(name=name, **kwargs)
        self.input_dim = input_dim
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        if use_rmsnorm:
            norm_fn = RMSLayerNormalization(epsilon=1e-5, p=0.0625)
        else:
            norm_fn = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.ffn = tf.keras.Sequential([
            norm_fn,
            tf.keras.layers.Dense(ffn_dim),
            activation,
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(input_dim),
            tf.keras.layers.Dropout(dropout_rate),
        ])
        self.layernorm_out = norm_fn
        self.add1 = tf.keras.layers.Add()
        self.add2 = tf.keras.layers.Add()

    def summary(self):
        rc_output = tf.keras.Input(shape=(None, self.input_dim),
                                   batch_size=None)
        center_context = tf.keras.Input(shape=(None, self.input_dim),
                                        batch_size=None)
        right_context = tf.keras.Input(shape=(None, self.input_dim),
                                       batch_size=None)
        output = self.call(rc_output, center_context, right_context)
        model = tf.keras.Model(
            inputs=[rc_output, center_context, right_context],
            outputs=output,
            name=self.name
        )
        model.summary()

    def call(
        self,
        rc_output: tf.Tensor,
        center_context: tf.Tensor,
        right_context: tf.Tensor,
        training: bool = False
    ) -> tf.Tensor:
        """
        Args:
            rc_output (tf.Tensor): Output from previous MHSA layer
                                   (R + C, B, feat_dim)
            center_context (tf.Tensor): Center context (C, B, feat_dim)
            right_context (tf.Tensor): Right context (R, B, feat_dim)
            training (bool): Whether in training mode or not

        Return:
            output (tf.Tensor): Concat of right_context and center_context for
                                upper layer (R + C, B, feat_dim)
        """
        context = tf.concat([right_context, center_context], axis=0)
        output = self.dropout(rc_output, training=training)
        output = self.add1([context, output], training=training)
        output_ffn = self.ffn(output, training=training)
        output = self.add2([output_ffn, output], training=training)
        output = self.layernorm_out(output, training=training)
        return output
