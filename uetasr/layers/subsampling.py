import tensorflow as tf

from typing import List, Tuple, Union

from .positional_encoding import PositionalEncoding
from ..utils.common import get_shape

L2 = tf.keras.regularizers.l2(1e-6)


class Conv2DSubsampling(tf.keras.layers.Layer):
    def __init__(
        self,
        out_dim: int,
        strides: Union[int, Tuple, List] = 2,
        kernel_size: Union[int, Tuple, List] = 3,
        kernel_regularizer: tf.keras.regularizers.Regularizer = L2,
        bias_regularizer: tf.keras.regularizers.Regularizer = L2,
        name: str = 'conv2d_subsampling',
        **kwargs,
    ):
        super(Conv2DSubsampling, self).__init__(name=name, **kwargs)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=out_dim,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            name=f"{name}_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=out_dim,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            name=f"{name}_2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.time_reduction_factor = self.conv1.strides[0] * self.conv2.strides[0]

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.conv1(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x, training=training)
        x = tf.nn.relu(x)
        return x


class Conv2dSubsamplingV2(tf.keras.layers.Layer):
    """Convolutional 2D subsampling (to 1/4 length).
    Args:
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc: Custom position encoding layer.
    """

    def __init__(
        self,
        odim: int,
        dropout_rate: float = 0.1,
        kernel_regularizer: tf.keras.regularizers.Regularizer = L2,
        bias_regularizer: tf.keras.regularizers.Regularizer = L2,
        pos_enc: tf.keras.layers.Layer = None
    ):
        """Construct an Conv2dSubsampling object."""
        super().__init__()
        self.conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=odim,
                                   kernel_size=3,
                                   strides=2,
                                   padding="valid",
                                   data_format='channels_last',
                                   kernel_regularizer=kernel_regularizer,
                                   bias_regularizer=bias_regularizer,),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=odim,
                                   kernel_size=3,
                                   strides=2,
                                   padding="valid",
                                   data_format='channels_last',
                                   kernel_regularizer=kernel_regularizer,
                                   bias_regularizer=bias_regularizer,),
            tf.keras.layers.ReLU(),
        ])
        self.out = tf.keras.layers.Dense(odim)
        self.pos_enc = pos_enc if pos_enc else PositionalEncoding(
            odim, dropout_rate)

    def call(
        self,
        x: tf.Tensor,
        x_mask: tf.Tensor = None,
        training: bool = False
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """Subsample x.
        Args:
            x (tf.Tensor): Input tensor (#batch, time, 1, idim).
            x_mask (tf.Tensor): Input mask (#batch, 1, time).
        Returns:
            tf.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            tf.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
        """
        x = tf.transpose(x, perm=[0, 1, 3, 2])
        x = self.conv(x, training=training)
        b, t, f, c = get_shape(x)
        x = self.out(tf.reshape(tf.transpose(x, perm=[0, 1, 3, 2]),
                                shape=(b, t, c * f)),
                     training=training)
        x = self.pos_enc(x, training=training)
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]
