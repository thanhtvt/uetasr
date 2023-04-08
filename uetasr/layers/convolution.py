import math
import tensorflow as tf


class GLU(tf.keras.layers.Layer):

    def __init__(self, axis=-1, **kwargs):
        super(GLU, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        a, b = tf.split(inputs, 2, axis=self.axis)
        b = tf.nn.sigmoid(b)
        return tf.multiply(a, b)


class ConvolutionModule(tf.keras.layers.Layer):
    """ConvolutionModule in Conformer model.
    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.
    """

    def __init__(self,
                 channels,
                 kernel_size,
                 activation=tf.keras.layers.ReLU(),
                 causal=False,
                 bias=True):
        """Construct an ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        initializer1 = tf.keras.initializers.RandomUniform(
            minval=-math.sqrt(1.0 / channels),
            maxval=math.sqrt(1.0 / channels),
            seed=0
        )
        initializer2 = tf.keras.initializers.RandomUniform(
            minval=-math.sqrt(channels / channels / 2),
            maxval=math.sqrt(channels / channels / 2),
            seed=0
        )
        initializer3 = tf.keras.initializers.RandomUniform(
            minval=-math.sqrt(1.0 / channels),
            maxval=math.sqrt(1.0 / channels),
            seed=0
        )

        self.pointwise_conv1 = tf.keras.layers.Conv1D(
            2 * channels,
            kernel_size=1,
            strides=1,
            padding='valid',
            data_format='channels_last',
            use_bias=bias,
            kernel_initializer=initializer1,
            bias_initializer=initializer1
        )
        self.depthwise_conv = tf.keras.layers.Conv1D(
            channels,
            kernel_size,
            strides=1,
            padding='same' if not causal else 'causal',
            data_format='channels_last',
            groups=channels,
            use_bias=bias,
            kernel_initializer=initializer2,
            bias_initializer=initializer2
        )
        self.glu = GLU(axis=-1)
        self.norm = tf.keras.layers.BatchNormalization()
        self.pointwise_conv2 = tf.keras.layers.Conv1D(
            channels,
            kernel_size=1,
            strides=1,
            padding='valid',
            data_format='channels_last',
            use_bias=bias,
            kernel_initializer=initializer3,
            bias_initializer=initializer3
        )
        self.activation = activation
        self.kernel_size = kernel_size

    def call(self, x, cache=None, training=False):
        """Compute convolution module.
        Args:
            x (tf.Tensor): Input tensor (#batch, time, channels).
        Returns:
            tf.Tensor: Output tensor (#batch, time, channels).
        """
        # GLU mechanism
        x = self.pointwise_conv1(x, training=training)  # (batch, dim, 2*channel)
        x = self.glu(x)  # (batch, dim, channel)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x, training=training)
        x = self.activation(self.norm(x, training=training), training=training)

        x = self.pointwise_conv2(x, training=training)
        return x
