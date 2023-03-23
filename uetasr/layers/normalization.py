import tensorflow as tf


class RMSLayerNormalization(tf.keras.layers.Layer):
    """
    Root Mean Square Layer Normalization

    Paper: https://arxiv.org/pdf/1910.07467.pdf
    Implementation: https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_tensorflow.py
    """

    def __init__(self, epsilon=1e-8, p=-1.0, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.bias = bias
        self.p = p

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.ones_initializer(),
            trainable=True
        )
        if self.bias:
            self.offset = self.add_weight(
                name='offset',
                shape=input_shape[-1:],
                initializer=tf.zeros_initializer(),
                trainable=True
            )
        super().build(input_shape)

    def call(self, x):
        if self.p < 0 or self.p > 1:
            ms = tf.reduce_mean(x ** 2, axis=-1, keepdims=True)
        else:
            layer_size = tf.shape(x)[-1]
            partial_size = tf.cast(tf.cast(layer_size, tf.float32) * self.p,
                                   dtype=tf.int32)
            partial_x, _ = tf.split(
                x, [partial_size, layer_size - partial_size], axis=-1
            )
            ms = tf.reduce_mean(partial_x ** 2, axis=-1, keepdims=True)

        norm_x = self.scale * x * tf.math.rsqrt(ms + self.epsilon)
        if self.bias:
            return norm_x + self.offset
        return norm_x
