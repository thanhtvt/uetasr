import math
import tensorflow as tf


class RNNTJointer(tf.keras.Model):

    def __init__(self,
                 encoder_dim: int,
                 decoder_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 name: str = 'rnnt_jointer',
                 **kwargs):
        super(RNNTJointer, self).__init__(name=name, **kwargs)

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.activation = tf.keras.layers.Activation(tf.nn.tanh)

        initializer1 = tf.keras.initializers.RandomUniform(
            minval=-math.sqrt(1.0 / hidden_dim),
            maxval=math.sqrt(1.0 / hidden_dim),
            seed=0
        )
        initializer2 = tf.keras.initializers.RandomUniform(
            minval=-math.sqrt(1.0 / hidden_dim),
            maxval=math.sqrt(1.0 / hidden_dim),
            seed=0
        )
        initializer3 = tf.keras.initializers.RandomUniform(
            minval=-math.sqrt(1.0 / hidden_dim),
            maxval=math.sqrt(1.0 / hidden_dim),
            seed=0
        )

        self.enc_fc = tf.keras.layers.Dense(hidden_dim,
                                            kernel_initializer=initializer1,
                                            bias_initializer=initializer1)
        self.dec_fc = tf.keras.layers.Dense(hidden_dim,
                                            use_bias=False,
                                            kernel_initializer=initializer2,
                                            bias_initializer=initializer2)

        self.joint = tf.keras.layers.Add()

        self.out = tf.keras.layers.Dense(output_dim,
                                         kernel_initializer=initializer3,
                                         bias_initializer=initializer3)

    def summary(self):
        input_enc = tf.keras.Input(shape=(None, self.encoder_dim),
                                   batch_size=None)
        input_dec = tf.keras.Input(shape=(None, self.decoder_dim),
                                   batch_size=None)
        outputs = self.call(input_enc, input_dec)
        model = tf.keras.Model(inputs=[input_enc, input_dec],
                               outputs=outputs,
                               name=self.name)
        model.summary()

    def call(self,
             enc_out: tf.Tensor,
             dec_out: tf.Tensor,
             training: bool = False) -> tf.Tensor:
        enc_out = self.enc_fc(enc_out, training=training)
        dec_out = self.dec_fc(dec_out, training=training)

        enc_out = tf.expand_dims(enc_out, axis=2)
        dec_out = tf.expand_dims(dec_out, axis=1)

        x = self.joint([enc_out, dec_out], training=training)
        x = self.activation(x, training=training)
        x = self.out(x, training=training)
        return x
