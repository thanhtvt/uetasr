import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import \
    PreprocessingLayer

from typing import Tuple


class CTC(tf.keras.Model):

    def __init__(
        self,
        encoder: tf.keras.layers.Layer,
        audio_preprocess: PreprocessingLayer = None,
        name: str = "ctc",
        **kwargs,
    ):
        super(CTC, self).__init__(name=name, **kwargs)
        self.encoder = encoder
        self.audio_preprocess = audio_preprocess
        self.window_size = encoder.window_size
        self.num_features = encoder.num_features

    def summary(self):
        self.encoder.summary()

        if self.audio_preprocess:
            inputs = tf.keras.Input(shape=(None, ),
                                    batch_size=None,
                                    name="audio")
        else:
            inputs = tf.keras.Input(shape=(None, self.window_size,
                                           self.num_features),
                                    batch_size=None,
                                    name="feature")

        inputs_length = tf.keras.Input(shape=(1, ),
                                       batch_size=None,
                                       name="inputs_length")

        model = tf.keras.Model(inputs=[inputs, inputs_length],
                               outputs=self((inputs, inputs_length)))
        model.summary()

    def call(self,
             inputs: Tuple[tf.Tensor, tf.Tensor],
             training: bool = False):
        inputs, inputs_length = inputs
        if self.audio_preprocess:
            feature, feature_length = self.audio_preprocess(
                inputs, lengths=inputs_length)
        else:
            feature, feature_length = inputs, inputs_length

        logits = self.encoder(feature, training=training)
        logits.length = feature_length
        return logits
