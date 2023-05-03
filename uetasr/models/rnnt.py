import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import \
    PreprocessingLayer

from typing import Tuple, Union
from ..utils.common import get_shape


class RNNT(tf.keras.Model):

    def __init__(
        self,
        encoder: tf.keras.layers.Layer,
        decoder: tf.keras.layers.Layer,
        jointer: tf.keras.layers.Layer,
        use_cmvn: bool = False,
        audio_preprocess: PreprocessingLayer = None,
        ctc_lin: tf.keras.layers.Layer = None,
        ctc_dropout: float = 0.1,
        lm_lin: tf.keras.layers.Layer = None,
        name: str = "rnnt",
        **kwargs,
    ):
        super(RNNT, self).__init__(name=name, **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.jointer = jointer
        self.use_cmvn = use_cmvn
        self.ctc_lin = ctc_lin
        self.lm_lin = lm_lin
        self.audio_preprocess = audio_preprocess
        self.window_size = encoder.window_size
        self.num_features = encoder.num_features
        self.ctc_dropout = tf.keras.layers.Dropout(ctc_dropout)
        self.cmvn = tf.keras.layers.Normalization(axis=-1) if use_cmvn else None

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.jointer.summary()

        if self.audio_preprocess:
            inputs = tf.keras.Input(shape=(None,),
                                    batch_size=None,
                                    name="audio")
        else:
            inputs = tf.keras.Input(shape=(None, self.window_size,
                                           self.num_features),
                                    batch_size=None,
                                    name="feature")

        inputs_length = tf.keras.Input(shape=(1,),
                                       batch_size=None,
                                       name="inputs_length")

        labels = tf.keras.Input(shape=(None,),
                                batch_size=None,
                                dtype=tf.int32,
                                name="labels")

        model = tf.keras.Model(inputs=[inputs, inputs_length, labels],
                               outputs=self((inputs, inputs_length, labels)),
                               name="rnnt_model")
        model.summary()

    def adapt(self, inputs, batch_size):
        if self.cmvn:
            self.cmvn.adapt(inputs, batch_size=batch_size)

    @property
    def is_adapted(self):
        return self.cmvn.is_adapted if self.cmvn else False

    def call(
        self,
        inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
        training: bool = False
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:

        inputs, inputs_length, labels = inputs
        if len(get_shape(inputs_length)) == 2:
            inputs_length = tf.squeeze(inputs_length, axis=-1)

        if self.audio_preprocess:
            feature, feature_length = self.audio_preprocess(
                inputs, lengths=inputs_length)
        else:
            feature, feature_length = inputs, inputs_length

        mask = tf.sequence_mask(feature_length, maxlen=tf.shape(feature)[1])
        mask = tf.expand_dims(mask, axis=1)
        encoder_output, encoder_mask = self.encoder(feature, mask, training=training)

        encoder_mask = tf.squeeze(encoder_mask, axis=1)
        length = tf.math.reduce_sum(tf.cast(encoder_mask, tf.int32), axis=1)

        decoder_output = self.decoder(labels, training=training)
        logits = self.jointer(encoder_output, decoder_output, training=training)
        logits.length = length

        if self.ctc_lin:
            ctc_logits = self.ctc_lin(encoder_output, training=training)
            ctc_logits = self.ctc_dropout(ctc_logits, training=training)
            ctc_logits.length = length
            return logits, ctc_logits
        else:
            return logits

    def get_config(self):
        config = super(RNNT, self).get_config()
        config.update({
            "use_cmvn": self.use_cmvn,
            "window_size": self.window_size,
            "num_features": self.num_features,
        })
        config.update(self.encoder.get_config())
        config.update(self.decoder.get_config())
        config.update(self.jointer.get_config())
        config.update(self.ctc_lin.get_config()) if self.ctc_lin else None
        config.update(self.lm_lin.get_config()) if self.lm_lin else None
        config.update(self.audio_preprocess.get_config()) if self.audio_preprocess else None
        config.update(self.ctc_dropout.get_config())
        config.update(self.cmvn.get_config()) if self.cmvn else None
        return config
