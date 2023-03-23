import numpy as np
import tensorflow as tf
from typing import List, Union
from tensorflow.keras.layers.experimental.preprocessing import \
    PreprocessingLayer


class GaussianNoise(PreprocessingLayer):

    def __init__(self,
                 prob: float = 0.5,
                 min_gaussian_snr: float = -20,
                 max_gaussian_snr: float = 60,
                 name: str = 'gaussian_noise',
                 **kwargs):
        super().__init__(trainable=False, name=name, **kwargs)
        self.prob = prob
        self.min_gaussian_snr = min_gaussian_snr
        self.max_gaussian_snr = max_gaussian_snr

    def call(
        self,
        audio: Union[tf.Tensor, List[float], np.ndarray],
    ) -> tf.Tensor:
        snr = tf.random.uniform(shape=(),
                                minval=self.min_gaussian_snr,
                                maxval=self.min_gaussian_snr,
                                dtype=tf.float32)
        rms = tf.math.sqrt(tf.math.reduce_mean(tf.math.square(audio)))
        noise_rms = rms / (10**(snr / 20))
        noise = tf.random.normal(tf.shape(audio), mean=0.0, stddev=noise_rms)
        augmented_audio = audio + noise
        prob = tf.random.uniform(shape=(),
                                 minval=0,
                                 maxval=1,
                                 dtype=tf.float32)
        return tf.cond(prob <= self.prob, lambda: augmented_audio,
                       lambda: audio)