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
                                maxval=self.max_gaussian_snr,
                                dtype=tf.float32)
        rms = tf.math.sqrt(tf.math.reduce_mean(tf.math.square(audio)))
        noise_rms = rms / (10**(snr / 20))
        noise = tf.random.normal(tf.shape(audio), mean=0.0, stddev=noise_rms)
        augmented_audio = audio + noise
        prob = tf.random.uniform(shape=(),
                                 minval=0,
                                 maxval=1,
                                 dtype=tf.float32)
        return tf.cond(prob <= self.prob,
                       lambda: augmented_audio,
                       lambda: audio)


class Reverber(PreprocessingLayer):

    def __init__(
        self,
        rir_path: str,
        prob: float = 0.5,
        sample_rate: int = 16000,
        name: str = 'reverber',
        **kwargs,
    ):
        super().__init__(trainable=False, name=name, **kwargs)
        self.prob = prob
        self.sample_rate = sample_rate

        with open(rir_path, encoding="utf-8") as fin:
            paths = [path.strip() for path in fin]

        self.length = len(paths)
        self.rir_dataset = tf.convert_to_tensor(paths, dtype=tf.string)
        self.rir_dataset = tf.random.shuffle(self.rir_dataset)

    def call(
        self,
        audio: Union[tf.Tensor, List[float], np.ndarray],
    ) -> tf.Tensor:
        idx = tf.random.uniform(shape=(),
                                minval=0,
                                maxval=self.length - 1,
                                dtype=tf.int32)
        rir_path = self.rir_dataset[idx]
        data = tf.io.read_file(rir_path)
        rir = tf.io.parse_tensor(data, tf.float32)

        rotation_index = tf.cast(tf.math.argmax(tf.math.abs(rir)), tf.int32)

        length = tf.cast(tf.shape(audio)[-1], dtype=tf.int32)
        rir = rir[tf.maximum(0, rotation_index -
                             int(0.15 * self.sample_rate)):
                  tf.minimum(length, rotation_index +
                             int(0.15 * self.sample_rate))]
        rir_rms = tf.sqrt(tf.reduce_mean(tf.square(rir), axis=0))
        rir = rir / rir_rms
        rir = tf.reverse(rir, axis=[-1])

        audio_padded = tf.cond(
            tf.shape(audio)[0] < tf.shape(rir)[0], lambda: tf.pad(
                audio, [[0, tf.shape(rir)[0] - tf.shape(audio)[0]]]),
            lambda: audio
        )
        audio_padded = tf.reshape(audio_padded, [-1, length, 1])

        rir = tf.reshape(rir, shape=(-1, 1, 1))
        augmented_audio = tf.nn.conv1d(audio_padded,
                                       rir,
                                       stride=1,
                                       padding='VALID')
        augmented_audio = tf.reshape(augmented_audio,
                                     shape=(tf.shape(augmented_audio)[1], ))

        prob = tf.random.uniform(shape=(),
                                 minval=0,
                                 maxval=1,
                                 dtype=tf.float32)
        return tf.cond(prob <= self.prob,
                       lambda: augmented_audio,
                       lambda: audio)
