import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import \
    PreprocessingLayer
from typing import Union, List
try:
    from telecodecs.g711.g711 import G711
except ImportError:
    print("You must install `telecodecs` package to use this augmentor.")
    print("Run script ./tools/install_g711_augment.sh to install it.")
    raise ImportError


class Telephony(PreprocessingLayer):
    def __init__(
        self,
        prob: float = 0.5,
        law: str = 'A',
        input_sr: int = 16000,
        max_band_len: int = 160000,
        random_band_size: int = 10,
        name: str = 'telephony',
        **kwargs
    ):
        super().__init__(trainable=False, name=name, **kwargs)
        self.prob = prob
        self.g711 = G711(law=law, input_sr=input_sr)
        self.max_band_len = max_band_len
        self.random_band_size = random_band_size
        self.band_limited_noise = [
            self.get_max_band_limited_noise(samples=max_band_len)
            for _ in range(random_band_size)
        ]
        self.band_limited_noise = tf.stack(self.band_limited_noise)

    def call(
        self,
        audio: Union[tf.Tensor, List[float], np.ndarray],
    ) -> tf.Tensor:
        prob = tf.random.uniform(shape=(),
                                 minval=0,
                                 maxval=1,
                                 dtype=tf.float32)
        band_idx = tf.random.uniform(shape=(),
                                     minval=0,
                                     maxval=self.random_band_size - 1,
                                     dtype=tf.int32)
        band_noise = self.band_limited_noise[band_idx]
        augmented_audio = tf.numpy_function(self.g711.convert,
                                            [audio, band_noise], audio.dtype)
        augmented_audio.set_shape(audio.shape)
        return tf.cond(prob <= self.prob,
                       lambda: augmented_audio,
                       lambda: audio)

    @staticmethod
    def get_max_band_limited_noise(
        min_freq: int = 200,
        max_freq: int = 4000,
        samples: int = 16000,
        samplerate: int = 8000
    ):
        t = np.linspace(0, samples / samplerate, samples)
        freqs = np.arange(min_freq, max_freq + 1, samples / samplerate)
        phases = np.random.rand(len(freqs)) * 2 * np.pi
        t = np.expand_dims(t, axis=0)
        freqs = np.expand_dims(freqs, axis=1)
        phases = np.expand_dims(phases, axis=1)
        signals = np.sin(2 * np.pi * freqs * t + phases)
        signal = np.add.reduce(signals)
        signal /= np.max(signal)
        return signal
