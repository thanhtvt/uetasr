import math
import numpy as np
from typing import Union, List
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing \
    import PreprocessingLayer

from ...utils.common import get_shape


class TimeStretch(PreprocessingLayer):

    def __init__(self,
                 prob: float = 0.5,
                 min_speed_rate: float = 0.9,
                 max_speed_rate: float = 1.1,
                 sample_rate: int = 16000,
                 win_size: int = 25,
                 win_step: int = 10,
                 name: str = 'time_stretch',
                 **kwargs):
        super().__init__(trainable=False, name=name, **kwargs)
        self.prob = prob
        self.min_speed_rate = min_speed_rate
        self.max_speed_rate = max_speed_rate
        self.sample_rate = sample_rate
        win_size = int(sample_rate * win_size / 1000)
        self.win_size = 2**(win_size - 1).bit_length()
        self.win_step = int(sample_rate * win_step / 1000)

    def call(
        self,
        audio: Union[tf.Tensor, List[float], np.ndarray],
    ) -> tf.Tensor:
        rate = tf.random.uniform(shape=(),
                                 minval=self.min_speed_rate,
                                 maxval=self.max_speed_rate,
                                 dtype=tf.float32)
        stft = tf.signal.stft(audio,
                              self.win_size,
                              self.win_step,
                              fft_length=512,
                              pad_end=True)
        stft = phase_vocoder(stft, hop_len=512, rate=rate)
        augmented_audio = tf.signal.inverse_stft(stft,
                                                 self.win_size,
                                                 self.win_step,
                                                 fft_length=512)

        prob = tf.random.uniform(shape=(),
                                 minval=0,
                                 maxval=1,
                                 dtype=tf.float32)
        return tf.cond(prob <= self.prob, lambda: augmented_audio,
                       lambda: audio)


def phase_vocoder(stft, rate: float, hop_len: int = 512):
    if rate == 1.0:
        return stft

    shape = get_shape(stft)
    num_frames, freq = shape[-2:]

    perm = list(range(len(shape)))
    perm[-2], perm[-1] = perm[-1], perm[-2]
    stft = tf.transpose(stft, perm=perm)

    complex_specgrams = tf.reshape(stft, shape=[-1, freq, num_frames])

    real_dtype = tf.math.real(complex_specgrams).dtype
    time_steps = tf.range(0, num_frames, rate, dtype=real_dtype)

    alphas = time_steps % 1.0
    phase_0 = tf.math.angle(complex_specgrams[..., :1])

    complex_specgrams = tf.pad(complex_specgrams, [(0, 0), (0, 0), (0, 2)],
                               mode='CONSTANT')

    complex_specgrams_0 = tf.gather(complex_specgrams,
                                    tf.cast(time_steps, dtype=tf.int64),
                                    axis=-1)
    complex_specgrams_1 = tf.gather(complex_specgrams,
                                    tf.cast(time_steps + 1, dtype=tf.int64),
                                    axis=-1)

    angle_0 = tf.math.angle(complex_specgrams_0)
    angle_1 = tf.math.angle(complex_specgrams_1)

    norm_0 = tf.math.abs(complex_specgrams_0)
    norm_1 = tf.math.abs(complex_specgrams_1)

    phase_advance = tf.linspace(0.0, math.pi * hop_len, freq)[..., None]
    phase = angle_1 - angle_0 - phase_advance
    phase = phase - 2 * math.pi * tf.math.round(phase / (2 * math.pi))

    phase = phase + phase_advance
    phase = tf.concat([phase_0, phase[..., :-1]], axis=-1)
    phase_acc = tf.math.cumsum(phase, axis=-1)

    mag = alphas * norm_1 + (1 - alphas) * norm_0

    phase_acc = tf.cast(phase_acc, tf.complex64)
    mag = tf.cast(mag, tf.complex64)
    complex_specgrams_stretch = mag * tf.exp(1.j * phase_acc)

    shape = shape[:-2] + get_shape(complex_specgrams_stretch)[1:]
    complex_specgrams_stretch = tf.reshape(complex_specgrams_stretch,
                                           shape=shape)
    perm = list(range(len(shape)))
    perm[-2], perm[-1] = perm[-1], perm[-2]
    complex_specgrams_stretch = tf.transpose(complex_specgrams_stretch,
                                             perm=perm)
    return complex_specgrams_stretch
