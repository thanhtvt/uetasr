import math
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from librosa.effects import pitch_shift
from tensorflow.keras.layers.experimental.preprocessing import \
    PreprocessingLayer
from typing import Union, List

from ...utils.common import get_shape
from ...utils.audio import fix_length


class PitchShift(PreprocessingLayer):

    def __init__(
        self,
        prob: float = 0.5,
        min_semitones: int = -4,
        max_semitones: int = 4,
        sample_rate: int = 16000,
        name='pitch_shift',
    ):
        super().__init__(trainable=False, name=name)
        self.prob = prob
        self.min_semitones = min_semitones
        self.max_semitones = max_semitones
        self.sr = sample_rate

    def call(self, audio: tf.Tensor) -> tf.Tensor:
        num_semitones = tf.random.uniform(shape=(),
                                          minval=self.min_semitones,
                                          maxval=self.max_semitones,
                                          dtype=tf.int32)
        augmented_audio = tf.numpy_function(pitch_shift_librosa,
                                            [audio, self.sr, num_semitones],
                                            tf.float32)
        prob = tf.random.uniform(shape=(),
                                 minval=0,
                                 maxval=1,
                                 dtype=tf.float32)
        return tf.cond(prob <= self.prob,
                       lambda: augmented_audio,
                       lambda: audio)


def pitch_shift_librosa(audio, sr, n_steps):
    return pitch_shift(audio, sr=sr, n_steps=n_steps)


class PitchShift2(PreprocessingLayer):

    def __init__(self,
                 prob: float = 0.5,
                 sample_rate: int = 16000,
                 min_semitones: int = -4,
                 max_semitones: int = 4,
                 win_size: int = 25,
                 win_step: int = 10,
                 name='pitch_shift',
                 **kwargs):
        super().__init__(trainable=False, name=name, **kwargs)
        self.prob = prob
        self.min_semitones = min_semitones
        self.max_semitones = max_semitones
        self.sample_rate = sample_rate
        win_size = int(sample_rate * win_size / 1000)
        self.win_size = 2**(win_size - 1).bit_length()
        self.win_step = int(sample_rate * win_step / 1000)

    def call(
        self,
        audio: Union[tf.Tensor, List[float], np.ndarray],
    ) -> tf.Tensor:
        num_semitones = tf.random.uniform(shape=(),
                                          minval=self.min_semitones,
                                          maxval=self.max_semitones,
                                          dtype=tf.int32)
        rate = 2.0**(-num_semitones / 12)
        stft = tf.signal.stft(audio,
                              self.win_size,
                              self.win_step,
                              fft_length=512,
                              pad_end=True)
        stft = phase_vocoder(stft, hop_len=512, rate=rate)
        audio_stretch = tf.signal.inverse_stft(stft,
                                               self.win_size,
                                               self.win_step,
                                               fft_length=512)
        target_sample_rate = tf.cast(self.sample_rate / rate, tf.int64)
        augmented_audio = tfio.audio.resample(audio_stretch,
                                              target_sample_rate,
                                              self.sample_rate)
        augmented_audio = fix_length(augmented_audio,
                                     target_size=get_shape(audio)[-1])

        prob = tf.random.uniform(shape=(),
                                 minval=0,
                                 maxval=1,
                                 dtype=tf.float32)
        return tf.cond(prob <= self.prob, lambda: augmented_audio,
                       lambda: audio)


def phase_vocoder(stft, rate, hop_len=512):
    if rate == 1.0:
        return stft

    shape = get_shape(stft)
    num_frames, freq = shape[-2:]

    perm = list(range(len(shape)))
    perm[-2], perm[-1] = perm[-1], perm[-2]
    stft = tf.transpose(stft, perm=perm)

    complex_specgrams = tf.reshape(stft, shape=[-1, freq, num_frames])

    real_dtype = tf.math.real(complex_specgrams).dtype
    time_steps = tf.range(0, num_frames - 1, rate, dtype=real_dtype)

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
