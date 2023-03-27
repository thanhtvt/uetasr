import librosa
import numpy as np
import tensorflow as tf
import tensorflow_text as tftext
from tensorflow.keras.layers.experimental.preprocessing import \
    PreprocessingLayer
from typing import List, Tuple, Union


class BaseFeaturizer(PreprocessingLayer):
    def __init__(self, name: str = "base_featurizer", **kwargs):
        super(BaseFeaturizer, self).__init__(trainable=False,
                                             name=name,
                                             **kwargs)

    def call(
        self,
        audio: Union[tf.Tensor, np.ndarray, List[float]],
        lengths: tf.Tensor = None,
        feature_augmentor: PreprocessingLayer = None
    ) -> tf.Tensor:
        """ Featurizes audio.

        Args:
            audio: A tensor of shape [time] or [time, 1]
                containing the audio sequences.
            lengths: A scalar tensor containing the lengths of the
                audio sequences.
            feature_augmentor: A preprocessing layer that can be used to augment

        Returns:
            (tf.Tensor): Features
        """

        raise NotImplementedError


class MFCC(BaseFeaturizer):
    def __init__(
        self,
        use_stack: bool = True,
        window_size: int = 3,
        window_step: int = 3,
        num_mel_bins: int = 40,
        sample_rate: int = 16000,
        frame_ms: int = 25,
        stride_ms: int = 10,
        lower_edge_hertz: int = 0,
        upper_edge_hertz: int = 8000,
        name: str = "mfcc",
        **kwargs
    ):
        super(MFCC, self).__init__(name=name, **kwargs)
        self.use_stack = use_stack
        self.window_size = window_size
        self.window_step = window_step
        self.num_mel_bins = num_mel_bins
        self.sample_rate = sample_rate
        self.frame_length = int(self.sample_rate * (frame_ms / 1000))
        self.frame_step = int(self.sample_rate * (stride_ms / 1000))
        self.lower_edge_hertz = lower_edge_hertz
        self.upper_edge_hertz = upper_edge_hertz

    def call(
        self,
        audio: Union[tf.Tensor, np.ndarray, List[float]],
        lengths: tf.Tensor = None,
        feature_augmentor: PreprocessingLayer = None
    ) -> Union[tf.Tensor, Tuple[tf.Tensor]]:
        log_mel_spectrograms = compute_fbanks(
            audio,
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            sample_rate=self.sample_rate,
            num_mel_bins=self.num_mel_bins,
            lower_edge_hertz=self.lower_edge_hertz,
            upper_edge_hertz=self.upper_edge_hertz
        )
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)

        if self.use_stack:
            begin = [0, 0, 0, 0] if len(audio.shape) > 1 else [0, 0, 0]
            end = [-1, -1, -1, -1] if len(audio.shape) > 1 else [-1, -1, -1]
            if len(audio.shape) > 1:
                strides = [1, self.window_step, 1, 1]
            else:
                strides = [self.window_step, 1, 1]
            axis = 1 if len(audio.shape) > 1 else 0

            mfccs = tf.strided_slice(
                tftext.sliding_window(
                    mfccs, width=self.window_size, axis=axis),
                begin=begin,
                end=end,
                strides=strides,
                end_mask=15)
        else:
            self.window_step = 1

        if feature_augmentor:
            mfccs = feature_augmentor(mfccs)

        if lengths is not None:
            lengths = tf.cast(
                (-(-lengths // self.frame_step) - self.window_size)
                // self.window_step + 1,
                dtype=tf.int32
            )
            return mfccs, lengths

        return mfccs


class LogMelSpectrogram(BaseFeaturizer):

    def __init__(
        self,
        fs: Union[int, str] = 16000,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        fmin: int = None,
        fmax: int = None,
        htk: bool = False,
        name: str = 'log_mel_spectrogram',
        **kwargs
    ):
        super(LogMelSpectrogram, self).__init__(name=name, **kwargs)

        self.sample_rate = self.fs = fs
        self.n_fft = n_fft
        self.win_length = win_length
        self.frame_step = self.hop_length = hop_length
        self.num_mel_bins = self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.window_size = self.window_step = 1

        _mel_options = dict(
            sr=fs,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=htk,
        )
        self.melmat = tf.convert_to_tensor(
            librosa.filters.mel(**_mel_options).T, dtype=tf.float32)

    def call(
        self,
        audio: Union[tf.Tensor, np.ndarray, List[float]],
        lengths: tf.Tensor = None,
        feature_augmentor: PreprocessingLayer = None
    ) -> tf.Tensor:
        stfts = tf.signal.stft(audio,
                               frame_length=self.win_length,
                               frame_step=self.hop_length,
                               pad_end=True)
        spectrograms = tf.abs(stfts)**2
        mel_spectrograms = tf.tensordot(spectrograms, self.melmat, 1)

        mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
            self.melmat.shape[-1:]))

        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-10)

        if feature_augmentor:
            log_mel_spectrograms = feature_augmentor(log_mel_spectrograms)

        log_mel_spectrograms = tf.expand_dims(log_mel_spectrograms, axis=-2)
        return log_mel_spectrograms


class FBank(BaseFeaturizer):

    def __init__(
        self,
        use_stack: bool = True,
        window_size: int = 3,
        window_step: int = 3,
        num_mel_bins: int = 80,
        sample_rate: int = 16000,
        frame_ms: int = 25,
        stride_ms: int = 10,
        lower_edge_hertz: int = 0,
        upper_edge_hertz: int = 8000,
        pad_end_stft: bool = True,
        name: str = "fbank",
        **kwargs,
    ):
        super(FBank, self).__init__(name=name, **kwargs)
        self.use_stack = use_stack
        self.window_size = window_size
        self.window_step = window_step
        self.num_mel_bins = num_mel_bins
        self.sample_rate = sample_rate
        self.frame_length = int(self.sample_rate * (frame_ms / 1000))
        self.frame_step = int(self.sample_rate * (stride_ms / 1000))
        self.lower_edge_hertz = lower_edge_hertz
        self.upper_edge_hertz = upper_edge_hertz
        self.pad_end_stft = pad_end_stft

    def call(
        self,
        audio: Union[tf.Tensor, tf.SparseTensor, List, np.ndarray],
        lengths: tf.Tensor = None,
        feature_augmentor=None,
    ) -> Union[tf.Tensor, Tuple[tf.Tensor]]:
        log_mel_spectrograms = compute_fbanks(
            audio,
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            sample_rate=self.sample_rate,
            num_mel_bins=self.num_mel_bins,
            lower_edge_hertz=self.lower_edge_hertz,
            upper_edge_hertz=self.upper_edge_hertz,
            pad_end_stft=self.pad_end_stft
        )

        if feature_augmentor:
            log_mel_spectrograms = feature_augmentor(log_mel_spectrograms)

        if self.use_stack:
            if self.window_step == self.window_size == 1:
                log_mel_spectrograms = tf.expand_dims(log_mel_spectrograms,
                                                      axis=-2)
            else:
                begin = [0, 0, 0, 0] if len(audio.shape) > 1 else [0, 0, 0]
                end = [-1, -1, -1, -1] if \
                    len(audio.shape) > 1 else [-1, -1, -1]
                strides = [1, self.window_step, 1, 1] if len(audio.shape) > 1 \
                    else [self.window_step, 1, 1]
                axis = 1 if len(audio.shape) > 1 else 0

                log_mel_spectrograms = tf.strided_slice(tftext.sliding_window(
                    log_mel_spectrograms, width=self.window_size, axis=axis),
                                                        begin=begin,
                                                        end=end,
                                                        strides=strides,
                                                        end_mask=15)
        else:
            self.window_step = 1
            log_mel_spectrograms = log_mel_spectrograms

        if lengths is not None:
            lengths = tf.cast(
                (-(-lengths // self.frame_step) - self.window_size)
                // self.window_step + 1,
                dtype=tf.int32
            )
            return log_mel_spectrograms, lengths
        return log_mel_spectrograms

    def get_config(self):
        config = super(FBank, self).get_config()
        config.update({
            'use_stack': self.use_stack,
            'window_size': self.window_size,
            'window_step': self.window_step,
            'num_mel_bins': self.num_mel_bins,
            'sample_rate': self.sample_rate,
            'frame_length': self.frame_length,
            'frame_step': self.frame_step,
            'lower_edge_hertz': self.lower_edge_hertz,
            'upper_edge_hertz': self.upper_edge_hertz,
            'pad_end_stft': self.pad_end_stft
        })
        return config


@tf.function
def compute_fbanks(audio: Union[tf.Tensor, List, np.ndarray],
                   frame_length: int = 512,
                   frame_step: int = 160,
                   sample_rate: int = 16000,
                   num_mel_bins: int = 40,
                   lower_edge_hertz: int = 0,
                   upper_edge_hertz: int = 8000,
                   pad_end_stft: bool = True) -> tf.Tensor:
    stfts = tf.signal.stft(audio,
                           frame_length=frame_length,
                           frame_step=frame_step,
                           pad_end=pad_end_stft)
    spectrograms = tf.abs(stfts)

    num_spectrogram_bins = tf.shape(stfts)[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins,
        num_spectrogram_bins,
        sample_rate,
        lower_edge_hertz,
        upper_edge_hertz
    )

    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)

    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))

    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    return log_mel_spectrograms
