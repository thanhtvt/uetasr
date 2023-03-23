import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.keras.layers.experimental.preprocessing \
    import PreprocessingLayer
from typing import Union, List


class TimeMask(PreprocessingLayer):

    def __init__(self,
                 prob: float = 0.5,
                 time_masking_ratio: float = 0.05,
                 time_mask_num: int = 2,
                 name='time_mask',
                 **kwargs):
        super().__init__(trainable=False, name=name, **kwargs)
        self.prob = prob
        self.time_masking_ratio = tf.cast(time_masking_ratio, dtype=tf.float32)
        self.time_mask_num = time_mask_num

    def call(
        self,
        feature: Union[tf.Tensor, List[float], np.ndarray],
    ) -> tf.Tensor:
        augmented_feature = feature
        time_masking_para = tf.cast(
            tf.cast(tf.shape(feature)[0], dtype=tf.float32) *
            self.time_masking_ratio,
            dtype=tf.int32
        )
        time_masking_para = tf.math.maximum(time_masking_para, 1)
        for _ in range(self.time_mask_num):
            augmented_feature = tfio.audio.time_mask(augmented_feature,
                                                     param=time_masking_para)
        prob = tf.random.uniform(shape=(),
                                 minval=0,
                                 maxval=1,
                                 dtype=tf.float32)
        return tf.cond(prob <= self.prob,
                       lambda: augmented_feature,
                       lambda: feature)


class FrequencyMask(PreprocessingLayer):

    def __init__(self,
                 prob: float = 0.5,
                 frequency_masking_para: int = 27,
                 frequency_mask_num: int = 2,
                 name='freq_mask',
                 **kwargs):
        super().__init__(trainable=False, name=name, **kwargs)
        self.prob = prob
        self.frequency_masking_para = frequency_masking_para
        self.frequency_mask_num = frequency_mask_num

    def call(
        self,
        feature: Union[tf.Tensor, List[float], np.ndarray],
    ) -> tf.Tensor:
        augmented_feature = feature
        for _ in range(self.frequency_mask_num):
            augmented_feature = tfio.audio.freq_mask(
                augmented_feature, param=self.frequency_masking_para)
        prob = tf.random.uniform(shape=(),
                                 minval=0,
                                 maxval=1,
                                 dtype=tf.float32)

        return tf.cond(prob <= self.prob,
                       lambda: augmented_feature,
                       lambda: feature)
