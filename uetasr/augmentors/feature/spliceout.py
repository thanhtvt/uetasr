import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing \
    import PreprocessingLayer
from typing import Union, List


class SpliceOut(PreprocessingLayer):

    def __init__(self,
                 prob: float = 0.5,
                 max_splice_out_ratio: float = 0.1,
                 splice_out_num: int = 2,
                 name: str = 'splice_out',
                 **kwargs):
        super().__init__(trainable=False, name=name, **kwargs)
        self.prob = prob
        self.max_splice_out_ratio = tf.cast(max_splice_out_ratio,
                                            dtype=tf.float32)
        self.splice_out_num = splice_out_num

    def call(
        self,
        feature: Union[tf.Tensor, List[float], np.ndarray],
    ) -> tf.Tensor:
        augmented_feature = feature
        tau = tf.shape(feature)[0]
        max_splice_out_width = tf.cast(
            tf.cast(tf.shape(feature)[0], dtype=tf.float32) *
            self.max_splice_out_ratio,
            dtype=tf.int32
        )
        max_splice_out_width = tf.math.maximum(max_splice_out_width, 1)
        mask = tf.ones(shape=(tau,), dtype=tf.bool)
        for _ in range(self.splice_out_num):
            length = tf.random.uniform(shape=(),
                                       minval=1,
                                       maxval=max_splice_out_width,
                                       dtype=tf.int32)
            start = tf.random.uniform(shape=(),
                                      minval=0,
                                      maxval=tau - length,
                                      dtype=tf.int32)
            mask_indices = tf.range(start, start + length)
            mask = tf.tensor_scatter_nd_update(mask,
                                               mask_indices[:, None],
                                               [False] * length)
        augmented_feature = feature[mask]
        prob = tf.random.uniform(shape=(),
                                 minval=0,
                                 maxval=1,
                                 dtype=tf.float32)
        return tf.cond(prob <= self.prob,
                       lambda: augmented_feature,
                       lambda: feature)
