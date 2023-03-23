import numpy as np
import tensorflow as tf
from typing import List, Union
from tensorflow.keras.layers.experimental.preprocessing import \
    PreprocessingLayer


class Gain(PreprocessingLayer):

    def __init__(self,
                 prob: float = 0.5,
                 min_gain: float = 10,
                 max_gain: float = 12,
                 name: str = 'gain',
                 **kwargs):
        super().__init__(trainable=False, name=name, **kwargs)
        self.prob = prob
        self.min_gain = min_gain
        self.max_gain = max_gain

    def call(
        self,
        audio: Union[tf.Tensor, List[float], np.ndarray],
    ) -> tf.Tensor:
        gain = tf.random.uniform(shape=(),
                                 minval=self.min_gain,
                                 maxval=self.max_gain,
                                 dtype=tf.float32)
        augmented_audio = audio * 10**(gain / 20)
        prob = tf.random.uniform(shape=(),
                                 minval=0,
                                 maxval=1,
                                 dtype=tf.float32)
        return tf.cond(prob <= self.prob, lambda: augmented_audio,
                       lambda: audio)
