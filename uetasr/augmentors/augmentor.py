import numpy as np
import tensorflow as tf
from typing import List, Union
from tensorflow.keras.layers.experimental.preprocessing import \
    PreprocessingLayer


class Augmentor(PreprocessingLayer):
    """ Augmentor class"""

    def __init__(self,
                 augmentors: List[PreprocessingLayer],
                 name='augmentor',
                 **kwargs):
        super().__init__(trainable=False, name=name, **kwargs)
        self.augmentors = augmentors

    def call(
        self,
        audio: Union[tf.Tensor, List[float], np.ndarray]
    ) -> tf.Tensor:
        for augmentor in self.augmentors:
            audio = augmentor(audio)
        return audio
