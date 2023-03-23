from typing import Union, List

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


class LmLoss(tf.keras.losses.Loss):
    def __init__(self, blank: int = 0, name: str = None):
        super(LmLoss, self).__init__(reduction=tf.keras.losses.Reduction.NONE,
                                     name=name)
        self.blank = blank

    def call(self,
             labels: dict,
             logits: Union[tf.Tensor, np.ndarray, List]):

        not_blanks = tf.not_equal(labels, self.blank)
        labels_length = tf.reduce_sum(
            tf.cast(not_blanks, dtype=tf.int32), axis=-1) + 1

        mask = tf.sequence_mask(labels_length, maxlen=tf.shape(logits)[1])
        mask = tf.cast(mask, dtype=logits.dtype)
        loss = tfa.seq2seq.sequence_loss(logits=logits,
                                         targets=labels,
                                         weights=mask,
                                         average_across_timesteps=True,
                                         average_across_batch=True)

        return loss
