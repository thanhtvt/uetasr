import numpy as np
import tensorflow as tf
from typing import Union, List

try:
    from warpctc_tensorflow import ctc as warp_ctc_loss
    have_warpctc = True
except ImportError:
    have_warpctc = False


class CtcLoss(tf.keras.losses.Loss):

    def __init__(self,
                 blank: int = 0,
                 use_tf: bool = False,
                 name: str = "ctc_loss"):
        super(CtcLoss, self).__init__(reduction=tf.keras.losses.Reduction.NONE,
                                      name=name)
        self.blank = blank
        self.use_tf = use_tf

    def call(self, labels: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
        not_blanks = tf.not_equal(labels, self.blank)
        labels_length = tf.reduce_sum(tf.cast(not_blanks, dtype=tf.int32),
                                      axis=-1)
        logits_length = logits.length
        logits_length = tf.cast(logits_length, dtype=tf.int32)

        logits = tf.transpose(logits, [1, 0, 2])

        loss_fn = ctc_loss_tf if self.use_tf or not have_warpctc \
            else ctc_loss
        if not self.use_tf:
            labels = tf.reshape(labels, [-1])
            equal = tf.not_equal(labels, self.blank)
            labels = tf.gather_nd(labels, tf.where(equal))
            logits = tf.nn.log_softmax(logits, axis=-1)

        loss = loss_fn(logits=logits,
                       logits_length=logits_length,
                       labels=labels,
                       labels_length=labels_length,
                       blank=self.blank)
        is_inf = tf.math.is_inf(loss)
        loss = tf.where(is_inf, 0.0, loss)
        if not self.use_tf:
            batch_size = tf.shape(logits)[1]
        else:
            batch_size = tf.shape(logits)[0]
        return tf.nn.compute_average_loss(loss, global_batch_size=batch_size)


@tf.function
def ctc_loss(labels: Union[tf.Tensor, np.ndarray, List],
             logits: Union[tf.Tensor, np.ndarray, List],
             labels_length: Union[tf.Tensor, np.ndarray, List] = None,
             logits_length: Union[tf.Tensor, np.ndarray, List] = None,
             blank: int = 0) -> tf.Tensor:
    return warp_ctc_loss(activations=tf.cast(logits, tf.float32),
                         flat_labels=tf.cast(labels, tf.int32),
                         label_lengths=tf.cast(labels_length, tf.int32),
                         input_lengths=tf.cast(logits_length, tf.int32),
                         blank_label=blank)


@tf.function
def ctc_loss_tf(labels: Union[tf.Tensor, np.ndarray, List],
                logits: Union[tf.Tensor, np.ndarray, List],
                labels_length: Union[tf.Tensor, np.ndarray, List] = None,
                logits_length: Union[tf.Tensor, np.ndarray, List] = None,
                blank: int = 0) -> tf.Tensor:
    return tf.nn.ctc_loss(logits=tf.cast(logits, tf.float32),
                          labels=tf.cast(labels, tf.int32),
                          label_length=tf.cast(labels_length, tf.int32),
                          logit_length=tf.cast(logits_length, tf.int32),
                          logits_time_major=True,
                          blank_index=blank)
