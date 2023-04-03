import numpy as np
import tensorflow as tf
from typing import Union, List
from tensorflow.python.ops.gen_array_ops import matrix_diag_part_v2

from ..utils.common import has_devices
try:
    from warprnnt_tensorflow import rnnt_loss as warp_rnnt_loss
    have_warprnnt = True
except ImportError:
    have_warprnnt = False


class RnntLoss(tf.keras.losses.Loss):

    def __init__(
        self,
        blank: int = 0,
        use_logsoftmax: bool = False,
        use_tf: bool = False,
        name: str = "rnnt_loss",
    ):
        super(RnntLoss, self).__init__(
            reduction=tf.keras.losses.Reduction.NONE,
            name=name
        )
        self.blank = blank
        self.use_logsoftmax = use_logsoftmax
        self.use_tf = use_tf

    def call(self, labels: Union[tf.Tensor, np.ndarray, List],
             logits: Union[tf.Tensor, np.ndarray, List]) -> tf.Tensor:

        not_blanks = tf.not_equal(labels, self.blank)
        labels_length = tf.reduce_sum(tf.cast(not_blanks, tf.int32), axis=-1)
        logits_length = tf.cast(logits.length, tf.int32)

        if self.use_logsoftmax:
            logits = tf.nn.log_softmax(logits, axis=-1)

        loss = self.rnnt_loss(logits=logits,
                              labels=labels,
                              label_lengths=labels_length,
                              logit_lengths=logits_length,
                              blank=self.blank,
                              name=self.name)

        batch_size = tf.shape(logits)[0]
        return tf.nn.compute_average_loss(loss, global_batch_size=batch_size)

    @tf.function
    def rnnt_loss(
        self,
        logits: tf.Tensor,
        labels: tf.Tensor,
        label_lengths: tf.Tensor,
        logit_lengths: tf.Tensor,
        blank: int = 0,
        name: str = None,
    ):
        if self.use_tf or not have_warprnnt:
            return rnnt_loss_tf(logits=logits,
                                labels=labels,
                                label_length=label_lengths,
                                logit_length=logit_lengths,
                                name=name)
        else:
            return warp_rnnt_loss(
                acts=tf.cast(logits, tf.float32),
                labels=tf.cast(labels, tf.int32),
                label_lengths=tf.cast(label_lengths, tf.int32),
                input_lengths=tf.cast(logit_lengths, tf.int32),
                blank_label=blank,
            )


def nan_to_zero(input_tensor):
    return tf.where(tf.math.is_nan(input_tensor), tf.zeros_like(input_tensor),
                    input_tensor)


def reduce_logsumexp(
    input_tensor,
    axis,
):
    maximum = tf.reduce_max(input_tensor, axis=axis)
    input_tensor = nan_to_zero(input_tensor - maximum)
    return tf.math.log(tf.reduce_sum(tf.exp(input_tensor),
                                     axis=axis)) + maximum


def extract_diagonals(log_probs, ):
    time_steps = tf.shape(log_probs)[1]  # T
    output_steps = tf.shape(log_probs)[2]  # U + 1
    reverse_log_probs = tf.reverse(log_probs, axis=[-1])
    paddings = [[0, 0], [0, 0], [time_steps - 1, 0]]
    padded_reverse_log_probs = tf.pad(reverse_log_probs,
                                      paddings,
                                      "CONSTANT",
                                      constant_values=float("-inf"))
    diagonals = matrix_diag_part_v2(
        padded_reverse_log_probs,
        k=(0, time_steps + output_steps - 2),
        padding_value=float("-inf"),
    )

    return tf.transpose(diagonals, perm=[1, 0, 2])


def transition_probs(
    one_hot_labels,
    log_probs,
):
    """
    :return: blank_probs (batch_size x input_max_len x target_max_len)
             truth_probs (batch_size x input_max_len x (target_max_len - 1))
    """
    blank_probs = log_probs[:, :, :, 0]
    truth_probs = tf.reduce_sum(tf.multiply(log_probs[:, :, :-1, :],
                                            one_hot_labels),
                                axis=-1)

    return blank_probs, truth_probs


def forward_dp(
    bp_diags,
    tp_diags,
    batch_size,
    input_max_len,
    target_max_len,
):
    """
    :return: forward variable alpha with
             shape batch_size x input_max_len x target_max_len
    """

    def next_state(x, trans_probs):
        blank_probs = trans_probs[0]
        truth_probs = trans_probs[1]

        x_b = tf.concat([
            float("-inf") * tf.ones(shape=[batch_size, 1]),
            x[:, :-1] + blank_probs
        ], axis=1)
        x_t = x + truth_probs

        x = tf.math.reduce_logsumexp(tf.stack([x_b, x_t], axis=0), axis=0)
        return x

    initial_alpha = tf.concat(
        [
            tf.zeros(shape=[batch_size, 1]),
            tf.ones(shape=[batch_size, input_max_len - 1]) * float("-inf"),
        ],
        axis=1,
    )

    fwd = tf.scan(next_state, (bp_diags[:-1, :, :-1], tp_diags),
                  initializer=initial_alpha)

    alpha = tf.transpose(tf.concat(
        [tf.expand_dims(initial_alpha, axis=0), fwd], axis=0),
                         perm=[1, 2, 0])
    alpha = matrix_diag_part_v2(alpha,
                                k=(0, target_max_len - 1),
                                padding_value=float("-inf"))
    alpha = tf.transpose(tf.reverse(alpha, axis=[1]), perm=[0, 2, 1])

    return alpha


def backward_dp(
    bp_diags,
    tp_diags,
    batch_size,
    input_max_len,
    target_max_len,
    label_length,
    logit_length,
    blank_sl,
):
    """
    :return: backward variable beta with shape
             (batch_size x input_max_len x target_max_len)
    """

    def next_state(x, mask_and_trans_probs):
        mask_s, blank_probs_s, truth_probs = mask_and_trans_probs

        beta_b = tf.concat([
            x[:, 1:] + blank_probs_s,
            float("-inf") * tf.ones(shape=[batch_size, 1])
        ], axis=1)
        beta_t = tf.concat([
            x[:, :-1] + truth_probs,
            float("-inf") * tf.ones(shape=[batch_size, 1])
        ], axis=1)

        beta_next = reduce_logsumexp(tf.stack([beta_b, beta_t], axis=0),
                                     axis=0)
        masked_beta_next = nan_to_zero(
            beta_next * tf.expand_dims(mask_s, axis=1)) + nan_to_zero(
                x * tf.expand_dims((1.0 - mask_s), axis=1))
        return tf.reshape(masked_beta_next, shape=tf.shape(x))

    # Initial beta for batches.
    initial_beta_mask = tf.one_hot(logit_length - 1, depth=input_max_len + 1)
    initial_beta = tf.expand_dims(
        blank_sl, axis=1) * initial_beta_mask + nan_to_zero(
            float("-inf") * (1.0 - initial_beta_mask))

    # Mask for scan iterations.
    mask = tf.sequence_mask(
        logit_length + label_length - 1,
        input_max_len + target_max_len - 2,
        dtype=tf.dtypes.float32,
    )
    mask = tf.transpose(mask, perm=[1, 0])

    bwd = tf.scan(
        next_state,
        (mask, bp_diags[:-1, :, :], tp_diags),
        initializer=initial_beta,
        reverse=True,
    )

    beta = tf.transpose(tf.concat(
        [bwd, tf.expand_dims(initial_beta, axis=0)], axis=0),
                        perm=[1, 2, 0])[:, :-1, :]
    beta = matrix_diag_part_v2(beta,
                               k=(0, target_max_len - 1),
                               padding_value=float("-inf"))
    beta = tf.transpose(tf.reverse(beta, axis=[1]), perm=[0, 2, 1])

    return beta


def compute_rnnt_loss_and_grad_helper(logits, labels, label_length,
                                      logit_length):
    batch_size = tf.shape(logits)[0]
    input_max_len = tf.shape(logits)[1]
    target_max_len = tf.shape(logits)[2]
    vocab_size = tf.shape(logits)[3]

    one_hot_labels = tf.one_hot(
        tf.tile(tf.expand_dims(labels, axis=1),
                multiples=[1, input_max_len, 1]),
        depth=vocab_size,
    )

    log_probs = tf.nn.log_softmax(logits)
    blank_probs, truth_probs = transition_probs(one_hot_labels, log_probs)
    bp_diags = extract_diagonals(blank_probs)
    tp_diags = extract_diagonals(truth_probs)

    label_mask = tf.expand_dims(
        tf.sequence_mask(label_length + 1,
                         maxlen=target_max_len,
                         dtype=tf.float32),
        axis=1,
    )
    small_label_mask = tf.expand_dims(tf.sequence_mask(label_length,
                                                       maxlen=target_max_len,
                                                       dtype=tf.float32),
                                      axis=1)
    input_mask = tf.expand_dims(tf.sequence_mask(logit_length,
                                                 maxlen=input_max_len,
                                                 dtype=tf.float32),
                                axis=2)
    small_input_mask = tf.expand_dims(
        tf.sequence_mask(logit_length - 1,
                         maxlen=input_max_len,
                         dtype=tf.float32),
        axis=2,
    )
    mask = label_mask * input_mask
    grad_blank_mask = (label_mask * small_input_mask)[:, :-1, :]
    grad_truth_mask = (small_label_mask * input_mask)[:, :, :-1]

    alpha = forward_dp(bp_diags, tp_diags, batch_size, input_max_len,
                       target_max_len) * mask

    indices = tf.stack([logit_length - 1, label_length], axis=1)
    blank_sl = tf.gather_nd(blank_probs, indices, batch_dims=1)

    beta = (backward_dp(
        bp_diags,
        tp_diags,
        batch_size,
        input_max_len,
        target_max_len,
        label_length,
        logit_length,
        blank_sl,
    ) * mask)
    beta = tf.where(tf.math.is_nan(beta), tf.zeros_like(beta), beta)
    final_state_probs = beta[:, 0, 0]

    # Compute gradients of loss w.r.t. blank log-probabilities.
    grads_blank = (-tf.exp(
        (alpha[:, :-1, :] + beta[:, 1:, :] -
         tf.reshape(final_state_probs, shape=[batch_size, 1, 1]) +
         blank_probs[:, :-1, :]) * grad_blank_mask) * grad_blank_mask)
    grads_blank = tf.concat(
        [grads_blank,
         tf.zeros(shape=(batch_size, 1, target_max_len))], axis=1)
    last_grads_blank = -1 * tf.scatter_nd(
        tf.concat(
            [
                tf.reshape(tf.range(batch_size, dtype=tf.int64),
                           shape=[batch_size, 1]),
                tf.cast(indices, dtype=tf.int64),
            ],
            axis=1,
        ),
        tf.ones(batch_size, dtype=tf.float32),
        [batch_size, input_max_len, target_max_len],
    )
    grads_blank = grads_blank + last_grads_blank

    # Compute gradients of loss w.r.t. truth log-probabilities.
    grads_truth = (-tf.exp((alpha[:, :, :-1] + beta[:, :, 1:] - tf.reshape(
        final_state_probs, shape=[batch_size, 1, 1]) + truth_probs) *
                           grad_truth_mask) * grad_truth_mask)

    # Compute gradients of loss w.r.t. activations.
    a = tf.tile(
        tf.reshape(
            tf.range(target_max_len - 1, dtype=tf.int64),
            shape=(1, 1, target_max_len - 1, 1),
        ),
        multiples=[batch_size, 1, 1, 1],
    )
    b = tf.cast(
        tf.reshape(labels - 1, shape=(batch_size, 1, target_max_len - 1, 1)),
        dtype=tf.int64,
    )
    if not has_devices("GPU"):
        # for cpu testing (index -1 on cpu will raise errors)
        b = tf.where(tf.equal(b, -1), tf.zeros_like(b), b)
    c = tf.concat([a, b], axis=3)
    d = tf.tile(c, multiples=(1, input_max_len, 1, 1))
    e = tf.tile(
        tf.reshape(tf.range(input_max_len, dtype=tf.int64),
                   shape=(1, input_max_len, 1, 1)),
        multiples=(batch_size, 1, target_max_len - 1, 1),
    )
    f = tf.concat([e, d], axis=3)
    g = tf.tile(
        tf.reshape(tf.range(batch_size, dtype=tf.int64),
                   shape=(batch_size, 1, 1, 1)),
        multiples=[1, input_max_len, target_max_len - 1, 1],
    )
    scatter_idx = tf.concat([g, f], axis=3)
    # TODO - improve the part of code for scatter_idx computation.
    probs = tf.exp(log_probs)
    grads_truth_scatter = tf.scatter_nd(
        scatter_idx,
        grads_truth,
        [batch_size, input_max_len, target_max_len, vocab_size - 1],
    )
    grads = tf.concat(
        [
            tf.reshape(grads_blank,
                       shape=(batch_size, input_max_len, target_max_len, -1)),
            grads_truth_scatter,
        ],
        axis=3,
    )
    grads_logits = grads - probs * (tf.reduce_sum(grads, axis=3,
                                                  keepdims=True))

    loss = -final_state_probs
    return loss, grads_logits


def rnnt_loss_tf(
    logits,
    labels,
    label_length,
    logit_length,
    name=None,
):
    name = "rnnt_loss" if name is None else name
    with tf.name_scope(name):
        logits = tf.convert_to_tensor(logits, name="logits")
        labels = tf.convert_to_tensor(labels, name="labels")
        label_length = tf.convert_to_tensor(label_length, name="label_length")
        logit_length = tf.convert_to_tensor(logit_length, name="logit_length")

        args = [logits, labels, label_length, logit_length]

        @tf.custom_gradient
        def compute_rnnt_loss_and_grad(logits_t, labels_t, label_length_t,
                                       logit_length_t):
            """Compute RNN-T loss and gradients."""
            logits_t.set_shape(logits.shape)
            labels_t.set_shape(labels.shape)
            label_length_t.set_shape(label_length.shape)
            logit_length_t.set_shape(logit_length.shape)
            kwargs = dict(
                logits=logits_t,
                labels=labels_t,
                label_length=label_length_t,
                logit_length=logit_length_t,
            )
            result = compute_rnnt_loss_and_grad_helper(**kwargs)

            def grad(grad_loss):
                grads = [tf.reshape(grad_loss, [-1, 1, 1, 1]) * result[1]]
                grads += [None] * (len(args) - len(grads))
                return grads

            return result[0], grad

        return compute_rnnt_loss_and_grad(*args)
