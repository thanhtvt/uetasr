import tensorflow as tf


class CharacterErrorRate(tf.keras.metrics.Metric):
    """ Compute the Charactor Error Rate """

    def __init__(self, accumulate: bool = False, name: str = 'CER', **kwargs):
        super(CharacterErrorRate, self).__init__(name=name, **kwargs)
        self.cer_accumulator = tf.Variable(0.0,
                                           name="cer_accumulator",
                                           dtype=tf.float32)
        self.counter = tf.Variable(0, name="cer_counter", dtype=tf.int32)
        self.on = tf.Variable(False, name="cer_on", dtype=tf.bool)
        self.accumulate = accumulate

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.on:
            input_shape = tf.keras.backend.shape(y_pred)

            input_length = tf.ones(shape=input_shape[0],
                                   dtype='int32') * tf.cast(
                                       input_shape[1], tf.int32)

            decode, _ = tf.keras.backend.ctc_decode(y_pred,
                                                    input_length,
                                                    greedy=True)

            decode = tf.keras.backend.ctc_label_dense_to_sparse(
                decode[0], input_length)
            y_true_sparse = tf.cast(
                tf.keras.backend.ctc_label_dense_to_sparse(
                    y_true, input_length
                ),
                dtype=tf.int64
            )

            decode = tf.sparse.retain(decode, tf.not_equal(decode.values, -1))
            distance = tf.edit_distance(decode, y_true_sparse, normalize=True)

            self.cer_accumulator.assign_add(tf.reduce_sum(distance))
            self.counter.assign_add(len(y_true))

    def result(self):
        return {
            "CER":
            tf.math.divide_no_nan(self.cer_accumulator,
                                  tf.cast(self.counter, tf.float32))
        }

    def reset_states(self):
        if self.accumulate:
            self.cer_accumulator.assign(0.0)
            self.counter.assign(0)
