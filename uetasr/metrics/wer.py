import tensorflow as tf


class WordErrorRate(tf.keras.metrics.Metric):
    """ Compute the Word Error Rate """
    def __init__(self, name: str = 'WER', **kwargs):
        super(WordErrorRate, self).__init__(name=name, **kwargs)
        self.wer_accumulator = tf.Variable(0.0, name="wer_accumulator", dtype=tf.float32)
        self.counter = tf.Variable(0, name="counter", dtype=tf.int32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        input_shape = tf.keras.backend.shape(y_pred)

        input_length = tf.ones(shape=input_shape[0], dtype='int32') * tf.cast(input_shape[1], 'int32')

        decode, _ = tf.keras.backend.ctc_decode(y_pred, input_length, greedy=True)

        decode = tf.keras.backend.ctc_label_dense_to_sparse(decode[0], input_length)
        y_true_sparse = tf.cast(tf.keras.backend.ctc_label_dense_to_sparse(y_true, input_length), "int64")

        decode = tf.sparse.retain(decode, tf.not_equal(decode.values, -1))
        distance = tf.edit_distance(decode, y_true_sparse, normalize=True)

        correct_words_amount = tf.reduce_sum(tf.cast(tf.not_equal(distance, 0), tf.float32))

        self.wer_accumulator.assign_add(correct_words_amount)
        self.counter.assign_add(len(y_true))

    def result(self):
        return {
                "WER": tf.math.divide_no_nan(self.wer_accumulator, tf.cast(self.counter, tf.float32))
        }
