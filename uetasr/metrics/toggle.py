import tensorflow as tf


class ToggleMetrics(tf.keras.callbacks.Callback):
    """ On test begin (i.e. when evaluate() is called or
    validation data is run during fit()) toggle metric flag """

    def __init__(self, metric):
        super(ToggleMetrics, self).__init__()
        self.metric = metric

    def on_test_begin(self, logs=None):
        self.metric.on.assign(True)

    def on_test_end(self, logs=None):
        self.metric.on.assign(False)
