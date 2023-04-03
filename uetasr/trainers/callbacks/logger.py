import tensorflow as tf


class LRLogger(tf.keras.callbacks.Callback):

    def __init__(self, **kwargs):
        super(LRLogger, self).__init__(**kwargs)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        steps = self.model.optimizer.iterations
        try:
            lr = self.model.optimizer.learning_rate(steps).numpy().item()
        except TypeError:
            lr = self.model.optimizer.learning_rate.numpy().item()
        logs['steps'] = steps
        logs['lr'] = lr
