import tensorflow as tf


class LRLogger(tf.keras.callbacks.Callback):

    def __init__(self, name: str = "lr_logger", **kwargs):
        super(LRLogger, self).__init__(name=name, **kwargs)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        steps = self.model.optimizer.iterations
        lr = self.model.optimizer.learning_rate(steps).numpy().item()
        # print()
        # print('steps', steps.numpy().item())
        # print('learning rate', lr)
        logs['steps'] = steps
        logs['lr'] = lr
