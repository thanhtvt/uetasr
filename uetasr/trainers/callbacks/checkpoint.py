import tensorflow as tf


class TopKModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    """ Inherit ModelCheckpoint to save the top k models """
    def __init__(self, save_top_k: int = 5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ckpt_list = tf.queue.FIFOQueue(save_top_k, [tf.string])
        self.top_k = save_top_k

    def _save_model(self, epoch, batch, logs):
        """ Inherit _save_model to save the top k models

        Args:
            epoch: the epoch this iteration is in.
            batch: the batch this iteration is in. `None` if the `save_freq`
              is set to `epoch`.
            logs: the `logs` dict passed into `on_batch_end` or `on_epoch_end`.
        """
        super()._save_model(epoch, batch, logs)
        filepath = super()._get_file_path(epoch, batch, logs)
        current_size = self.ckpt_list.size().numpy()
        if current_size < self.top_k:
            self.ckpt_list.enqueue([filepath])
        else:
            removed_ckpt = self.ckpt_list.dequeue()
            self.ckpt_list.enqueue([filepath])
            removed_ckpts = tf.io.gfile.glob(
                removed_ckpt.numpy().decode() + "*"
            )
            for ckpt in removed_ckpts:
                tf.io.gfile.remove(ckpt)
