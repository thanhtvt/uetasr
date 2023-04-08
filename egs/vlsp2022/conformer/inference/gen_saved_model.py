import numpy as np
np.random.seed(1337)
import os
import pyrootutils
import sys
import tensorflow as tf
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path

tf.get_logger().setLevel("DEBUG")

pyrootutils.setup_root(
    Path(__file__).resolve().parents[3],
    indicator=".project_root",
    pythonpath=True
)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
config_file = sys.argv[1]

with open(config_file) as fin:
    modules = load_hyperpyyaml(fin)


class Conformer(tf.Module):
    def __init__(
        self,
        audio_encoder: tf.keras.layers.experimental.preprocessing.PreprocessingLayer,
        decoder: tf.keras.layers.Layer,
        encoder: tf.keras.layers.Layer,
        model: tf.keras.Model,
        checkpoint_path: str,
        name: str = "conformer"
    ):
        super().__init__(name=name)
        self.model = model
        self.audio_encoder = audio_encoder
        self.decoder = decoder
        self.encoder = encoder
        self.model.load_weights(checkpoint_path).expect_partial()

    @tf.function(input_signature=[tf.TensorSpec(shape=[1, None], dtype=tf.float32)])
    def pred(self, signal):
        features = self.audio_encoder(signal)
        features = self.model.cmvn(features) if self.model.use_cmvn else features

        mask = tf.sequence_mask([tf.shape(features)[1]], maxlen=tf.shape(features)[1])
        mask = tf.expand_dims(mask, axis=1)
        encoder_outputs, encoder_masks = self.encoder(
            features, mask, training=False)

        encoder_mask = tf.squeeze(encoder_masks, axis=1)
        features_length = tf.math.reduce_sum(
            tf.cast(encoder_mask, tf.int32),
            axis=1
        )
        
        # not utf-8 decoded
        outputs = self.decoder(encoder_outputs, features_length)
        return outputs

module = Conformer(
    modules["audio_encoder"],
    modules["decoder"],
    modules["encoder_model"],
    modules["model"],
    modules["ckpt_path"]
)
tf.saved_model.save(module, modules["saved_dir"], signatures=module.pred.get_concrete_function())
print("Saved model to {}".format(modules["saved_dir"]))
