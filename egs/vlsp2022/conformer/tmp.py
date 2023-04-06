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


def read_audio(filepath):
    audio = tf.io.read_file(filepath)
    _, file_ext = os.path.splitext(filepath)
    if file_ext == ".wav":
        audio = tf.audio.decode_wav(audio)[0]
    elif file_ext == ".flac":
        audio = tf.cast(tfio.audio.decode_flac(audio, dtype=tf.int16),
                        dtype=tf.float32) / 32768
    else:
        raise ValueError(f"Unsupported audio format: {file_ext}")

    audio = tf.expand_dims(tf.squeeze(audio, axis=-1), axis=0)
    return audio


def test_model(saved_dir: str, ckpt_path: str, filepath: str, name: str = "conformer"):
    # Define keras Input
    # inputs = tf.keras.Input(shape=(None,), batch_size=None, dtype=tf.float32)
    inputs = read_audio(filepath)
    # Define model procedure
    encoder_model = modules["encoder_model"]
    searcher = modules["decoder"]
    model = modules["model"]
    model.load_weights(ckpt_path).expect_partial()

    featurizer = modules["audio_encoder"]
    features = featurizer(inputs)
    features = model.cmvn(features) if model.use_cmvn else features

    batch_size = tf.shape(features)[0]
    dim = tf.shape(features)[-1]
    mask = tf.sequence_mask([tf.shape(features)[1]], maxlen=tf.shape(features)[1])
    mask = tf.expand_dims(mask, axis=1)
    encoder_outputs, encoder_masks = encoder_model(
        features, mask, training=False)

    encoder_mask = tf.squeeze(encoder_masks, axis=1)
    features_length = tf.math.reduce_sum(
        tf.cast(encoder_mask, tf.int32),
        axis=1
    )
    
    outputs = searcher(encoder_outputs, features_length)
    outputs = tf.squeeze(outputs).numpy().decode("utf-8")
    # outputs = [out.decode("utf-8") for out in outputs.numpy()]
    print(outputs)
    # Define keras model
    # model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    # tf.saved_model.save(model, saved_dir)
    # print(f"Saved model to {saved_dir}")


os.makedirs(modules["saved_dir"], exist_ok=True)
test_model(modules["saved_dir"], modules["ckpt_path"],
           filepath="/data/vlsp2022/training/labeled/wav/121-00427081-00427287.wav")
