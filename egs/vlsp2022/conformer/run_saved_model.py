import argparse
import numpy as np
import os
import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_text as tftext
from pathlib import Path
from time import perf_counter


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


def run_saved_model(saved_model: str, audio_path: str):
    module = tf.saved_model.load(export_dir=saved_model)
    signal = read_audio(audio_path)
    time1 = perf_counter()
    outputs = module(signal)
    outputs = tf.squeeze(outputs).numpy().decode("utf-8")
    time2 = perf_counter()
    print(f"Time: {time2 - time1}")
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a SavedModel",
    )
    parser.add_argument("saved_model", type=str, help="Path to saved model.")
    parser.add_argument("--audio_path", type=str, help="Path to audio file.")
    parser.add_argument("--devices", default="-1", type=str,
                        help="Devices for training, separate by comma.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    np.random.seed(0)

    pred = run_saved_model(args.saved_model, args.audio_path)
    print(pred)
