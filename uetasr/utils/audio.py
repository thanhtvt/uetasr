import tensorflow as tf


def load_audio(audio_path: str):
    audio = tf.io.read_file(audio_path)
    audio, sr = tf.audio.decode_wav(audio)
    audio = tf.squeeze(audio, axis=-1)
    return audio, sr


def save_audio(audio: tf.Tensor, sample_rate: int, audio_path: str):
    audio = tf.convert_to_tensor(audio)
    if len(audio.shape) == 1:
        audio = tf.expand_dims(audio, axis=1)
    encoded = tf.audio.encode_wav(audio, sample_rate)
    tf.io.write_file(audio_path, encoded)
