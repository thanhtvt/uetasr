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


def fix_length(audio: tf.Tensor, target_size: int, axis: int = -1):
    n = audio.shape[axis]
    if n > target_size:
        slices = [slice(None)] * audio.ndim
        slices[axis] = slice(0, target_size)
        return audio[tuple(slices)]
    elif n < target_size:
        lengths = [(0, 0)] * audio.ndim
        lengths[axis] = (0, target_size - n)
        return tf.pad(audio, lengths)

    return audio
