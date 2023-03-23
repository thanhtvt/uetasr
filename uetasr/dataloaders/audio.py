import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.keras.layers.experimental.preprocessing import \
    PreprocessingLayer


class AudioDataloader(tf.data.Dataset):

    def __new__(cls, data_path: str,
                audio_encoder: PreprocessingLayer = None,
                audio_type: str = 'wav',
                sample_rate: int = 16000,
                audio_max_length: int = 35,
                audio_min_length: int = 1,
                shuffle: bool = False,
                shuffle_buffer_size: int = 1024,
                num_parallel_calls: int = 64,
                name: str = "audio_dataloader"):

        data = tf.data.TextLineDataset(data_path)

        if shuffle:
            data = data.shuffle(shuffle_buffer_size,
                                reshuffle_each_iteration=True)

        data = data.map(
            lambda audio_path: tf.io.read_file(audio_path),
            num_parallel_calls=num_parallel_calls
        )

        if audio_type == "wav":
            data = data.map(
                lambda audio: tf.audio.decode_wav(audio)[0],
                num_parallel_calls=num_parallel_calls
            )
        elif audio_type == "flac":
            data = data.map(
                lambda audio: tf.cast(tfio.audio.decode_flac(audio, dtype=tf.int16),
                                      dtype=tf.float32) / 32768,
                num_parallel_calls=num_parallel_calls
            )
        else:
            raise NotImplementedError(f"Audio type {audio_type} not supported")

        data = data.map(
            lambda audio: tf.squeeze(audio, axis=-1),
            num_parallel_calls=num_parallel_calls
        )

        data = data.filter(
            lambda audio: tf.shape(audio)[0] <= audio_max_length * sample_rate
        )
        data = data.filter(
            lambda audio: tf.shape(audio)[0] >= audio_min_length * sample_rate
        )

        if audio_encoder:
            data = data.map(
                lambda audio: audio_encoder(audio),
                num_parallel_calls=num_parallel_calls
            )

        data = data.prefetch(tf.data.AUTOTUNE)
        data.name = name
        return data
