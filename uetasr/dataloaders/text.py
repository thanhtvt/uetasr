import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing \
    import PreprocessingLayer
from typing import List


class TextDataloader(tf.data.Dataset):

    def __new__(cls, data: tf.data.Dataset,
                text_encoder: PreprocessingLayer,
                text_max_length: int = 400,
                text_min_length: int = 1,
                use_bucket: bool = True,
                bucket_boundaries: List[float] = [5, 10],
                bucket_batch_sizes: List[int] = [64, 32, 16],
                batch_size: int = 8,
                drop_remainder: bool = True,
                teacher_forcing: bool = True,
                shuffle: bool = True,
                shuffle_buffer_size: int = 1024,
                num_parallel_calls: int = 64,
                name: str = 'text_dataloader'):

        if shuffle:
            data = data.shuffle(shuffle_buffer_size,
                                reshuffle_each_iteration=True)

        def length_fn(text):
            return tf.strings.length(text, unit="UTF8_CHAR")

        data = data.filter(
            lambda label: length_fn(label) <= text_max_length
        )
        data = data.filter(
            lambda label: length_fn(label) >= text_min_length
        )

        if use_bucket:
            data = data.bucket_by_sequence_length(
                element_length_func=lambda label: length_fn(label),
                bucket_boundaries=bucket_boundaries,
                bucket_batch_sizes=bucket_batch_sizes,
                padded_shapes=None,
                padding_values='',
                drop_remainder=drop_remainder
            )
        else:
            data = data.padded_batch(
                batch_size,
                padded_shapes=None,
                padding_values='',
                drop_remainder=drop_remainder
            )

        if teacher_forcing:
            data = data.map(lambda label: (text_encoder(label, preblank=True),
                                           text_encoder(label, postblank=True)),
                            num_parallel_calls=num_parallel_calls)

        data = data.prefetch(tf.data.AUTOTUNE)
        data.name = name
        return data