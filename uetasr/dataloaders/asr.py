import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.keras.layers.experimental.preprocessing import \
    PreprocessingLayer
from typing import List


def normalize_audio(audio: tf.Tensor):
    return (audio - tf.math.reduce_mean(audio)) / \
        tf.math.sqrt(tf.math.reduce_variance(audio) + 1e-7)


class ASRDataloader(tf.data.Dataset):

    def __new__(cls,
                data: tf.data.Dataset,
                text_encoder: PreprocessingLayer,
                audio_encoder: PreprocessingLayer = None,
                audio_augmentor: PreprocessingLayer = None,
                feature_augmentor: PreprocessingLayer = None,
                use_ctc_target: bool = False,
                num_parallel_calls: int = -1,
                shuffle: bool = True,
                use_audio_path: bool = False,
                pad_ms: int = 0,
                pad_segment: int = None,
                shuffle_buffer_size: int = 1024,
                sample_rate: int = 16000,
                audio_type: str = 'wav',
                audio_max_length: int = 20,
                audio_min_length: int = 1,
                text_max_length: int = 400,
                text_min_length: int = 1,
                use_bucket: bool = True,
                use_norm: bool = False,
                bucket_boundaries: List[float] = [5, 10],
                bucket_batch_sizes: List[int] = [768, 384, 192],
                batch_size: int = 4,
                drop_remainder: bool = True,
                teacher_forcing: bool = True,
                name: str = 'dataloader'):

        sr = audio_encoder.sample_rate if audio_encoder else sample_rate
        frame_step = sr // audio_encoder.frame_step if audio_encoder else sr
        subsampling_factor = audio_encoder.window_step if audio_encoder else 1
        append_eos = True if text_encoder.eos_id >= 0 else False

        if shuffle:
            data = data.shuffle(shuffle_buffer_size,
                                reshuffle_each_iteration=True)

        data = data.map(lambda audio_path, label:
                        (audio_path, tf.io.read_file(audio_path), label),
                        num_parallel_calls=num_parallel_calls)

        if audio_type == 'wav':
            data = data.map(lambda audio_path, audio, label:
                            (audio_path, tf.audio.decode_wav(audio)[0], label),
                            num_parallel_calls=num_parallel_calls)
        elif audio_type == 'flac':
            data = data.map(
                lambda audio_path, audio, label:
                (audio_path,
                 tf.cast(tfio.audio.decode_flac(audio, dtype=tf.int16),
                         dtype=tf.float32) / 32768, label),
                num_parallel_calls=num_parallel_calls)

        data = data.map(lambda audio_path, audio, label:
                        (audio_path, tf.squeeze(audio, axis=-1), label),
                        num_parallel_calls=num_parallel_calls)

        pad_frame = int(pad_ms / 1000 * sample_rate)
        if pad_segment is None:
            pad_segment = int(pad_frame / 160)
        data = data.map(lambda autio_path, audio, label:
                        (autio_path, tf.pad(audio, [[0, pad_frame]]), label),
                        num_parallel_calls=num_parallel_calls)

        def length_fn(text):
            return tf.strings.length(text, unit="UTF8_CHAR")

        if audio_augmentor:
            data = data.map(lambda audio_path, audio, label:
                            (audio_path, audio_augmentor(audio), label),
                            num_parallel_calls=num_parallel_calls)

        data = data.filter(lambda audio_path, audio, _: tf.shape(audio)[0] <=
                           audio_max_length * sr)
        data = data.filter(
            lambda audio_path, _, label: length_fn(label) <= text_max_length)
        data = data.filter(lambda audio_path, audio, _: tf.shape(audio)[0] >=
                           audio_min_length * sr)
        data = data.filter(
            lambda audio_path, _, label: length_fn(label) >= text_min_length)

        if use_norm:
            data = data.map(lambda audio_path, audio, label:
                            (audio_path, normalize_audio(audio), label),
                            num_parallel_calls=num_parallel_calls)

        if audio_encoder:
            data = data.map(
                lambda audio_path, audio, label:
                (audio_path,
                 audio_encoder(audio, feature_augmentor=feature_augmentor),
                 label),
                num_parallel_calls=num_parallel_calls)
            audio_padded_shapes = [
                None, audio_encoder.window_size, audio_encoder.num_mel_bins
            ]
        else:
            audio_padded_shapes = [None]

        data = data.map(
            lambda audio_path, features, label:
            (audio_path, features, tf.shape(features)[0] - pad_segment, label),
            num_parallel_calls=num_parallel_calls)

        if use_bucket:
            data = data.bucket_by_sequence_length(
                element_length_func=lambda ___, _, length, __: length *
                subsampling_factor // frame_step,
                bucket_boundaries=bucket_boundaries,
                bucket_batch_sizes=bucket_batch_sizes,
                padded_shapes=([], audio_padded_shapes, [], []),
                padding_values=(None, 0.0, 0, None),
                drop_remainder=drop_remainder)
        else:
            data = data.padded_batch(batch_size,
                                     padded_shapes=([], audio_padded_shapes,
                                                    [], []),
                                     padding_values=(None, 0.0, 0, None),
                                     drop_remainder=drop_remainder)

        if teacher_forcing:
            if not use_ctc_target:
                data = data.map(
                    lambda audio_paths, features, lengths, label:
                    (audio_paths,
                     (features, lengths,
                      text_encoder(label, preblank=True, append_eos=append_eos)
                      ), text_encoder(label, append_eos=append_eos)),
                    num_parallel_calls=num_parallel_calls)
            else:
                data = data.map(
                    lambda audio_paths, features, lengths, label:
                    (audio_paths,
                     (features, lengths,
                      text_encoder(label, preblank=True, append_eos=append_eos)
                      ), (text_encoder(label, append_eos=append_eos),
                          text_encoder(label))),
                    num_parallel_calls=num_parallel_calls)
        else:
            if not use_ctc_target:
                data = data.map(lambda audio_paths, features, lengths, label:
                                (audio_paths,
                                 (features, lengths), text_encoder(label)),
                                num_parallel_calls=num_parallel_calls)
            else:
                data = data.map(lambda audio_paths, features, lengths, label:
                                (audio_paths, (features, lengths),
                                 (text_encoder(label), text_encoder(label))),
                                num_parallel_calls=num_parallel_calls)

        if not use_audio_path:
            data = data.map(lambda audio_paths, inputs, targets:
                            (inputs, targets),
                            num_parallel_calls=num_parallel_calls)

        data = data.prefetch(tf.data.AUTOTUNE)
        data.name = name
        return data
