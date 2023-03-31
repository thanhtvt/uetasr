import tensorflow as tf
from typing import List, Union


def get_shape(inputs: tf.Tensor, out_type=tf.int32) -> List[int]:
    static = inputs.shape.as_list()
    dynamic = tf.shape(inputs, out_type=out_type)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def has_devices(devices: Union[List[str], str]):
    if isinstance(devices, list):
        return all([len(tf.config.list_logical_devices(d)) != 0 for d in devices])
    return len(tf.config.list_logical_devices(devices)) != 0


def get_rnn(rnn_type):
    if rnn_type.lower() == 'lstm':
        return tf.keras.layers.LSTM
    elif rnn_type.lower() == 'gru':
        return tf.keras.layers.GRU
    elif rnn_type.lower() == 'rnn':
        return tf.keras.layers.SimpleRNN
    else:
        raise ValueError(f'Invalid rnn_type: {rnn_type}')
