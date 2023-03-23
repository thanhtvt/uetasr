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
