import numpy as np
import tensorflow as tf
from tensorflow_addons.image import sparse_image_warp
from tensorflow.keras.layers.experimental.preprocessing \
    import PreprocessingLayer
from typing import List, Union


class SparseWarp(PreprocessingLayer):

    def __init__(self,
                 prob: float = 0.5,
                 time_warping_para: float = 80,
                 name: str = 'sparse_warp',
                 **kwargs):
        super().__init__(trainable=False, name=name, **kwargs)
        self.prob = prob
        self.time_warping_para = time_warping_para

    def call(
        self,
        feature: Union[tf.Tensor, List[float], np.ndarray],
    ) -> tf.Tensor:
        augmented_feature = self.sparse_warp(feature)
        prob = tf.random.uniform(shape=(),
                                 minval=0,
                                 maxval=1,
                                 dtype=tf.float32)
        return tf.cond(prob <= self.prob, lambda: augmented_feature,
                       lambda: feature)

    def sparse_warp(self, mel_spectrogram: tf.Tensor):
        """ Spec augmentation Calculation Function.
        'SpecAugment' have 3 steps for audio data augmentation.
            1st: Time warping using Tensorflow's image_sparse_warp function.
            2nd: Frequency masking.
            3rd: Time masking.

        Args:
            mel_spectrogram (np.ndarray): mel-spec of audio you want to warping and masking.
            time_warping_para (float, optional): Augmentation parameter, Defaults to 80.

        Returns:
            np.ndarray: warped and masked mel spectrogram.
        """

        fbank_size = tf.shape(mel_spectrogram)
        n, v = fbank_size[0], fbank_size[1]

        # Step 1 : Time warping
        # Image warping control point setting.
        # Source
        pt = tf.random.uniform([], self.time_warping_para,
                               n - self.time_warping_para,
                               tf.int32)  # radnom point along the time axis
        src_ctr_pt_freq = tf.range(v // 2)  # control points on freq-axis
        src_ctr_pt_time = tf.ones_like(
            src_ctr_pt_freq) * pt  # control points on time-axis
        src_ctr_pts = tf.stack((src_ctr_pt_time, src_ctr_pt_freq), -1)
        src_ctr_pts = tf.cast(src_ctr_pts, dtype=tf.float32)

        # Destination
        w = tf.random.uniform([], -self.time_warping_para,
                              self.time_warping_para, tf.int32)  # distance
        dest_ctr_pt_freq = src_ctr_pt_freq
        dest_ctr_pt_time = src_ctr_pt_time + w
        dest_ctr_pts = tf.stack((dest_ctr_pt_time, dest_ctr_pt_freq), -1)
        dest_ctr_pts = tf.cast(dest_ctr_pts, dtype=tf.float32)

        # warp
        source_control_point_locations = tf.expand_dims(src_ctr_pts, 0)
        dest_control_point_locations = tf.expand_dims(dest_ctr_pts, 0)

        warped_image, _ = sparse_image_warp(mel_spectrogram,
                                            source_control_point_locations,
                                            dest_control_point_locations)
        return warped_image
