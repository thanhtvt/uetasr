import math
import tensorflow as tf
from typing import Optional

from ..utils.common import get_shape


class VGG2L(tf.keras.layers.Layer):

    def __init__(self,
                 input_dim: int,
                 pos_enc: tf.keras.layers.Layer = None,
                 output_dim: Optional[int] = None,
                 name: str = 'vgg2l',
                 **kwargs):
        """Construct a VGG2L object."""
        super(VGG2L, self).__init__(name=name, **kwargs)

        initializer1 = tf.keras.initializers.RandomUniform(
            minval=-math.sqrt(1.0),
            maxval=math.sqrt(1.0),
            seed=0
        )
        initializer2 = tf.keras.initializers.RandomUniform(
            minval=-math.sqrt(1.0 / 64),
            maxval=math.sqrt(1.0 / 64),
            seed=0
        )
        initializer3 = tf.keras.initializers.RandomUniform(
            minval=-math.sqrt(1.0 / 128),
            maxval=math.sqrt(1.0 / 128),
            seed=0)
        initializer4 = tf.keras.initializers.RandomUniform(
            minval=-math.sqrt(1.0 / 128),
            maxval=math.sqrt(1.0 / 128),
            seed=0
        )
        self.vgg2l = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64,
                                   3,
                                   strides=1,
                                   padding="same",
                                   data_format='channels_last',
                                   kernel_initializer=initializer1,
                                   bias_initializer=initializer1),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64,
                                   3,
                                   strides=1,
                                   padding="same",
                                   data_format='channels_last',
                                   kernel_initializer=initializer2,
                                   bias_initializer=initializer2),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((3, 2), data_format='channels_last'),
            tf.keras.layers.Conv2D(128,
                                   3,
                                   strides=1,
                                   padding="same",
                                   data_format='channels_last',
                                   kernel_initializer=initializer2,
                                   bias_initializer=initializer2),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(128,
                                   3,
                                   strides=1,
                                   padding="same",
                                   data_format='channels_last',
                                   kernel_initializer=initializer3,
                                   bias_initializer=initializer3),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2, 2), data_format='channels_last'),
        ])
        self.input_dim = input_dim

        self.out = None
        if output_dim is not None:
            self.out = tf.keras.layers.Dense(output_dim,
                                             kernel_initializer=initializer4,
                                             bias_initializer=initializer4)
        self.pos_enc = pos_enc
        self.subsampling_factor = 4

    def call(self, feats, feats_mask=None, training=False):
        """Forward VGG2L bottleneck.
        Args:
            feats: Feature sequences. (B, T, C, D_feats)
            feats_mask: Mask of feature sequences. (B, 1, T)
        Returns:
            vgg_output: VGG output sequences.
                   (B, sub(T), D_out) or ((B, sub(T), D_out), (B, sub(T), D_att))
            vgg_mask: Mask of VGG output sequences. (B, 1, sub(T))
        """

        feats = tf.transpose(feats, perm=[0, 1, 3, 2])
        vgg_output = self.vgg2l(feats, training=training)

        b, t, f, c = get_shape(vgg_output)
        vgg_output = tf.reshape(tf.transpose(vgg_output, perm=[0, 1, 3, 2]),
                                shape=(b, t, c * f))

        if self.out is not None:
            vgg_output = self.out(vgg_output, training=training)

        if self.pos_enc is not None:
            vgg_output = self.pos_enc(vgg_output, training=training)

        return vgg_output, self.create_new_conformer_mask(feats_mask)

    def create_new_conformer_mask(self, feats_mask: tf.Tensor) -> tf.Tensor:
        """Create a subsampled mask of feature sequences.
        Args:
            feats_mask: Mask of feature sequences. (B, 1, F)
        Returns:
            vgg_mask: Mask of VGG2L output sequences. (B, 1, sub(F))
        """
        b, t, f = get_shape(feats_mask)
        vgg1_t_len = f - (f % 3)
        vgg_mask = feats_mask[:, :, :vgg1_t_len][:, :, ::3]

        b, t, f = get_shape(vgg_mask)
        vgg2_t_len = f - (f % 2)
        vgg_mask = vgg_mask[:, :, :vgg2_t_len][:, :, ::2]

        return vgg_mask

    def get_config(self):
        conf = super(VGG2L, self).get_config()
        conf.update({
            "input_dim": self.input_dim,
            "subsampling_factor": self.subsampling_factor,
        })
        conf.update(self.vgg2l.get_config())
        conf.update(self.out.get_config()) if self.out else None
        conf.update(self.pos_enc.get_config()) if self.pos_enc else None
        return conf


class VGG2LV2(tf.keras.layers.Layer):

    def __init__(self,
                 dim_input: int,
                 pos_enc: tf.keras.layers.Layer = None,
                 output_dim: Optional[int] = None,
                 name: str = 'vgg2l_v2',
                 **kwargs):
        """Construct a VGG2L object."""
        super(VGG2LV2, self).__init__(name=name, **kwargs)

        initializer1 = tf.keras.initializers.RandomUniform(
            minval=-math.sqrt(1.0), maxval=math.sqrt(1.0), seed=0)
        initializer2 = tf.keras.initializers.RandomUniform(
            minval=-math.sqrt(1.0 / 64), maxval=math.sqrt(1.0 / 64), seed=0)
        initializer3 = tf.keras.initializers.RandomUniform(
            minval=-math.sqrt(1.0 / 128),
            maxval=math.sqrt(1.0 / 128),
            seed=0)
        initializer4 = tf.keras.initializers.RandomUniform(
            minval=-math.sqrt(1.0 / 128),
            maxval=math.sqrt(1.0 / 128),
            seed=0)
        self.vgg2l = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64,
                                   3,
                                   strides=1,
                                   padding="same",
                                   data_format='channels_last',
                                   kernel_initializer=initializer1,
                                   bias_initializer=initializer1),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64,
                                   3,
                                   strides=1,
                                   padding="same",
                                   data_format='channels_last',
                                   kernel_initializer=initializer2,
                                   bias_initializer=initializer2),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2, 2), data_format='channels_last'),
            tf.keras.layers.Conv2D(128,
                                   3,
                                   strides=1,
                                   padding="same",
                                   data_format='channels_last',
                                   kernel_initializer=initializer2,
                                   bias_initializer=initializer2),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(128,
                                   3,
                                   strides=1,
                                   padding="same",
                                   data_format='channels_last',
                                   kernel_initializer=initializer3,
                                   bias_initializer=initializer3),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D((2, 2), data_format='channels_last'),
        ])
        self.dim_input = dim_input

        self.out = None
        if output_dim is not None:
            self.out = tf.keras.layers.Dense(output_dim,
                                             kernel_initializer=initializer4,
                                             bias_initializer=initializer4)
        self.pos_enc = pos_enc

        self.subsampling_factor = 4
        self.create_new_mask = self.create_new_conformer_mask

    def call(self, feats, feats_mask=None, training=False):
        """Forward VGG2L bottleneck.
        Args:
            feats: Feature sequences. (B, T, C, D_feats)
            feats_mask: Mask of feature sequences. (B, 1, T)
        Returns:
            vgg_output: VGG output sequences.
                   (B, sub(T), D_out) or ((B, sub(T), D_out), (B, sub(T), D_att))
            vgg_mask: Mask of VGG output sequences. (B, 1, sub(T))
        """

        feats = tf.transpose(feats, perm=[0, 1, 3, 2])
        vgg_output = self.vgg2l(feats, training=training)

        b, t, f, c = get_shape(vgg_output)
        vgg_output = tf.reshape(tf.transpose(vgg_output, perm=[0, 1, 3, 2]),
                                shape=(b, t, c * f))

        if self.out is not None:
            vgg_output = self.out(vgg_output, training=training)

        if self.pos_enc is not None:
            vgg_output = self.pos_enc(vgg_output, training=training)

        return vgg_output, self.create_new_mask(feats_mask)

    def create_new_conformer_mask(self, feats_mask: tf.Tensor) -> tf.Tensor:
        """Create a subsampled mask of feature sequences.
        Args:
            feats_mask: Mask of feature sequences. (B, 1, F)
        Returns:
            vgg_mask: Mask of VGG2L output sequences. (B, 1, sub(F))
        """
        b, t, f = get_shape(feats_mask)
        vgg1_t_len = f - (f % 2)
        vgg_mask = feats_mask[:, :, :vgg1_t_len][:, :, ::2]

        b, t, f = get_shape(vgg_mask)
        vgg2_t_len = f - (f % 2)
        vgg_mask = vgg_mask[:, :, :vgg2_t_len][:, :, ::2]

        return vgg_mask

    def get_config(self):
        conf = super(VGG2LV2, self).get_config()
        conf.update({
            "dim_input": self.dim_input,
            "subsampling_factor": self.subsampling_factor,
        })
        conf.update(self.vgg2l.get_config())
        conf.update(self.out.get_config()) if self.out else None
        conf.update(self.pos_enc.get_config()) if self.pos_enc else None
        return conf
