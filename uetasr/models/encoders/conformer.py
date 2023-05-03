import tensorflow as tf

from typing import Tuple, Union

from ...layers.vgg import VGG2L, VGG2LV2
# from ...layers.subsampling import Conv2DSubsampling
from ...layers.positional_encoding import (
    PositionalEncoding,
    ScaledPositionalEncoding,
    RelPositionalEncoding,
)
from ...layers.attention import (
    MultiHeadAttention,
    RelPositionMultiHeadAttention
)
from ...layers.feed_forward import PositionwiseFeedForward
from ...layers.convolution import ConvolutionModule


class ConformerLayer(tf.keras.Model):
    """Conformer module definition.
    Args:
        size: Input/output dimension.
        self_att: Self-attention module instance.
        feed_forward: Feed-forward module instance.
        feed_forward_macaron: Feed-forward module instance for macaron network.
        conv_mod: Convolution module instance.
        dropout_rate: Dropout rate.
        eps_layer_norm: Epsilon value for LayerNorm.
    """

    def __init__(
        self,
        size: int,
        self_att: tf.keras.layers.Layer,
        feed_forward: tf.keras.layers.Layer,
        feed_forward_macaron: tf.keras.layers.Layer,
        conv_mod: tf.keras.layers.Layer,
        dropout_rate: float = 0.0,
        eps_layer_norm: float = 1e-12,
        **kwargs
    ):
        """Construct a Conformer object."""
        super(ConformerLayer, self).__init__(**kwargs)

        self.self_att = self_att
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_mod = conv_mod

        self.norm_feed_forward = tf.keras.layers.LayerNormalization(
            epsilon=eps_layer_norm
        )
        self.norm_multihead_att = tf.keras.layers.LayerNormalization(
            epsilon=eps_layer_norm
        )

        if feed_forward_macaron is not None:
            self.norm_macaron = tf.keras.layers.LayerNormalization(
                epsilon=eps_layer_norm
            )
            self.feed_forward_scale = 0.5
        else:
            self.feed_forward_scale = 1.0

        if self.conv_mod is not None:
            self.norm_conv = tf.keras.layers.LayerNormalization(
                epsilon=eps_layer_norm
            )
            self.norm_final = tf.keras.layers.LayerNormalization(
                epsilon=eps_layer_norm
            )

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.size = size
        self.cache = None

    def summary(self):
        inputs = tf.keras.Input(shape=(None, self.size), batch_size=None)
        masks = tf.keras.Input(shape=(None,), batch_size=None, dtype=tf.bool)
        pos_emb = tf.keras.Input(shape=(None, self.size), batch_size=None)
        outputs = self.call((inputs, pos_emb), masks)
        model = tf.keras.Model(
            inputs=((inputs, pos_emb), masks),
            outputs=outputs,
            name=self.name
        )
        model.summary()

    def call(
        self,
        sequence: Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]],
        mask: tf.Tensor,
        training=False,
        cache: tf.Tensor = None,
    ) -> Tuple[Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]], tf.Tensor]:
        """Encode input sequences.
        Args:
            sequence: Conformer input sequences.
                     (B, T, D_emb) or ((B, T, D_emb), (1, T, D_emb))
            mask: Mask of input sequences. (B, T)
            cache: Conformer cache. (B, T-1, D_hidden)
        Returns:
            sequence: Conformer output sequences.
               (B, T, D_enc) or ((B, T, D_enc), (1, T, D_enc))
            mask: Mask of output sequences. (B, T)
        """
        if isinstance(sequence, tuple):
            sequence, pos_emb = sequence[0], sequence[1]
        else:
            sequence, pos_emb = sequence, None

        if self.feed_forward_macaron is not None:
            residual = sequence

            sequence = self.norm_macaron(sequence, training=training)
            sequence = residual + self.feed_forward_scale * self.dropout(
                self.feed_forward_macaron(sequence, training=training),
                training=training
            )

        residual = sequence
        sequence = self.norm_multihead_att(sequence, training=training)
        x_q = sequence

        if pos_emb is not None:
            sequence_att = self.self_att(x_q,
                                         sequence,
                                         sequence,
                                         pos_emb,
                                         mask,
                                         training=training)
        else:
            sequence_att = self.self_att(x_q,
                                         sequence,
                                         sequence,
                                         mask,
                                         training=training)

        sequence = residual + self.dropout(sequence_att, training=training)

        if self.conv_mod is not None:
            residual = sequence

            sequence = self.norm_conv(sequence, training=training)
            sequence = residual + self.dropout(
                self.conv_mod(sequence, training=training), training=training)

        residual = sequence

        sequence = self.norm_feed_forward(sequence, training=training)
        sequence = residual + self.feed_forward_scale * self.dropout(
            self.feed_forward(sequence, training=training), training=training)

        if self.conv_mod is not None:
            sequence = self.norm_final(sequence, training=training)

        if pos_emb is not None:
            return (sequence, pos_emb), mask

        return sequence, mask

    def infer(
        self,
        sequence: Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]],
        mask: tf.Tensor,
        cache: tf.Tensor = None,
        left_context: int = 0,
        right_context: int = 0,
    ) -> Tuple[Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]], tf.Tensor]:
        """Encode input sequences.
        Args:
            sequence: Conformer input sequences.
                     (B, T, D_emb) or ((B, T, D_emb), (1, T, D_emb))
            mask: Mask of input sequences. (B, T)
            cache: Conformer cache. (B, T-1, D_hidden)
        Returns:
            sequence: Conformer output sequences.
               (B, T, D_enc) or ((B, T, D_enc), (1, T, D_enc))
            mask: Mask of output sequences. (B, T)
        """
        if isinstance(sequence, tuple):
            sequence, pos_emb = sequence[0], sequence[1]
        else:
            sequence, pos_emb = sequence, None

        if self.feed_forward_macaron is not None:
            residual = sequence

            sequence = self.norm_macaron(sequence, training=False)
            sequence = residual + self.feed_forward_scale * self.dropout(
                self.feed_forward_macaron(sequence, training=False),
                training=False)

        residual = sequence
        sequence = self.norm_multihead_att(sequence, training=False)
        # [B, 2 * (C + R) - 1, H]
        pos_emb
        # [B, C + R, H]
        x_q = sequence
        # [B, L + C + R, H]
        key = tf.concat([self.cache[0], sequence], axis=1)
        val = key

        if right_context > 0:
            att_cache = key[:,
                            -(left_context + right_context):-right_context, :]
        else:
            att_cache = key[:, -left_context:, :]

        if pos_emb is not None:
            sequence_att = self.self_att(x_q,
                                         key,
                                         val,
                                         pos_emb,
                                         mask,
                                         training=False)
        else:
            sequence_att = self.self_att(x_q,
                                         key,
                                         val,
                                         mask,
                                         training=False)

        sequence = residual + self.dropout(sequence_att, training=False)

        if self.conv_mod is not None:
            residual = sequence

            sequence = self.norm_conv(sequence, training=False)
            sequence = residual + self.dropout(
                self.conv_mod(sequence, training=False), training=False)

        residual = sequence

        sequence = self.norm_feed_forward(sequence, training=False)
        sequence = residual + self.feed_forward_scale * self.dropout(
            self.feed_forward(sequence, training=False), training=False)

        if self.conv_mod is not None:
            sequence = self.norm_final(sequence, training=False)

        # self.cache = [att_cache, conv_cache]
        self.cache = att_cache
        if pos_emb is not None:
            return (sequence, pos_emb), mask

        return sequence, mask

    def get_config(self):
        conf = super(ConformerLayer, self).get_config()
        conf.update({"size": self.size, "feed_forward_scale": self.feed_forward_scale})
        conf.update(self.self_att.get_config())
        conf.update(self.feed_forward.get_config())
        conf.update(self.feed_forward_macaron.get_config())
        conf.update(self.conv_mod.get_config())
        conf.update(self.norm_multihead_att.get_config())
        conf.update(self.norm_feed_forward.get_config())
        conf.update(self.norm_macaron.get_config()) if self.feed_forward_macaron else None
        conf.update(self.norm_conv.get_config()) if self.conv_mod else None
        conf.update(self.norm_final.get_config()) if self.conv_mod else None
        conf.update(self.dropout.get_config())
        return conf


class Conformer(tf.keras.Model):

    def __init__(self,
                 num_features: int = 80,
                 window_size: int = 1,
                 d_model: int = 256,
                 input_layer="vgg2l",
                 pos_enc_layer_type="rel_pos",
                 dropout_rate_pos_enc=0.1,
                 selfattention_layer_type: str = 'rel_selfattn',
                 attention_heads: int = 4,
                 dropout_rate_att: float = 0.1,
                 dropout_rate_pos_wise: float = 0.1,
                 dropout_rate: float = 0.1,
                 positionwise_layer_type: str = 'linear',
                 linear_units: int = 1024,
                 conv_mod_kernel: int = 31,
                 num_blocks: int = 18,
                 use_macaron: bool = True,
                 use_cnn_module: bool = True,
                 eps_layer_norm: float = 1e-12,
                 name: str = 'conformer_encoder'):
        """Construct an Conformer object."""
        super(Conformer, self).__init__(name=name)

        self.num_features = num_features
        self.window_size = window_size
        self.d_model = d_model

        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "scaled_abs_pos":
            pos_enc_class = ScaledPositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            assert selfattention_layer_type == "rel_selfattn"
            pos_enc_class = RelPositionalEncoding
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        if input_layer == "vgg2l":
            self.embed = VGG2L(
                d_model,
                pos_enc=pos_enc_class(d_model, dropout_rate_pos_enc),
                output_dim=d_model
            )
        elif input_layer == "vgg2lv2":
            self.embed = VGG2LV2(
                d_model,
                pos_enc=pos_enc_class(d_model, dropout_rate_pos_enc),
                output_dim=d_model
            )
        elif input_layer is None:
            self.embed = pos_enc_class(d_model, dropout_rate_pos_enc)
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        # self-attention module definition
        if selfattention_layer_type == "selfattn":
            encoder_selfattn_layer = MultiHeadAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                d_model,
                dropout_rate_att,
            )
        elif selfattention_layer_type == "rel_selfattn":
            assert pos_enc_layer_type == "rel_pos"
            encoder_selfattn_layer = RelPositionMultiHeadAttention
            encoder_selfattn_layer_args = (attention_heads, d_model,
                                           dropout_rate_att)
        else:
            raise ValueError("unknown encoder_attn_layer: " +
                             selfattention_layer_type)

        # feed-forward module definition
        if positionwise_layer_type == "linear":
            pos_wise_act = tf.keras.layers.Activation(tf.nn.swish)
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                d_model,
                linear_units,
                dropout_rate_pos_wise,
                pos_wise_act,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")

        # convolution module definition
        if use_cnn_module:
            convolution_layer = ConvolutionModule
            conv_mod_act = tf.keras.layers.Activation(tf.nn.swish)
            convolution_layer_args = (d_model, conv_mod_kernel, conv_mod_act)

        if use_macaron:
            macaron_net = PositionwiseFeedForward
            macaron_act = tf.keras.layers.Activation(tf.nn.swish)
            macaron_args = (
                d_model,
                linear_units,
                dropout_rate_pos_wise,
                macaron_act,
            )

        self.encoders = [
            ConformerLayer(
                d_model, encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                macaron_net(*macaron_args) if use_macaron else None,
                convolution_layer(
                    *convolution_layer_args) if use_cnn_module else None,
                dropout_rate, eps_layer_norm) for _ in range(num_blocks)
        ]

        self.after_norm = tf.keras.layers.LayerNormalization(
            epsilon=eps_layer_norm)

    def summary(self):
        self.encoders[0].summary()
        inputs = tf.keras.Input(shape=(None, self.window_size,
                                       self.num_features),
                                batch_size=None)
        masks = tf.keras.Input(shape=(1, None), batch_size=None, dtype=tf.bool)
        outputs = self.call(inputs, masks)
        model = tf.keras.Model(inputs=(inputs, masks),
                               outputs=outputs,
                               name=self.name)
        model.summary()

    def call(self, xs, masks, training=False):
        """Encode input sequence.
        Args:
            xs (tf.Tensor): Input tensor (#batch, time, window_size, idim).
            masks (tf.Tensor): Mask tensor (#batch, 1, time).
        Returns:
            tf.Tensor: Output tensor (#batch, time, attention_dim).
            tf.Tensor: Mask tensor (#batch, time).
        """

        sequence, masks = self.embed(xs, masks, training=training)
        for encoder in self.encoders:
            sequence, masks = encoder(sequence, masks, training=training)

        if isinstance(sequence, tuple):
            sequence = self.after_norm(sequence[0], training=training)
        else:
            sequence = self.after_norm(sequence, training=training)

        return sequence, masks

    def get_config(self):
        conf = super(Conformer, self).get_config()
        conf.update({
            "num_features": self.num_features,
            "window_size": self.window_size,
            "d_model": self.d_model,
        })
        conf.update(self.embed.get_config())
        conf.update(self.after_norm.get_config())
        for encoder in self.encoders:
            conf.update(encoder.get_config())
        return conf
