import tensorflow as tf

from functools import partial
from typing import List

from ...layers.subsampling import Conv2dSubsamplingV2
from ...utils.common import get_shape
from ...layers.normalization import RMSLayerNormalization
from ...layers.feed_forward import PointwiseFeedForward


class EmformerAttention(tf.keras.layers.Layer):

    def __init__(self,
                 input_dim: int = 512,
                 num_heads: int = 8,
                 dropout_rate: float = 0.1,
                 tanh_on_mem: bool = False,
                 use_talking_heads: bool = False,
                 name: str = 'emformer_attention',
                 **kwargs):
        super(EmformerAttention, self).__init__(name=name, **kwargs)
        # assume d_v == d_k == d_q
        self.d_k = input_dim // num_heads
        self.scaling = (input_dim // num_heads)**-0.5
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.tanh_on_mem = tanh_on_mem
        self.dropout_rate = dropout_rate
        self.linear_q = tf.keras.layers.Dense(
            input_dim,
        )
        self.linear_kv = tf.keras.layers.Dense(
            input_dim * 2,
        )
        self.linear_out = tf.keras.layers.Dense(input_dim)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_attention = tf.keras.layers.Dropout(dropout_rate)
        self.use_talking_heads = use_talking_heads
        if use_talking_heads:
            self.pre_softmax_weight = self.add_weight(
                name="pre_softmax_weight",
                shape=(num_heads, num_heads),
                initializer=tf.keras.initializers.GlorotUniform(),
                trainable=True,
            )
            self.post_softmax_weight = self.add_weight(
                name="post_softmax_weight",
                shape=(num_heads, num_heads),
                initializer=tf.keras.initializers.GlorotUniform(),
                trainable=True,
            )
            self.pre_softmax_talking_heads = tf.keras.layers.Conv2D(
                num_heads,
                kernel_size=1,
                data_format="channels_first",
                use_bias=False
            )
            self.post_softmax_talking_heads = tf.keras.layers.Conv2D(
                num_heads,
                kernel_size=1,
                data_format="channels_first",
                use_bias=False
            )

    def summary(self):
        center_context = tf.keras.Input(shape=(None, self.input_dim),
                                        batch_size=None,
                                        dtype=tf.float32)
        right_context = tf.keras.Input(shape=(None, self.input_dim),
                                       batch_size=None,
                                       dtype=tf.float32)
        lengths = tf.keras.Input(shape=(), batch_size=None, dtype=tf.int32)

        memory = tf.keras.Input(shape=(None, self.input_dim),
                                batch_size=None,
                                dtype=tf.float32)
        summary = tf.keras.Input(shape=(None, self.input_dim),
                                 batch_size=None,
                                 dtype=tf.float32)

        attention_mask = tf.keras.Input(shape=(None, None),
                                        batch_size=None,
                                        dtype=tf.bool)

        outputs = self.call(center_context, right_context, lengths, summary,
                            memory, attention_mask)
        model = tf.keras.Model(inputs=(center_context, right_context, lengths,
                                       summary, memory, attention_mask),
                               outputs=outputs,
                               name=self.name)
        model.summary()

    def compute_qkv(self,
                    center_context: tf.Tensor,
                    right_context: tf.Tensor,
                    lengths: tf.Tensor,
                    summary: tf.Tensor,
                    memory: tf.Tensor,
                    left_context_key: tf.Tensor = None,
                    left_context_value: tf.Tensor = None,
                    training: bool = False):

        batch_size = get_shape(center_context)[1]
        total_frame = get_shape(summary)[0] + get_shape(
            center_context)[0] + get_shape(right_context)[0]

        # put right context before for hard copying
        # (R + C + S, B, D)
        query = self.linear_q(
            tf.concat([right_context, center_context, summary], axis=0),
            training=training
        )
        # (R + C + M, B, D x 2)
        kv = self.linear_kv(
            tf.concat([memory, right_context, center_context], axis=0),
            training=training
        )
        org_key, org_value = tf.split(kv, 2, axis=2)

        if left_context_key is not None and left_context_value is not None:
            right_context_blocks_length = total_frame - \
                tf.reduce_max(lengths) - \
                get_shape(summary)[0]
            memory_size = get_shape(memory)[0]
            org_key = tf.concat([
                org_key[:memory_size + right_context_blocks_length],
                left_context_key,
                org_key[memory_size + right_context_blocks_length:]
            ], axis=0)
            org_value = tf.concat([
                org_value[:memory_size + right_context_blocks_length],
                left_context_value,
                org_value[memory_size + right_context_blocks_length:]
            ], axis=0)

        # (B * num_heads, Q/K/V, d_k)
        query = tf.transpose(
            tf.reshape(query, [-1, batch_size * self.num_heads, self.d_k]),
            perm=[1, 0, 2]
        )
        key = tf.transpose(
            tf.reshape(org_key, [-1, batch_size * self.num_heads, self.d_k]),
            perm=[1, 0, 2]
        )
        value = tf.transpose(
            tf.reshape(org_value, [-1, batch_size * self.num_heads, self.d_k]),
            perm=[1, 0, 2]
        )

        return query, key, value, org_key, org_value

    @staticmethod
    def create_padding_mask(center_context: tf.Tensor,
                            right_context: tf.Tensor,
                            lengths: tf.Tensor,
                            summary: tf.Tensor,
                            memory: tf.Tensor,
                            left_context_key: tf.Tensor = None):
        """
        Args:
            center_context (tf.Tensor): Center context (C, B, feat_dim)
            right_context (tf.Tensor): Right context (R, B, feat_dim)
            lengths (tf.Tensor): Lengths of center context (B,)
        Returns:
            padding_mask (tf.Tensor): (B, new_maxlen)
        """
        total_frames = get_shape(right_context)[0] + get_shape(
            center_context)[0] + get_shape(summary)[0]
        batch_size = get_shape(right_context)[1]

        right_context_blocks_length = total_frames - \
            tf.reduce_max(lengths) - \
            get_shape(summary)[0]
        left_context_blocks_length = get_shape(
            left_context_key)[0] if left_context_key is not None else 0
        key_lengths = lengths + get_shape(memory)[
            0] + right_context_blocks_length + left_context_blocks_length
        maxlen = tf.reduce_max(key_lengths)
        padding_mask = tf.tile(
            tf.range(0, maxlen)[tf.newaxis], [batch_size, 1])
        # padding_mask = tf.where(padding_mask >= key_lengths[..., tf.newaxis], True, False)
        padding_mask = tf.where(padding_mask >= key_lengths, True, False)

        return padding_mask

    def compute_attention(self,
                          value: tf.Tensor,
                          query: tf.Tensor,
                          key: tf.Tensor,
                          attention_mask: tf.Tensor,
                          padding_mask: tf.Tensor = None,
                          min_mask: float = -1e8,
                          training: bool = False):
        # (B * num_heads, Q, K)
        attention_weights = tf.einsum('bik,bjk->bij', query * self.scaling, key)

        # Change shape to (B, num_heads, Q, K) to apply talking heads
        total_frames = tf.shape(attention_weights)[1]
        batch_size = tf.shape(attention_weights)[0] // self.num_heads
        attention_weights = tf.reshape(
            attention_weights, [batch_size, self.num_heads, total_frames, -1])

        # Apply linear projection before softmax
        if self.use_talking_heads:
            attention_weights = tf.einsum("bnqk,nm->bmqk", attention_weights,
                                          self.pre_softmax_weight)

        # (B * num_heads, Q, K)
        attention_weights = tf.where(attention_mask,
                                     min_mask,
                                     attention_weights,
                                     name='attw')
        attention_weights = tf.where(padding_mask[:, tf.newaxis, tf.newaxis,
                                                  ...],
                                     min_mask,
                                     attention_weights,
                                     name='attw2')
        attention_probs = tf.nn.softmax(attention_weights, axis=-1)

        # Apply linear projection after softmax
        if self.use_talking_heads:
            attention_probs = tf.einsum("bmqk,mn->bnqk", attention_probs,
                                        self.post_softmax_weight)

        attention_probs = self.dropout_attention(attention_probs,
                                                 training=training)

        attention_probs = tf.reshape(
            attention_probs, [batch_size * self.num_heads, total_frames, -1])
        attention = tf.matmul(attention_probs, value)
        attention = tf.reshape(tf.transpose(attention, perm=[1, 0, 2]),
                               [total_frames, batch_size, self.input_dim])

        return attention

    def _forward(self,
                 center_context: tf.Tensor,
                 right_context: tf.Tensor,
                 lengths: tf.Tensor,
                 summary: tf.Tensor,
                 memory: tf.Tensor,
                 attention_mask: tf.Tensor,
                 left_context_key: tf.Tensor = None,
                 left_context_value: tf.Tensor = None,
                 training: bool = False,
                 **kwargs):
        """
        S = ceil(C / segment_length)
        M = ceil(C / segment_length) - 1
        Q = R + C + S
        K = M + R + C

        Args:
            center_context (tf.Tensor): Center context (C, B, D)
            right_context (tf.Tensor): Right context (R, B, D)
            lengths (tf.Tensor): i-th element representing
                                 number of valid frames for i-th batch elem (B)
            summary (tf.Tensor): Summary vector (S, B, D)
            memory (tf.Tensor): Memory vector (M, B, D)
            attention_mask (tf.Tensor): Attention mask (Q, K)
            left_context_key (tf.Tensor): Cached left context key (L, B, D)
            left_context_value (tf.Tensor): Cached left context value (L, B, D)
        """
        total_frames = get_shape(right_context)[0] + get_shape(
            center_context)[0] + get_shape(summary)[0]
        # (batch_size, tf.reduce_max(_lengths_))
        padding_mask = self.create_padding_mask(center_context,
                                                right_context,
                                                lengths,
                                                summary,
                                                memory,
                                                left_context_key)

        # (batch_size * num_heads, Q/K/V length, d_k)
        query, key, value, org_key, org_value = self.compute_qkv(
            center_context, right_context, lengths, summary, memory,
            left_context_key, left_context_value, training)

        # (total_frames, batch_size, self.input_dim)
        attention = self.compute_attention(value,
                                           query,
                                           key,
                                           attention_mask,
                                           padding_mask,
                                           training=training)

        # (total_frames, batch_size, input_dim)
        output_attention = self.linear_out(attention, training=training)
        summary_length = get_shape(summary)[0]
        output_right_center_context = output_attention[:total_frames -
                                                       summary_length]
        output_memory = output_attention[total_frames - summary_length:]
        if self.tanh_on_mem:
            output_memory = tf.tanh(output_memory)
        else:
            output_memory = tf.clip_by_value(output_memory, -10, 10)

        return output_right_center_context, output_memory, org_key, org_value

    def call(self,
             center_context: tf.Tensor,
             right_context: tf.Tensor,
             lengths: tf.Tensor,
             summary: tf.Tensor,
             memory: tf.Tensor,
             attention_mask: tf.Tensor,
             training: bool = False):

        # right_center_context, mems
        output, output_mems, *_ = self._forward(center_context,
                                                right_context,
                                                lengths,
                                                summary,
                                                memory,
                                                attention_mask,
                                                training=training)

        return output, output_mems[:-1]

    def infer(self,
              center_context: tf.Tensor,
              right_context: tf.Tensor,
              lengths: tf.Tensor,
              summary: tf.Tensor,
              memory: tf.Tensor,
              left_context_key: tf.Tensor,
              left_context_value: tf.Tensor,
              training: bool = False):
        total_right_context_length = get_shape(right_context)[0]
        memory_size = get_shape(memory)[0]
        query_length = total_right_context_length + get_shape(
            center_context)[0] + get_shape(summary)[0]
        key_length = total_right_context_length + get_shape(
            left_context_key)[0] + get_shape(center_context)[0] + memory_size
        attention_mask = tf.zeros([query_length - 1, key_length],
                                  dtype=tf.bool)
        # disallow attention between summary with memory
        summary_memory_attention = tf.concat([
            tf.ones([memory_size, ], dtype=tf.bool),
            tf.zeros([key_length - memory_size, ], dtype=tf.bool)], axis=0)
        attention_mask = tf.concat(
            [attention_mask, summary_memory_attention[tf.newaxis]], axis=0)
        output, output_memory, key, value = self._forward(center_context,
                                                          right_context,
                                                          lengths,
                                                          summary,
                                                          memory,
                                                          attention_mask,
                                                          left_context_key,
                                                          left_context_value,
                                                          training=training)
        return output, output_memory, key[memory_size +
                                          total_right_context_length:], value[
                                              memory_size +
                                              total_right_context_length:]


class EmformerAttentionMask(tf.keras.layers.Layer):

    def __init__(self,
                 segment_length,
                 right_context_length,
                 left_context_length,
                 max_memory_length,
                 name='emformer_attention_mask'):
        super().__init__(name=name)
        self.segment_length = segment_length
        self.right_context_length = right_context_length
        self.left_context_length = left_context_length
        self.max_memory_length = max_memory_length

    @staticmethod
    def create_attention_mask_block(col_widths, col_mask,
                                    num_rows) -> tf.Tensor:
        """
        Args:
            col_widths (tf.Tensor / List[int]) : length = 6/9
            col_mask (List[bool])  : length = 6/9
            num_rows (tf.Tensor)   : (1, 1)
        Returns:
            mask_blocks (tf.Tensor):
        """
        assert len(col_widths) == len(col_mask), \
            'col_widths and col_mask must have the same length'
        col_mask_ls = tf.split(col_mask, len(col_mask))
        _num_rows = tf.squeeze(num_rows, axis=0)
        mask_blocks = tf.TensorArray(dtype=tf.int32,
                                     size=len(col_mask),
                                     dynamic_size=True,
                                     infer_shape=False)

        for idx in range(len(col_mask)):
            col_width = tf.squeeze(tf.gather(col_widths, idx))
            is_ones_col = tf.gather(col_mask_ls, idx)
            if tf.reduce_all(is_ones_col):
                mask = tf.ones([col_width, _num_rows], dtype=tf.int32)
            else:
                mask = tf.zeros([col_width, _num_rows], dtype=tf.int32)
            mask_blocks = mask_blocks.write(idx, mask)
        mask_blocks = mask_blocks.concat()
        mask_blocks = tf.transpose(mask_blocks, [1, 0])

        return mask_blocks

    def create_attention_mask_col_widths(self, seg_idx, utterance_length):
        """
        Args:
            seg_idx (tf.Tensor): (1, 1)
            utterance_length (tf.Tensor): (1,)
        """
        num_segments = tf.cast(
            tf.math.ceil(utterance_length / self.segment_length), tf.int32)
        rc = tf.convert_to_tensor(self.right_context_length,
                                  dtype=tf.int32)[tf.newaxis]
        rc_start = seg_idx * self.right_context_length
        rc_end = rc_start + self.right_context_length
        seg_start = tf.cast(
            tf.maximum(0, seg_idx * self.segment_length - self.left_context_length),
            tf.int32
        )
        seg_end = tf.cast(
            tf.minimum(utterance_length, (seg_idx + 1) * self.segment_length),
            tf.int32
        )
        rc_length = self.right_context_length * num_segments

        if self.max_memory_length > 0:
            mem_start = tf.cast(
                tf.maximum(0, seg_idx - self.max_memory_length), tf.int32)
            mem_length = num_segments - 1
            col_widths = [
                mem_start,  # before memory
                seg_idx - mem_start,  # memory
                mem_length - seg_idx,  # after memory
                rc_start,  # before right context
                rc,  # right context
                rc_length - rc_end,  # after right context
                seg_start,  # before segment
                seg_end - seg_start,  # segment
                utterance_length - seg_end,  # after segment
            ]
        else:
            col_widths = [
                rc_start,  # before right context
                rc,  # right context
                rc_length - rc_end,  # after right context
                seg_start,  # before segment
                seg_end - seg_start,  # segment
                utterance_length - seg_end,  # after segment
            ]

        return col_widths

    def process_fn_w_summary(self, inputs):
        idx, utterance_length, rc_q_cols_mask, s_cols_mask = inputs
        rc = tf.convert_to_tensor(self.right_context_length,
                                  dtype=tf.int32)[tf.newaxis]
        col_widths = self.create_attention_mask_col_widths(
            idx, utterance_length)
        # (rc, key_length)
        rc_mask_block = self.create_attention_mask_block(
            col_widths, rc_q_cols_mask, rc)

        # (min(segment_length, utt_length - idx * segment_length), key_length)
        query_mask_block = self.create_attention_mask_block(
            col_widths, rc_q_cols_mask,
            tf.minimum(self.segment_length,
                       utterance_length - idx * self.segment_length))

        sum_num_rows = tf.ones(shape=(1, ), dtype=tf.int32)
        # (1, key_length)
        summary_mask_block = self.create_attention_mask_block(
            col_widths, s_cols_mask, sum_num_rows)
        return (rc_mask_block, query_mask_block, summary_mask_block)

    def process_fn_wo_summary(self, inputs):
        idx, utterance_length, rc_q_cols_mask = inputs
        rc = tf.convert_to_tensor(self.right_context_length,
                                  dtype=tf.int32)[tf.newaxis]
        col_widths = self.create_attention_mask_col_widths(
            idx, utterance_length)
        # (rc, key_length)
        rc_mask_block = self.create_attention_mask_block(
            col_widths, rc_q_cols_mask, rc)
        # (min(segment_length, utt_length - idx * segment_length), key_length)
        query_mask_block = self.create_attention_mask_block(
            col_widths, rc_q_cols_mask,
            tf.minimum(self.segment_length,
                       utterance_length - idx * self.segment_length))
        return (rc_mask_block, query_mask_block)

    def call(self,
             indices,
             utt_lengths,
             rc_q_cols_mask_tile,
             last_idx,
             last_utt_lengths,
             last_rc_q_cols_mask,
             s_cols_mask_tile=None,
             last_s_cols_mask=None):
        """
        Args:
            indices (tf.Tensor): (num_segments - 1, 1)
            utt_lengths (tf.Tensor): (num_segments, 1)
            rc_q_cols_mask_tile (tf.Tensor): (num_segments - 1, num_cols (6/9))
            last_idx (tf.Tensor): (1, 1)
            last_utt_lengths (tf.Tensor): (1, )
            last_rc_q_cols_mask (tf.Tensor): (num_cols (6/9), 1)
            s_cols_mask_tile (tf.Tensor): (num_segments - 1, num_cols (6/9)) (or None)
            last_s_cols_mask (tf.Tensor): (num_cols (6/9), 1) (or None)
        """
        if s_cols_mask_tile is None and last_s_cols_mask is None:
            results = tf.map_fn(
                self.process_fn_wo_summary,
                (indices, utt_lengths, rc_q_cols_mask_tile),
                fn_output_signature=(tf.int32, tf.int32),
                infer_shape=False,
            )
            last_inputs = (last_idx, last_utt_lengths, last_rc_q_cols_mask)
            last_results = self.process_fn_wo_summary(last_inputs)
        else:
            results = tf.map_fn(
                self.process_fn_w_summary,
                (indices, utt_lengths, rc_q_cols_mask_tile, s_cols_mask_tile),
                fn_output_signature=(tf.int32, tf.int32, tf.int32),
                infer_shape=False,
            )
            last_inputs = (last_idx, last_utt_lengths, last_rc_q_cols_mask,
                           last_s_cols_mask)
            last_results = self.process_fn_w_summary(last_inputs)

        col_widths = self.create_attention_mask_col_widths(
            tf.zeros((1, ), dtype=tf.int32), last_utt_lengths)
        total_widths = tf.reduce_sum(col_widths)
        results = [
            tf.reshape(res, shape=(-1, total_widths)) for res in results
        ]
        attention_masks = [
            tf.concat([r, lr], axis=0) for r, lr in zip(results, last_results)
        ]
        attention_mask = 1 - tf.concat(attention_masks, axis=0)
        attention_mask = attention_mask > 0

        return attention_mask


class EmformerLayer(tf.keras.layers.Layer):

    def __init__(self,
                 input_dim: int,
                 num_heads: int = 8,
                 ffn_dim: int = 1024,
                 segment_length: int = 4,
                 dropout_rate: float = 0.1,
                 left_context_length: int = 0,
                 max_memory_length: int = 1,
                 activation: tf.keras.layers.Layer = tf.keras.layers.ReLU(),
                 tanh_on_mem: bool = False,
                 use_iterated_loss: bool = False,
                 linear_output_dim: int = 500,
                 use_talking_heads: bool = False,
                 use_rmsnorm: bool = False,
                 name: str = 'emformer_layer',
                 **kwargs):
        super().__init__(**kwargs, name=name)

        if use_rmsnorm:
            norm_fn = partial(RMSLayerNormalization, epsilon=1e-5, p=0.0625)
        else:
            norm_fn = partial(tf.keras.layers.LayerNormalization, epsilon=1e-5)

        self.attention = EmformerAttention(
            input_dim=input_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            tanh_on_mem=tanh_on_mem,
            use_talking_heads=use_talking_heads,
            name=f'{name}/emformer_attention',
        )
        self.feed_forward = PointwiseFeedForward(
            input_dim=input_dim,
            dropout_rate=dropout_rate,
            ffn_dim=ffn_dim,
            activation=activation,
            use_rmsnorm=use_rmsnorm,
            name=f'{name}/pointwise_feedforward',
        )
        self.average = tf.keras.layers.AveragePooling1D(
            pool_size=segment_length, strides=segment_length, padding='same')
        self.layernorm = norm_fn()

        # Out layer for iterated loss
        self.use_iterated_loss = use_iterated_loss
        if use_iterated_loss:
            self.linear_out = tf.keras.layers.Dense(linear_output_dim)
            self.layernorm_out = norm_fn()

        self.input_dim = input_dim
        self.segment_length = segment_length
        self.left_context_length = left_context_length
        self.max_memory_length = max_memory_length

    def summary(self):
        center_context = tf.keras.Input(shape=(None, self.input_dim),
                                        batch_size=None,
                                        dtype=tf.float32)
        right_context = tf.keras.Input(shape=(None, self.input_dim),
                                       batch_size=None,
                                       dtype=tf.float32)
        lengths = tf.keras.Input(shape=(), batch_size=None, dtype=tf.int32)

        memory = tf.keras.Input(shape=(None, self.input_dim),
                                batch_size=None,
                                dtype=tf.float32)
        attention_mask = tf.keras.Input(shape=(None, ),
                                        batch_size=None,
                                        dtype=tf.bool)

        outputs = self.call(center_context, right_context, lengths, memory,
                            attention_mask)
        model = tf.keras.Model(inputs=(center_context, right_context, lengths,
                                       memory, attention_mask),
                               outputs=outputs)
        model.summary()

    def _init_states(self, batch_size: int):
        lengths = self.max_memory_length + self.left_context_length + self.left_context_length + 1
        return tf.zeros(shape=(lengths, batch_size, self.input_dim),
                        dtype=tf.float32)

    def _unpack_states(self, states: tf.Tensor):
        num_or_size_splits = tf.constant([
            self.max_memory_length, self.left_context_length,
            self.left_context_length, 1
        ],
                                         dtype=tf.int32)
        states = tf.split(states,
                          num_or_size_splits=num_or_size_splits,
                          axis=0)

        past_length = tf.cast(states[3][0][0][0], dtype=tf.int32)
        past_left_context_length = tf.cast(tf.minimum(self.left_context_length,
                                                      past_length),
                                           dtype=tf.int32)
        # TODO: Explain these
        past_memory_length = tf.cast(
            tf.minimum(
                self.max_memory_length,
                tf.cast(tf.math.ceil(past_length / self.segment_length),
                        tf.int32)), tf.int32)
        pre_memory = states[0][self.max_memory_length - past_memory_length:]
        left_context_key = states[1][self.left_context_length -
                                     past_left_context_length:]
        left_context_value = states[2][self.left_context_length -
                                       past_left_context_length:]

        return pre_memory, left_context_key, left_context_value

    def _pack_states(self, next_key: tf.Tensor, next_value: tf.Tensor,
                     update_length: int, memory: tf.Tensor, states: tf.Tensor):
        num_or_size_splits = tf.constant([
            self.max_memory_length, self.left_context_length,
            self.left_context_length, 1
        ],
                                         dtype=tf.int32)
        states = tf.split(states,
                          num_or_size_splits=num_or_size_splits,
                          axis=0)

        new_key = tf.concat([states[1], next_key], axis=0)
        new_value = tf.concat([states[2], next_value], axis=0)
        states[0] = tf.concat([states[0], memory],
                              axis=0)[-self.max_memory_length:]
        states[1] = new_key[get_shape(new_key)[0] - self.left_context_length:]
        states[2] = new_value[get_shape(new_value)[0] -
                              self.left_context_length:]
        states[3] = tf.cast(tf.cast(states[3], dtype=tf.int32) + update_length,
                            dtype=tf.float32)
        states = tf.concat(states, axis=0)
        return states

    def apply_pre_attention(self,
                            center_context: tf.Tensor,
                            right_context: tf.Tensor,
                            training: bool = False):
        inputs = tf.concat([right_context, center_context], axis=0)
        normed_inputs = self.layernorm(inputs, training=training)
        return normed_inputs[tf.shape(right_context)
                             [0]:], normed_inputs[:tf.shape(right_context)[0]]

    def apply_attention_forward(self,
                                center_context: tf.Tensor,
                                right_context: tf.Tensor,
                                lengths: tf.Tensor,
                                memory: tf.Tensor,
                                attention_mask: tf.Tensor,
                                training: bool = False):
        if self.max_memory_length > 0:
            summary = self.average(tf.transpose(center_context, perm=[1, 0,
                                                                      2]))
            summary = tf.transpose(summary, perm=[1, 0, 2])
        else:
            center_shape = tf.shape(center_context)
            sum_shape = tf.concat(
                [tf.zeros(shape=(1, ), dtype=tf.int32), center_shape[1:]],
                axis=0)
            summary = tf.reshape((), shape=sum_shape)

        right_center_output, next_memory = self.attention(center_context,
                                                          right_context,
                                                          lengths,
                                                          summary,
                                                          memory,
                                                          attention_mask,
                                                          training=training)

        return right_center_output, next_memory

    def apply_attention_infer(self,
                              center_context: tf.Tensor,
                              right_context: tf.Tensor,
                              lengths: tf.Tensor,
                              memory: tf.Tensor,
                              states: List[tf.Tensor] = None):
        if states is None:
            states = self._init_states(get_shape(center_context)[1])
        pre_memory, left_context_key, left_context_value = self._unpack_states(
            states)
        if self.max_memory_length > 0:
            summary = self.average(tf.transpose(center_context, perm=[1, 0,
                                                                      2]))
            summary = tf.transpose(summary, perm=[1, 0, 2])
        else:
            center_shape = tf.shape(center_context)
            sum_shape = tf.concat(
                [tf.zeros(shape=(1, ), dtype=tf.int32), center_shape[1:]],
                axis=0)
            summary = tf.reshape((), shape=sum_shape)
        right_center_output, next_memory, next_key, next_value = self.attention.infer(
            center_context, right_context, lengths, summary, pre_memory,
            left_context_key, left_context_value)
        states = self._pack_states(next_key, next_value,
                                   get_shape(center_context)[0], memory,
                                   states)
        return right_center_output, next_memory, states

    def apply_post_attention(self,
                             right_center_output: tf.Tensor,
                             center_context: tf.Tensor,
                             right_context: tf.Tensor,
                             training: bool = False):
        right_center_output = self.feed_forward(right_center_output,
                                                center_context,
                                                right_context,
                                                training=training)
        center_context_output = right_center_output[tf.shape(right_context
                                                             )[0]:]
        right_context_output = right_center_output[:tf.shape(right_context)[0]]
        if self.use_iterated_loss and training:
            center_context_output_compute_loss = tf.transpose(
                center_context_output, perm=[1, 0, 2])
            center_context_output_compute_loss = self.linear_out(
                center_context_output_compute_loss, training=training)
            center_context_output_compute_loss = self.layernorm_out(
                center_context_output_compute_loss, training=training)
            return center_context_output, right_context_output, \
                center_context_output_compute_loss
        return center_context_output, right_context_output, None

    def call(self,
             center_context: tf.Tensor,
             right_context: tf.Tensor,
             lengths: tf.Tensor,
             memory: tf.Tensor,
             attention_mask: tf.Tensor,
             training: bool = False):
        """
        Args:
            center_context: [C, B, D]
            right_context: [R, B, D]
            lengths: [B,]
            memory: [ceil(C / segment_length), B, D]
            attention_mask: [Q, K]

        Returns:
            center_context_output: (C, B, D)
            right_context_output: (R, B, D)
            memory_output: (ceil(C / segment_length), B, D)
        """

        # (C/R, B, D)
        normed_center_context, normed_right_context = self.apply_pre_attention(
            center_context, right_context, training)

        # right_center_output: [R + C, B, D]
        # memory_output: [ceil(C / segment_length), B, D]
        right_center_output, memory_output = self.apply_attention_forward(
            normed_center_context,
            normed_right_context,
            lengths,
            memory,
            attention_mask,
            training=training)

        # (C/R, B, D)
        center_context_output, right_context_output, center_context_output_compute_loss = \
            self.apply_post_attention(
                right_center_output,
                center_context,
                right_context,
                training=training
            )

        return center_context_output, right_context_output, memory_output, \
            center_context_output_compute_loss

    def infer(self,
              center_context: tf.Tensor,
              right_context: tf.Tensor,
              lengths: tf.Tensor,
              memory: tf.Tensor,
              states: List[tf.Tensor] = None,
              training: bool = False):
        normed_center_context, normed_right_context = self.apply_pre_attention(
            center_context, right_context, training=training)
        right_center_output, memory_output, states_output = self.apply_attention_infer(
            normed_center_context, normed_right_context, lengths, memory,
            states)
        center_context_output, right_context_output, _ = self.apply_post_attention(
            right_center_output,
            center_context,
            right_context,
            training=training)

        return center_context_output, right_context_output, states_output, memory_output


class EmformerEncoder(tf.keras.Model):

    def __init__(self,
                 input_dim: int,
                 num_heads: int = 8,
                 ffn_dim: int = 1024,
                 num_layers: int = 16,
                 segment_length: int = 4,
                 dropout_rate: float = 0.1,
                 activation_type: str = 'relu',
                 input_layer: str = 'linear',
                 left_context_length: int = 0,
                 right_context_length: int = 0,
                 max_memory_length: int = 0,
                 window_size: int = 4,
                 linear_dim: int = 128,
                 conv2d_out_dim: int = 512,
                 tanh_on_mem: bool = False,
                 auxiliary_layer: List = [],
                 linear_output_dim: int = 500,
                 weight_init_strategy: str = None,
                 use_talking_heads: bool = False,
                 use_rmsnorm: bool = False,
                 name: str = 'emformer_encoder',
                 **kwargs):
        super(EmformerEncoder, self).__init__(name=name, **kwargs)

        self.input_dim = input_dim
        self.linear = tf.keras.layers.Dense(linear_dim, use_bias=False)
        self.average = tf.keras.layers.AveragePooling1D(segment_length,
                                                        strides=segment_length,
                                                        padding='same')
        self.window_size = window_size
        self.linear_dim = linear_dim
        self.segment_length = segment_length
        self.left_context_length = left_context_length
        self.right_context_length = right_context_length
        self.max_memory_length = max_memory_length

        self.mask_attention = EmformerAttentionMask(segment_length,
                                                    right_context_length,
                                                    left_context_length,
                                                    max_memory_length)

        if activation_type == 'relu':
            activation = tf.keras.layers.ReLU()
        elif activation_type == 'gelu':
            activation = tf.keras.layers.Activation(tf.nn.gelu)
        else:
            raise ValueError(f'Unsupported activation {activation_type}')

        if input_layer == 'linear':
            self.encoder_dim = linear_dim * window_size
            self.embeded = self.subsampling_linear
        elif input_layer == 'conv2d':
            self.encoder_dim = conv2d_out_dim
            self.embeded = Conv2dSubsamplingV2(self.encoder_dim, dropout_rate)
            self.conv_subsampling_factor = 4
            self.kernel_size = 3
        else:
            raise ValueError(f'Unsupported layer type {input_layer}')

        self.emformer_layers = []
        self.num_layers = num_layers
        for idx in range(num_layers):
            use_iterated_loss = True if idx in auxiliary_layer else False

            layer = EmformerLayer(self.encoder_dim,
                                  num_heads,
                                  ffn_dim,
                                  segment_length,
                                  dropout_rate,
                                  left_context_length,
                                  max_memory_length,
                                  activation=activation,
                                  tanh_on_mem=tanh_on_mem,
                                  use_iterated_loss=use_iterated_loss,
                                  linear_output_dim=linear_output_dim,
                                  use_talking_heads=use_talking_heads,
                                  use_rmsnorm=use_rmsnorm,
                                  name=f'emformer_layer_{idx}')
            self.emformer_layers.append(layer)

    def summary(self):
        inputs = tf.keras.Input(shape=(None, self.window_size, self.input_dim),
                                batch_size=None)
        lengths = tf.keras.Input(shape=(), batch_size=None, dtype=tf.int32)
        outputs = self.call(inputs, lengths)
        model = tf.keras.Model(inputs=(inputs, lengths),
                               outputs=outputs,
                               name=self.name)
        model.summary()

    def create_right_context(self, input: tf.Tensor) -> tf.Tensor:
        """
        Create right context from input
        num_segments = ceil((total_frames - self.right_context_length) // self.segment_length)

        Args:
            input (tf.Tensor): (total_frames, B, D)
        Returns:
            right_context_blocks (tf.Tensor): (num_segments * self.right_context_length, B, D)
        """
        total_frames, batch_size, input_dim = get_shape(input)
        original_frames = total_frames - self.right_context_length

        num_segs = original_frames // self.segment_length
        redundant = original_frames - num_segs * self.segment_length

        redundant = tf.keras.layers.Lambda(
            lambda x: tf.cond(tf.equal(x, tf.zeros((), tf.int32)), lambda: self
                              .segment_length, lambda: x))(redundant)

        # (num_segments, self.right_context_length, B, D)
        right_context = tf.signal.frame(
            input[self.segment_length:(original_frames - redundant)],
            frame_length=self.right_context_length,
            frame_step=self.segment_length,
            pad_end=True,
            axis=0)
        right_context = tf.reshape(right_context,
                                   shape=(-1, batch_size, input_dim))
        right_context = tf.concat([
            right_context,
            input[(original_frames - redundant):(original_frames - redundant +
                                                 self.right_context_length)],
            input[original_frames:]
        ],
                                  axis=0)
        return right_context

    def create_attention_mask(self, input: tf.Tensor) -> tf.Tensor:
        """
        Create attention mask from input
        Args:
            input (tf.Tensor): (total_frames - self.right_context_length, B, D)
        Returns:
            attention_mask (tf.Tensor): (query_length, key_length)
        """
        utterance_length = get_shape(input)[0]
        num_segments = tf.cast(
            tf.math.ceil(utterance_length / self.segment_length), tf.int32)

        if self.max_memory_length > 0:
            num_cols = 9
            # memory, right context, query segment
            rc_q_cols_mask = [idx in [1, 4, 7] for idx in range(num_cols)]
            # right context, query segment
            s_cols_mask = [idx in [4, 7] for idx in range(num_cols)]
        else:
            num_cols = 6
            # right context, query segment
            rc_q_cols_mask = [idx in [1, 4] for idx in range(num_cols)]
            # s_cols_mask = None
            s_cols_mask = []

        # (1, num_cols)
        rc_q_cols_mask_tensor = tf.convert_to_tensor(rc_q_cols_mask,
                                                     dtype=tf.bool)[tf.newaxis]
        rc_q_cols_mask_tile = tf.tile(rc_q_cols_mask_tensor,
                                      [num_segments - 1, 1])
        indices = tf.range(num_segments - 1, dtype=tf.int32)[..., tf.newaxis]
        utt_lengths = tf.ones(
            (num_segments - 1, 1), dtype=tf.int32) * utterance_length
        last_idx = tf.expand_dims(num_segments - 1, axis=0)
        last_utt_lengths = tf.ones((1, ), dtype=tf.int32) * utterance_length
        last_s_cols_mask = tf.convert_to_tensor(s_cols_mask,
                                                dtype=tf.bool)[..., tf.newaxis]
        last_rc_q_cols_mask = tf.convert_to_tensor(rc_q_cols_mask,
                                                   dtype=tf.bool)[...,
                                                                  tf.newaxis]

        if len(s_cols_mask) != 0:
            s_cols_mask_tensor = tf.convert_to_tensor(
                s_cols_mask, dtype=tf.bool)[tf.newaxis]
            s_cols_mask_tile = tf.tile(s_cols_mask_tensor,
                                       [num_segments - 1, 1])
            attention_mask = self.mask_attention(indices, utt_lengths,
                                                 rc_q_cols_mask_tile, last_idx,
                                                 last_utt_lengths,
                                                 last_rc_q_cols_mask,
                                                 s_cols_mask_tile,
                                                 last_s_cols_mask)
            attention_mask = attention_mask[0]
        else:
            attention_mask = self.mask_attention(indices, utt_lengths,
                                                 rc_q_cols_mask_tile, last_idx,
                                                 last_utt_lengths,
                                                 last_rc_q_cols_mask)

        return attention_mask

    def subsampling_linear(self,
                           input: tf.Tensor,
                           training: bool = False) -> tf.Tensor:
        input = self.linear(input, training=training)
        batch_size, time_step = get_shape(input)[:2]
        input = tf.reshape(
            input, (batch_size, time_step, self.window_size * self.linear_dim))
        return input

    def call(self,
             input: tf.Tensor,
             lengths: tf.Tensor,
             training: bool = False):
        """
        Forward pass for training and non-streaming inference
        Args:
            input (tf.Tensor): right-padded input with right context frames
                               (batch_size, time_step + right_context_length, input_dim)
            lengths (tf.Tensor): number of valid utterance frames for i-th batch element
                                 (batch_size,)
        Returns:
            output (tf.Tensor): output of the encoder
                                (batch_size, time_step, input_dim)
            output_lengths (tf.Tensor): number of valid utterance frames for i-th batch element
                                        (batch_size,)
        """
        input = self.embeded(input, training=training)
        if isinstance(self.embeded, Conv2dSubsamplingV2):
            lengths = (
                lengths - self.kernel_size
            ) // self.conv_subsampling_factor - self.right_context_length
        input = tf.transpose(input, perm=[1, 0, 2])
        total_frames = get_shape(input)[0]
        # (R, B, D)
        right_context = self.create_right_context(input)
        # (C, B, D)
        center_context = input[:total_frames - self.right_context_length]
        attention_mask = self.create_attention_mask(center_context)

        if self.max_memory_length > 0:
            memory = self.average(tf.transpose(center_context, perm=[1, 0, 2]))
            memory = tf.transpose(memory, perm=[1, 0, 2])[:-1]
        else:
            center_shape = tf.shape(center_context)
            mem_shape = tf.concat(
                [tf.zeros(shape=(1, ), dtype=tf.int32), center_shape[1:]],
                axis=0)
            memory = tf.reshape((), shape=mem_shape)
        output = center_context

        auxiliary_outputs = []
        for layer in self.emformer_layers:
            output, right_context, memory, auxiliary_output = layer(
                output, right_context, lengths, memory, attention_mask,
                training)
        if auxiliary_output is not None:
            auxiliary_outputs.append(auxiliary_output)

        return tf.transpose(output, perm=[1, 0, 2]), lengths, auxiliary_outputs

    def infer(self,
              input: tf.Tensor,
              lengths: tf.Tensor,
              states: tf.Tensor,
              pe_offset: int = 0,
              training: bool = False):
        """
        Forward pass for streaming inference
        Args:
            input (tf.Tensor): right-padded input with right context frames
                               (batch_size, time_step + right_context_length, input_dim)
            lengths (tf.Tensor): number of valid utterance frames for i-th batch element
                                 (batch_size,)
        Returns:
            output (tf.Tensor): output of the encoder
                                (batch_size, time_step, input_dim)
            output_lengths (tf.Tensor): number of valid utterance frames for i-th batch element
                                        (batch_size,)
            output_states (List): (past_memory, past_left_key, past_left_value, past_length)
        """
        if isinstance(self.embeded, Conv2dSubsamplingV2):
            input = self.embeded(input, pe_offset=pe_offset, training=training)
            lengths = (lengths -
                       self.kernel_size) // self.conv_subsampling_factor
        else:
            input = self.embeded(input, training=training)
        assert get_shape(input)[1] == self.segment_length + self.right_context_length, \
            f"{get_shape(input)[1]} != {self.segment_length} + {self.right_context_length}"
        input = tf.transpose(input, perm=[1, 0, 2])
        right_context_start = get_shape(input)[0] - self.right_context_length
        right_context = input[right_context_start:]
        center_context = input[:right_context_start]
        output_lengths = lengths - self.right_context_length
        output_lengths = tf.where(output_lengths < 0, 0, output_lengths)
        if self.max_memory_length > 0:
            memory = self.average(tf.transpose(center_context, perm=[1, 0, 2]))
            memory = tf.transpose(memory, perm=[1, 0, 2])
        else:
            center_shape = tf.shape(center_context)
            mem_shape = tf.concat(
                [tf.zeros(shape=(1, ), dtype=tf.int32), center_shape[1:]],
                axis=0)
            memory = tf.reshape((), shape=mem_shape)

        output = center_context
        output_states = []
        for layer_idx, layer in enumerate(self.emformer_layers):
            output, right_context, output_state, memory = layer.infer(
                output, right_context, output_lengths, memory,
                None if states is None else states[layer_idx], training)
            output_states.append(output_state)
        output_states = tf.stack(output_states, axis=0)

        return tf.transpose(output, perm=[1, 0,
                                          2]), output_lengths, output_states
