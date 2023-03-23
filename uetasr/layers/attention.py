import math
import tensorflow as tf

from ..utils.common import get_shape


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-Head Attention layer.
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, n_head: int, n_feat: int, dropout_rate: float):
        """Construct an MultiHeadAttention object."""
        super(MultiHeadAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head

        initializer1 = tf.keras.initializers.RandomUniform(
            minval=-math.sqrt(1.0 / n_feat),
            maxval=math.sqrt(1.0 / n_feat),
            seed=None
        )
        initializer2 = tf.keras.initializers.RandomUniform(
            minval=-math.sqrt(1.0 / n_feat),
            maxval=math.sqrt(1.0 / n_feat),
            seed=None
        )
        initializer3 = tf.keras.initializers.RandomUniform(
            minval=-math.sqrt(1.0 / n_feat),
            maxval=math.sqrt(1.0 / n_feat),
            seed=None
        )
        initializer4 = tf.keras.initializers.RandomUniform(
            minval=-math.sqrt(1.0 / n_feat),
            maxval=math.sqrt(1.0 / n_feat),
            seed=None
        )

        self.linear_q = tf.keras.layers.Dense(n_feat,
                                              kernel_initializer=initializer1,
                                              bias_initializer=initializer1)
        self.linear_k = tf.keras.layers.Dense(n_feat,
                                              kernel_initializer=initializer2,
                                              bias_initializer=initializer2)
        self.linear_v = tf.keras.layers.Dense(n_feat,
                                              kernel_initializer=initializer3,
                                              bias_initializer=initializer3)
        self.linear_out = tf.keras.layers.Dense(
            n_feat,
            use_bias=False,
            kernel_initializer=initializer4,
            bias_initializer=initializer4
        )
        self.attn = None
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def compute_qkv(self, query, key, value, training=False):
        """Transform query, key and value.
        Args:
            query (tf.Tensor): Query tensor (#batch, time1, size).
            key (tf.Tensor): Key tensor (#batch, time2, size).
            value (tf.Tensor): Value tensor (#batch, time2, size).
        Returns:
            tf.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            tf.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            tf.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).
        """
        n_batch = get_shape(query)[0]
        q = tf.reshape(self.linear_q(query, training=training),
                       [n_batch, -1, self.h, self.d_k])
        k = tf.reshape(self.linear_k(key, training=training),
                       [n_batch, -1, self.h, self.d_k])
        v = tf.reshape(self.linear_v(value, training=training),
                       [n_batch, -1, self.h, self.d_k])
        q = tf.transpose(q, perm=[0, 2, 1, 3])  # (batch, head, time1, d_k)
        k = tf.transpose(k, perm=[0, 2, 1, 3])  # (batch, head, time2, d_k)
        v = tf.transpose(v, perm=[0, 2, 1, 3])  # (batch, head, time2, d_k)

        return q, k, v

    def compute_attention(self,
                          value,
                          scores,
                          mask,
                          training=False):
        """Compute attention context vector.
        Args:
            value (tf.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (tf.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (tf.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2)
        Returns:
            tf.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).
        """
        n_batch = get_shape(value)[0]
        if mask is not None:
            mask = tf.expand_dims(mask, axis=1)  # (batch, 1, *, time2)
            min_value = scores.dtype.min
            scores = tf.where(mask, scores, min_value)
            self.attn = tf.nn.softmax(scores, axis=-1)
            self.attn = tf.where(mask, self.attn, 0.0)
        else:
            # (batch, head, time1, time2)
            self.attn = tf.nn.softmax(scores, axis=-1)

        p_attn = self.dropout(self.attn, training=training)
        x = tf.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = tf.reshape(tf.transpose(x, perm=[0, 2, 1, 3]),
                       shape=[n_batch, -1, self.h * self.d_k])

        return self.linear_out(x, training=training)  # (batch, time1, d_model)

    def call(self, query, key, value, mask, training=False):
        """Compute scaled dot product attention.
        Args:
            query (tf.Tensor): Query tensor (#batch, time1, size).
            key (tf.Tensor): Key tensor (#batch, time2, size).
            value (tf.Tensor): Value tensor (#batch, time2, size).
            mask (tf.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
        Returns:
            tf.Tensor: Output tensor (#batch, time1, d_model).
        """
        q, k, v = self.compute_qkv(query, key, value, training=training)
        scores = tf.matmul(q, tf.transpose(
            k, perm=[0, 1, 3, 2])) / tf.math.sqrt(self.d_k * 1.0)
        return self.compute_attention(v, scores, mask, training=training)


class RelPositionMultiHeadAttention(MultiHeadAttention):
    """Multi-Head Attention layer with relative position encoding.
    Details can be found in https://github.com/espnet/espnet/pull/2816.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.
    """

    def __init__(self, n_head, n_feat, dropout_rate, zero_triu=False):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        self.zero_triu = zero_triu
        # linear transformation for positional encoding
        initializer1 = tf.keras.initializers.RandomUniform(
            minval=-math.sqrt(1.0 / n_feat),
            maxval=math.sqrt(1.0 / n_feat),
            seed=None
        )
        initializer2 = tf.keras.initializers.RandomUniform(
            minval=-math.sqrt(1.0 / n_feat),
            maxval=math.sqrt(1.0 / n_feat),
            seed=None
        )
        self.linear_out = tf.keras.layers.Dense(
            n_feat,
            use_bias=True,
            kernel_initializer=initializer1,
            bias_initializer=initializer1
        )
        self.linear_pos = tf.keras.layers.Dense(
            n_feat,
            use_bias=False,
            kernel_initializer=initializer2,
            bias_initializer=initializer2
        )
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        initializer = tf.keras.initializers.RandomUniform(
            minval=-math.sqrt(1.0 / self.h),
            maxval=math.sqrt(1.0 / self.h),
            seed=None
        )
        self.pos_bias_u = tf.Variable(
            initial_value=initializer([self.h, self.d_k], dtype=tf.float32),
            trainable=True
        )
        self.pos_bias_v = tf.Variable(
            initial_value=initializer([self.h, self.d_k], dtype=tf.float32),
            trainable=True
        )

    def rel_shift(self, x, left_context=0):
        """Compute relative positional encoding.
        Args:
            x (tf.Tensor): Input tensor (batch, head, time1, 2*time1-1).
            time1 means the length of query vector.
        Returns:
            tf.Tensor: Output tensor.
        """
        batch, head, time1, time11 = get_shape(x)
        # time2 = time1 + left_context
        zero_pad = tf.zeros((batch, head, time1, 1), dtype=x.dtype)
        x_padded = tf.concat([zero_pad, x], axis=-1)

        x_padded = tf.reshape(x_padded, shape=(batch, head, time11 + 1, time1))
        x = tf.reshape(
            x_padded[:, :, 1:],
            shape=tf.shape(x)
        )[:, :, :, :time11 // 2 + 1]  # only keep the positions from 0 to time2

        return x

    def call(self,
             query: tf.Tensor,
             key: tf.Tensor,
             value: tf.Tensor,
             pos_emb: tf.Tensor,
             mask: tf.Tensor,
             left_context: int = 0,
             training: bool = False):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding

        Args:
            query (tf.Tensor): Query tensor (#batch, time1, size).
            key (tf.Tensor): Key tensor (#batch, time2, size).
            value (tf.Tensor): Value tensor (#batch, time2, size).
            pos_emb (tf.Tensor): Positional embedding tensor
                (#batch, 2*time1-1, size).
            mask (tf.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            tf.Tensor: Output tensor (#batch, time1, d_model).
        """
        q, k, v = self.compute_qkv(query, key, value, training=training)
        q = tf.transpose(q, perm=[0, 2, 1, 3])  # (batch, time1, head, d_k)

        n_batch_pos = tf.shape(pos_emb)[0]
        p = tf.reshape(self.linear_pos(pos_emb, training=training),
                       shape=(n_batch_pos, -1, self.h, self.d_k))
        p = tf.transpose(p, perm=[0, 2, 1, 3])  # (batch, head, 2*time1-1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = tf.transpose(q + self.pos_bias_u, perm=[0, 2, 1, 3])
        # (batch, head, time1, d_k)
        q_with_bias_v = tf.transpose(q + self.pos_bias_v, perm=[0, 2, 1, 3])

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = tf.matmul(q_with_bias_u, tf.transpose(k, perm=[0, 1, 3, 2]))

        # compute matrix b and matrix d
        # (batch, head, time1, 2*time1-1)
        matrix_bd = tf.matmul(q_with_bias_v, tf.transpose(p, perm=[0, 1, 3, 2]))
        # print(matrix_bd.shape)
        matrix_bd = self.rel_shift(matrix_bd, left_context=left_context)
        # print(matrix_ac.shape)
        # print(matrix_bd.shape)
        # print('-' * 10)
        scores = (matrix_ac + matrix_bd) / tf.math.sqrt(
            self.d_k * 1.0)  # (batch, head, time1, time2)

        return self.compute_attention(v,
                                      scores,
                                      mask,
                                      training=training)
