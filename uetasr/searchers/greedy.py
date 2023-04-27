import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import \
    PreprocessingLayer
from typing import List

from .hypothesis import Hypothesis
from .base import BaseSearch


class GreedySearch(BaseSearch):
    """ Greedy search implementation for Transducer models """

    def __init__(
        self,
        decoder: tf.keras.Model,
        joint_network: tf.keras.Model,
        text_decoder: PreprocessingLayer,
        beam_size: int = 10,
        lm: tf.keras.Model = None,
        lm_weight: float = 0.0,
        score_norm: bool = False,
        nbest: int = 1,
        softmax_temperature: float = 1.0,
        name: str = "greedy_search",
        **kwargs,
    ):
        super(GreedySearch, self).__init__(
            decoder=decoder,
            joint_network=joint_network,
            text_decoder=text_decoder,
            beam_size=beam_size,
            lm=lm,
            lm_weight=lm_weight,
            score_norm=score_norm,
            nbest=nbest,
            softmax_temperature=softmax_temperature,
            name=name,
            **kwargs,
        )

    def search(self, enc_out: tf.Tensor,
               enc_len: tf.Tensor) -> List[Hypothesis]:
        """Greedy search implementation.

        Modified from https://arxiv.org/pdf/1211.3711.pdf

        Args:
            enc_out: Encoder output sequence. (T, D)

        Returns:
            nbest_hyps: N-best hypothesis.
        """
        hyp = Hypothesis(
            score=0.0,
            yseq=[self.blank_id],
            dec_state=self.decoder.make_initial_state(1),
        )
        label = tf.fill([1, 1], hyp.yseq[-1])
        dec_out, state = self.decoder.infer(label,
                                            states=hyp.dec_state,
                                            training=False)

        for t in range(enc_len.numpy()):
            enc_out_t = tf.expand_dims(enc_out[t], axis=0)
            logp = tf.nn.log_softmax(
                self.joint_network(enc_out_t, dec_out)
                / self.softmax_temperature,
                axis=-1
            )
            top_logp, pred = tf.math.top_k(logp, k=1)

            if pred != self.blank_id:
                hyp.yseq.append(int(pred))
                hyp.score += float(top_logp)
                hyp.dec_state = state

                label = tf.fill([1, 1], hyp.yseq[-1])
                dec_out, state = self.decoder.infer(label,
                                                    states=hyp.dec_state,
                                                    training=False)

        return [hyp]


class GreedyRNNT(tf.keras.layers.Layer):

    def __init__(self,
                 decoder: tf.keras.Model,
                 jointer: tf.keras.Model,
                 text_decoder: PreprocessingLayer,
                 max_symbols_per_step: int = 3,
                 return_scores: bool = True,
                 name: str = 'greedy_rnnt',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.decoder = decoder
        self.jointer = jointer
        self.text_decoder = text_decoder
        self.blank_id = text_decoder.pad_id
        self.max_symbols_per_step = max_symbols_per_step
        self.return_scores = return_scores

    @tf.function
    def call(
        self,
        encoder_outputs: tf.Tensor,
        encoder_next_states, hyps, cur_tokens, cur_states
    ) -> tf.Tensor:

        batch_size = tf.shape(encoder_outputs)[0]
        num_frames = 1
        blanks = tf.fill([batch_size, 1], self.blank_id)

        mask = tf.not_equal(hyps, self.blank_id)
        hyps = tf.ragged.boolean_mask(hyps, mask)
        is_blanks = tf.fill([batch_size, 1], False)

        # tf.range
        for i in tf.range(num_frames):
            end_flag = tf.fill([batch_size, 1], False)
            for j in tf.range(self.max_symbols_per_step):
                dec, next_states = self.decoder.infer(cur_tokens,
                                                      training=False,
                                                      states=cur_states)
                enc = tf.expand_dims(encoder_outputs[:, i, :], axis=1)
                pred = self.jointer(enc, dec, training=False)
                pred = tf.squeeze(pred, axis=1)
                pred = tf.nn.log_softmax(pred)
                next_tokens = tf.cast(tf.argmax(pred, axis=-1), dtype=tf.int32)

                _equal = tf.equal(next_tokens, self.blank_id)
                is_blanks = tf.where(tf.reshape(j > 0, [1, 1]), is_blanks,
                                     _equal)
                cur_tokens = tf.where(_equal, cur_tokens, next_tokens)

                end_flag = tf.logical_or(end_flag, _equal)

                hyp = tf.where(end_flag, blanks, next_tokens)
                hyps = tf.concat([hyps, hyp], axis=1)

                cur_states = tf.where(_equal, cur_states, next_states)

                # if _equal.numpy().all():
                #     break

        outputs = self.text_decoder.decode(hyps)
        hyps = hyps.to_tensor(shape=(batch_size, 1000))
        hyps = tf.cast(hyps, dtype=tf.float32)
        cur_tokens = tf.cast(cur_tokens, dtype=tf.float32)
        cached_states = tf.concat(
            (hyps, cur_tokens, encoder_next_states, cur_states), axis=1)
        return outputs, is_blanks, cached_states

    def infer(
        self,
        encoder_outputs: tf.Tensor,
        encoder_lengths: tf.Tensor
    ) -> tf.Tensor:

        batch_size, num_frames = tf.shape(encoder_outputs)[:2]
        cur_tokens = tf.fill([batch_size, 1], self.blank_id)
        cur_states = self.decoder.make_initial_states(batch_size)
        cur_states = tf.nest.flatten(cur_states)
        cur_states = tf.concat(cur_states, axis=1)

        blanks = tf.fill([batch_size, 1], self.blank_id)
        hyps = tf.zeros((batch_size, 1), dtype=tf.int32)
        for i in range(num_frames):
            end_flag = tf.fill([batch_size, 1], False)
            enc = tf.expand_dims(encoder_outputs[:, i, :], axis=1)
            for _ in range(self.max_symbols_per_step):
                dec, next_states = self.decoder.infer(cur_tokens,
                                                      training=False,
                                                      states=cur_states)
                pred = self.jointer(enc, dec, training=False)
                pred = tf.squeeze(pred, axis=1)
                pred = tf.nn.log_softmax(pred)
                next_tokens = tf.cast(tf.argmax(pred, axis=-1), dtype=tf.int32)

                scores = tf.reduce_max(pred, axis=-1)

                _equal = tf.equal(next_tokens, self.blank_id)
                cur_tokens = tf.where(_equal, cur_tokens, next_tokens)

                end_flag = tf.logical_or(end_flag, _equal)
                lengths_flag = tf.logical_or(
                    tf.expand_dims(encoder_lengths <= i, axis=1), end_flag)

                hyp = tf.where(lengths_flag, blanks, next_tokens)
                hyps = tf.concat([hyps, hyp], axis=1)

                cur_states = tf.where(_equal, cur_states, next_states)

                if _equal.numpy().all():
                    break

        if self.return_scores:
            return self.text_decoder.decode(hyps), scores
        else:
            return self.text_decoder.decode(hyps)

    def infer_step(self,
                   encoder_outputs: tf.Tensor,
                   encoder_lengths: tf.Tensor,
                   cur_tokens: tf.Tensor = None,
                   cur_states: tf.Tensor = None) -> tf.Tensor:

        batch_size, num_frames = tf.shape(encoder_outputs)[:2]
        cur_tokens = tf.fill([batch_size, 1],
                             self.blank_id) if cur_tokens is None else tf.cast(
                                 cur_tokens, dtype=tf.int32)
        cur_states = self.decoder.make_initial_states(
            batch_size) if cur_states is None else cur_states
        cur_states = tf.nest.flatten(cur_states)
        cur_states = tf.concat(cur_states, axis=1)

        blanks = tf.fill([batch_size, 1], self.blank_id)
        hyps = tf.zeros((batch_size, 1), dtype=tf.int32)

        for i in range(num_frames):
            end_flag = tf.fill([batch_size, 1], False)
            enc = tf.expand_dims(encoder_outputs[:, i, :], axis=1)
            for _ in range(self.max_symbols_per_step):
                dec, next_states = self.decoder.infer(cur_tokens,
                                                      training=False,
                                                      states=cur_states)
                pred = self.jointer(enc, dec, training=False)
                pred = tf.squeeze(pred, axis=1)
                pred = tf.nn.log_softmax(pred)
                next_tokens = tf.cast(tf.argmax(pred, axis=-1), dtype=tf.int32)

                _equal = tf.equal(next_tokens, self.blank_id)
                cur_tokens = tf.where(_equal, cur_tokens, next_tokens)

                end_flag = tf.logical_or(end_flag, _equal)
                lengths_flag = tf.logical_or(
                    tf.expand_dims(encoder_lengths <= i, axis=1), end_flag)

                hyp = tf.where(lengths_flag, blanks, next_tokens)
                hyps = tf.concat([hyps, hyp], axis=1)

                cur_states = tf.where(_equal, cur_states, next_states)

                if _equal.numpy().all():
                    break
        hyps = self.text_decoder.id_to_string(hyps)
        hyps = tf.strings.regex_replace(hyps, '▁', ' ')
        hyps = tf.strings.regex_replace(hyps, '<blank>', '')
        hyps = tf.strings.reduce_join(hyps, axis=-1)

        return hyps, cur_tokens, cur_states


class GreedyRNNTV2(tf.keras.layers.Layer):

    def __init__(self,
                 decoder: tf.keras.Model,
                 jointer: tf.keras.Model,
                 text_decoder: PreprocessingLayer,
                 max_symbols_per_step: int = 3,
                 name: str = 'greedy_rnnt',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.decoder = decoder
        self.jointer = jointer
        self.text_decoder = text_decoder
        self.blank_id = text_decoder.pad_id
        self.max_symbols_per_step = max_symbols_per_step

    @tf.function
    def call(self,
             encoder_outputs: tf.Tensor,
             encoder_lengths: tf.Tensor) -> tf.Tensor:

        # batch_size, num_frames = tf.shape(encoder_outputs)[:2]
        batch_size = tf.shape(encoder_outputs)[0]
        num_frames = tf.shape(encoder_outputs)[1]
        cur_tokens = tf.fill([batch_size, 1], self.blank_id)
        cur_states = self.decoder.make_initial_states(batch_size)
        cur_states = tf.nest.flatten(cur_states)
        cur_states = tf.concat(cur_states, axis=1)

        blanks = tf.fill([batch_size, 1], self.blank_id)
        hyps = tf.zeros((batch_size, 1), dtype=tf.int32)

        for i in range(num_frames):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(hyps, tf.TensorShape([None, None]))]
            )
            end_flag = tf.fill([batch_size, 1], False)
            enc = tf.expand_dims(encoder_outputs[:, i, :], axis=1)
            for _ in range(self.max_symbols_per_step):
                dec, next_states = self.decoder.infer(cur_tokens,
                                                      training=False,
                                                      states=cur_states)
                pred = self.jointer(enc, dec, training=False)
                pred = tf.squeeze(pred, axis=1)
                pred = tf.nn.log_softmax(pred)
                next_tokens = tf.cast(tf.argmax(pred, axis=-1), dtype=tf.int32)

                _equal = tf.equal(next_tokens, self.blank_id)
                cur_tokens = tf.where(_equal, cur_tokens, next_tokens)

                end_flag = tf.logical_or(end_flag, _equal)
                lengths_flag = tf.logical_or(
                    tf.expand_dims(encoder_lengths <= i, axis=1), end_flag)

                hyp = tf.where(lengths_flag, blanks, next_tokens)
                hyps = tf.concat([hyps, hyp], axis=1)

                cur_states = tf.where(_equal, cur_states, next_states)

                # if _equal.numpy().all():
                #     break
        preds = self.text_decoder.decode(hyps)
        preds = tf.squeeze(preds)
        return preds
