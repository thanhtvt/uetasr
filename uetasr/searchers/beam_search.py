import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import \
    PreprocessingLayer
from typing import List, Union

from .hypothesis import Hypothesis
from .base import BaseSearch


class BeamSearch(BaseSearch):
    """ Beam search implementation for Transducer models """

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
        name: str = "beam_search",
        **kwargs,
    ):
        super(BeamSearch, self).__init__(
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
        """Beam search implementation.

        Modified from https://arxiv.org/pdf/1211.3711.pdf

        Args:
            enc_out: Encoder output sequence. (T, D)

        Returns:
            nbest_hyps: N-best hypothesis.
        """

        beam = min(self.beam_size, self.vocab_size)
        beam_k = min(beam, self.vocab_size - 1)

        kept_hyps = [
            Hypothesis(
                score=0.0,
                yseq=[self.blank_id],
                dec_state=self.decoder.make_initial_state(1),
                lm_state=self.lm.get_initial_state(1) if self.lm else None
            )
        ]

        for t in range(enc_len.numpy()):
            hyps = kept_hyps
            kept_hyps = []

            cnt = 0
            while cnt < 100:
                max_hyp = max(hyps,
                              key=lambda x: x.score / len(x.yseq)
                              if self.score_norm else x.score)
                hyps.remove(max_hyp)
                label = tf.fill([1, 1], max_hyp.yseq[-1])

                dec_out, state = self.decoder.infer(label,
                                                    states=max_hyp.dec_state,
                                                    training=False)

                enc_out_t = tf.expand_dims(enc_out[t], axis=0)
                # [1, 1, 1, V]
                logp = tf.nn.log_softmax(
                    self.joint_network(enc_out_t, dec_out)
                    / self.softmax_temperature,
                    axis=-1
                )
                # [V]
                logp = tf.squeeze(logp, axis=(0, 1, 2)).numpy()
                # [B]
                top_k = tf.math.top_k(logp[1:], k=beam_k)
                topk_values = top_k.values.numpy()
                topk_indices = top_k.indices.numpy()

                if self.lm:
                    token = label[0]
                    lm_scores, lm_state = self.lm.score(
                        token, max_hyp.lm_state)
                else:
                    lm_state = max_hyp.lm_state

                kept_hyps.append(
                    Hypothesis(score=max_hyp.score + float(logp[0]),
                               yseq=max_hyp.yseq,
                               dec_state=max_hyp.dec_state,
                               lm_state=max_hyp.lm_state))

                for logp, k in zip(topk_values, topk_indices):
                    if self.lm:
                        score = max_hyp.score + float(logp) + \
                            self.lm_weight * float(lm_scores)
                    else:
                        score = max_hyp.score + float(logp)

                    hyps.append(
                        Hypothesis(score=score,
                                   yseq=max_hyp.yseq + [int(k + 1)],
                                   dec_state=state,
                                   lm_state=lm_state))

                hyps_max = max(hyps,
                               key=lambda x: x.score / len(x.yseq)
                               if self.score_norm else x.score)
                hyps_max_score = hyps_max.score / len(hyps_max.yseq) \
                    if self.score_norm else hyps_max.score

                kept_most_prob = sorted(
                    [
                        hyp for hyp in kept_hyps
                        if (hyp.score / len(hyp.yseq) if self.score_norm
                            else hyp.score) > hyps_max_score
                    ],
                    key=lambda x: -x.score / len(x.yseq)
                    if self.score_norm else -x.score,
                )
                if len(kept_most_prob) >= beam:
                    kept_hyps = kept_most_prob  # ???
                    break
                cnt += 1

            for hyp in kept_hyps:
                print(
                    hyp.score, hyp.score / len(hyp.yseq), ''.join([
                        token.decode('utf-8') for token in
                        self.text_decoder.id_to_string(hyp.yseq).numpy()
                    ]))
            print('+' * 10)

        return self.sort_nbest(kept_hyps)


class BeamRNNT(tf.keras.layers.Layer):

    def __init__(self,
                 decoder: tf.keras.Model,
                 jointer: tf.keras.Model,
                 text_decoder: PreprocessingLayer,
                 max_symbols_per_step: int = 3,
                 beam: int = 5,
                 alpha: float = 0.6,
                 lmwt: float = 0.3,
                 lm_path: str = '',
                 lm: tf.keras.Model = None,
                 name: str = 'beam_rnnt',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.decoder = decoder
        self.jointer = jointer
        self.text_decoder = text_decoder
        self.beam = beam
        self.alpha = alpha
        self.lmwt = lmwt
        if lm:
            lm.load_weights(lm_path).expect_partial()
            # lm = kenlm.LanguageModel(lm_path)
        self.lm = lm
        self.blank_id = text_decoder.pad_id
        self.max_symbols_per_step = max_symbols_per_step

    def infer(self, encoder_outputs: Union[tf.Tensor, np.ndarray],
              encoder_lengths: Union[tf.Tensor, np.ndarray]) -> tf.Tensor:

        beam_size = self.beam
        batch_size, num_frames = tf.shape(encoder_outputs)[:2]

        zeros = tf.fill([batch_size * beam_size, beam_size], 0.0)
        inf = tf.fill([batch_size * beam_size, 1], -float('inf'))

        cur_tokens = tf.fill([batch_size * beam_size, 1], self.blank_id)
        # [N, B x b, H]
        cur_states = self.decoder.make_initial_states(batch_size * beam_size)
        # cur_states = tf.nest.flatten(cur_states)
        # cur_states = tf.concat(cur_states, axis=1)

        if self.lm:
            lm_cur_states = self.lm.make_initial_states(batch_size * beam_size)
            # lm_cur_states = tf.nest.flatten(lm_cur_states)
            # lm_cur_states = tf.concat(lm_cur_states, axis=1)
            # lm_cur_states = [kenlm.State() for _ in range(batch_size * beam_size)]
            # self.lm.BeginSentenceWrite()

        encoder_lengths = tf.expand_dims(tf.repeat(encoder_lengths, beam_size),
                                         axis=1)
        zeros_mask = tf.fill([batch_size * beam_size, 1], False)
        blanks = tf.fill([batch_size * beam_size, 1], self.blank_id)
        hyps = tf.zeros((batch_size * beam_size, 1), dtype=tf.int32)

        scores = tf.tile(
            tf.convert_to_tensor([[0.0] + [-float("inf")] * (beam_size - 1)]),
            [batch_size, 1])
        scores = tf.reshape(scores, [-1, 1])
        for i in tf.range(num_frames):
            enc = tf.repeat(tf.expand_dims(encoder_outputs[:, i, :], axis=1),
                            beam_size,
                            axis=0)
            end_flag = tf.fill([batch_size * beam_size, 1], False)

            for j in tf.range(self.max_symbols_per_step):
                dec, next_states = self.decoder.infer(cur_tokens,
                                                      training=False,
                                                      states=cur_states)

                if self.lm:
                    lm_outs, lm_next_states = self.lm.infer(
                        cur_tokens, training=False, states=lm_cur_states)
                    # [B x b, n x dim x 2]
                    lm_next_states = tf.repeat(lm_next_states,
                                               repeats=beam_size,
                                               axis=0)
                    # [B x b, 1, V]
                    lm_outs = tf.squeeze(lm_outs, axis=1)
                    # [B x b, V]
                    lm_outs = tf.nn.log_softmax(lm_outs)

                pred = self.jointer(enc, dec, training=False)
                pred = tf.squeeze(pred, axis=[1, 2])
                pred = tf.nn.log_softmax(pred)

                topk_logp = tf.math.top_k(pred, k=beam_size, sorted=True)
                # [B x b, b]
                topk_logp_value = topk_logp.values
                # [B x b, b]
                topk_logp_index = topk_logp.indices

                if self.lm:
                    # [B x b, b]
                    topk_logp_lm = tf.gather(lm_outs,
                                             topk_logp_index,
                                             axis=1,
                                             batch_dims=1)
                    topk_logp_lm = tf.where(topk_logp_index == self.blank_id,
                                            0.0, topk_logp_lm)

                flag = tf.math.logical_or(end_flag, encoder_lengths <= i)
                if beam_size > 1:
                    unfinished = tf.concat(
                        [zeros_mask,
                         tf.tile(flag, [1, beam_size - 1])],
                        axis=1)
                    finished = tf.concat(
                        [flag, tf.tile(zeros_mask, [1, beam_size - 1])],
                        axis=1)
                else:
                    unfinished = zeros_mask
                    finished = flag

                # [B x b x b]
                topk_logp_index = tf.reshape(topk_logp_index, [-1])
                topk_logp_value = tf.where(unfinished, inf, topk_logp_value)
                topk_logp_value = tf.where(finished, zeros, topk_logp_value)

                if self.lm:
                    topk_logp_lm = tf.where(unfinished, inf, topk_logp_lm)
                    topk_logp_lm = tf.where(finished, zeros, topk_logp_lm)

                topk_logp_index = tf.where(
                    tf.repeat(tf.squeeze(flag, axis=1), beam_size),
                    tf.fill([batch_size * beam_size * beam_size],
                            self.blank_id), topk_logp_index)

                # [B x b, b]
                scores = scores + topk_logp_value
                if self.lm:
                    scores = scores + self.lmwt * topk_logp_lm

                lengths = tf.cast(tf.not_equal(hyps, self.blank_id),
                                  dtype=tf.float32)
                lengths = tf.math.reduce_sum(lengths, axis=-1, keepdims=True)
                lengths = lengths + tf.cast(tf.not_equal(
                    tf.reshape(topk_logp_index, (
                        batch_size * beam_size,
                        beam_size,
                    )), self.blank_id),
                                            dtype=tf.float32)

                # ref: https://opennmt.net/OpenNMT/translation/beam_search/
                # alpha = 0 to off normalize by lengths
                lengths = tf.cast(tf.math.pow(5 + lengths, self.alpha) /
                                  6**self.alpha,
                                  dtype=tf.float32)
                # lengths = tf.math.maximum(lengths, 1)
                # norm_scores = tf.cast(scores / lengths, dtype=tf.float32)
                norm_scores = scores

                # [B, b x b]
                scores = tf.reshape(scores,
                                    [batch_size, beam_size * beam_size])
                norm_scores = tf.reshape(norm_scores,
                                         [batch_size, beam_size * beam_size])
                # [B, b]
                topk_scores = tf.math.top_k(norm_scores,
                                            k=beam_size,
                                            sorted=True)
                # [B x b, 1]
                # scores = tf.reshape(topk_scores.values, [-1, 1])
                scores = tf.gather(scores, topk_scores.indices, batch_dims=-1)
                scores = tf.reshape(scores, [-1, 1])
                # [B, b]
                topk_scores_index = topk_scores.indices
                # [B x b]
                topk_scores_index = tf.reshape(topk_scores_index, [-1])

                # [B x b]
                base_index = tf.repeat(tf.range(0, batch_size),
                                       repeats=beam_size)
                base_index = base_index * beam_size * beam_size
                # [B x b]
                best_index = base_index + topk_scores_index

                # [B x b]
                next_tokens = tf.gather(topk_logp_index, best_index)

                # [B x b, 1]
                next_tokens = tf.expand_dims(next_tokens, axis=1)

                _equal = tf.equal(next_tokens, self.blank_id)
                end_flag = tf.logical_or(end_flag, _equal)

                # [B x b]
                best_hyp_index = best_index // beam_size
                best_hyps = tf.gather(hyps, best_hyp_index, axis=0)

                # cur_tokens is not true with new next_tokens, regather them from best_hyps
                hyps_mask = tf.where(best_hyps != self.blank_id, 1, 0)
                last_nonzero_indices = tf.argmax(tf.multiply(
                    hyps_mask, tf.range(1, best_hyps.shape[1] + 1)),
                                                 axis=1,
                                                 output_type=tf.int32)
                cur_tokens = tf.gather(best_hyps,
                                       last_nonzero_indices,
                                       axis=1,
                                       batch_dims=1)
                cur_tokens = tf.expand_dims(cur_tokens, axis=1)

                cur_tokens = tf.where(end_flag, cur_tokens, next_tokens)
                hyp = tf.where(
                    tf.math.logical_or(end_flag, encoder_lengths <= i), blanks,
                    next_tokens)

                hyps = tf.concat([best_hyps, hyp], axis=1)

                # cur_states is not true with new cur_tokens
                # regather them from cur_states by best_hyp_index
                # choose cur_states by new best_hyp_index
                # [B x b, N x H]
                cur_states = tf.gather(cur_states, best_hyp_index, axis=0)

                # [B x b, N x H] -> [B x b, N x H]
                # repeat b times, [B x b, N x H] -> [B x b x b, N x H]
                next_states = tf.repeat(next_states, repeats=beam_size, axis=0)

                # choose next_states by best_index
                next_states = tf.gather(next_states, best_index, axis=0)

                cur_states = tf.where(end_flag, cur_states, next_states)
                if self.lm:
                    lm_cur_states = tf.gather(lm_cur_states,
                                              best_hyp_index,
                                              axis=0)
                    lm_next_states = tf.gather(lm_next_states,
                                               best_index,
                                               axis=0)
                    lm_cur_states = tf.where(end_flag, lm_cur_states,
                                             lm_next_states)

                if _equal.numpy().all():
                    break

        # [B, b]
        scores = tf.reshape(scores, [batch_size, beam_size])
        # [B]
        best_index = tf.cast(tf.argmax(scores, axis=-1), dtype=tf.int32)
        base_index = tf.cast(tf.range(batch_size) * beam_size, dtype=tf.int32)
        best_index = best_index + base_index
        best_hyps = tf.gather(hyps, best_index)

        # Get final score [B]
        best_scores = tf.reduce_max(scores, axis=-1)
        outputs = self.text_decoder.decode(best_hyps)

        return outputs, best_scores
