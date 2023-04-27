import kenlm
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import \
    PreprocessingLayer
from typing import List, Union

from .hypothesis import Hypothesis
from .base import BaseSearch
from ..utils.common import get_shape


class ALSDSearch(BaseSearch):
    """ Alignment-length synchronous decoding implementation
    for Transducer models
    Based on: https://ieeexplore.ieee.org/document/9053040
    """

    def __init__(
        self,
        decoder: tf.keras.Model,
        joint_network: tf.keras.Model,
        text_decoder: PreprocessingLayer,
        beam_size: int = 10,
        lm: tf.keras.Model = None,
        lm_weight: float = 0.0,
        u_max: Union[int, float] = 50,
        score_norm: bool = False,
        nbest: int = 1,
        softmax_temperature: float = 1.0,
        name: str = "alsd_search",
        **kwargs,
    ):
        super(ALSDSearch, self).__init__(
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
        self.u_max = u_max

    def search(self, enc_out: tf.Tensor,
               enc_len: tf.Tensor) -> List[Hypothesis]:
        """ALSD search implementation.

        Args:
            enc_out: Encoder output sequence. (T, D)

        Returns:
            nbest_hyps: N-best hypothesis.
        """
        beam = min(self.beam_size, self.vocab_size)

        t_max = get_shape(enc_out)[0]
        if 0 < self.u_max <= 1:
            u_max = int(self.u_max * t_max)
        else:
            u_max = self.u_max

        beam_state = self.decoder.make_initial_state(beam)

        B = [
            Hypothesis(
                score=0.0,
                yseq=[self.blank_id],
                dec_state=self.decoder.select_state(beam_state, 0),
                lm_state=self.lm.get_initial_state(beam) if self.lm else None)
        ]

        final = []
        for i in range(t_max + u_max):
            A = []
            B_ = []
            B_enc_out = []
            for hyp in B:
                u = len(hyp.yseq) - 1
                t = i - u
                if t > t_max - 1:
                    continue
                B_.append(hyp)
                B_enc_out.append((t, enc_out[t]))

            if B_:
                labels = tf.convert_to_tensor([[hyp.yseq[-1]] for hyp in B_])
                beam_dec_out, beam_state = self.decoder.infer(labels,
                                                              states=beam_state,
                                                              training=False)
                beam_env_out = tf.stack([x[1] for x in B_enc_out])

                beam_logp = tf.nn.log_softmax(
                    self.joint_network(beam_enc_out, beam_dec_out)
                    / self.softmax_temperature,
                    axis=-1
                )
                beam_logp = tf.squeeze(beam_logp, axis=(0, 1))
                beam_topk = tf.math.top_k(beam_logp[:, 1:], k=beam)
                beam_topk_values = beam_topk.values.numpy()
                beam_topk_indices = beam_topk.indices.numpy()

                if self.lm:
                    raise NotImplementedError("Not support LM for ALSD yet!")

                for i, hyp in enumerate(B_):
                    new_hyp = Hypothesis(
                        score=hyp.score + float(beam_logp[i, 0]),
                        yseq=hyp.yseq[:],
                        dec_state=hyp.dec_state,
                        lm_state=hyp.lm_state,
                    )

                    A.append(new_hyp)

                    if B_enc_out[i][0] == t_max - 1:
                        final.append(new_hyp)

                    for logp, k in zip(beam_topk_values[i], beam_topk_indices[i] + 1):
                        new_hyp = Hypothesis(
                            score=hyp.score + float(logp),
                            yseq=hyp.yseq[:] + [int(k)],
                            dec_state=self.decoder.select_state(beam_state, i),
                            lm_state=hyp.lm_state,
                        )

                        if self.lm:
                            raise NotImplementedError("Not support LM for ALSD yet!")

                        A.append(new_hyp)

                B = sorted(A, key=lambda x: x.score, reverse=True)[:beam]
                B = self.recombine_hyps(B)

        if final:
            return self.sort_nbest(final)
        else:
            return self.sort_nbest(B)


    def recombine_hyps(self, hyps: List[Hypothesis]) -> List[Hypothesis]:
        """Recombine hypotheses with same label ID sequence.

        Args:
            hyps: Hypotheses.

        Returns:
            final: Recombined hypotheses.
        """
        final = []

        for hyp in hyps:
            seq_final = [f.yseq for f in final if f.yseq]

            if hyp.yseq in seq_final:
                seq_pos = seq_final.index(hyp.yseq)

                final[seq_pos].score = np.logaddexp(final[seq_pos].score,
                                                    hyp.score)
            else:
                final.append(hyp)

        return final


class ALSDBeamRNNT(tf.keras.layers.Layer):
    """
    Alignment-length synchronous decoding for RNN Transducer implementation
    Paper:  https://ieeexplore.ieee.org/document/9053040
    """

    def __init__(self,
                 model: tf.keras.Model,
                 text_decoder: PreprocessingLayer,
                 fraction: float = 0.65,
                 beam_size: int = 16,
                 temperature: float = 1.0,
                 use_lm: bool = False,
                 lmwt: float = 0.5,
                 lm_path: str = '',
                 name: str = 'alsd_rnnt',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.decoder = model.decoder
        self.jointer = model.jointer
        self.text_decoder = text_decoder
        self.blank_id = text_decoder.pad_id
        self.beam_size = beam_size
        self.temperature = temperature
        self.fraction = fraction
        self.use_lm = use_lm

        if use_lm:
            self.lm = kenlm.LanguageModel(lm_path)
            self.lmwt = lmwt

    def call(self):
        pass

    def infer(self, encoder_outputs: tf.Tensor,
              encoder_lengths: tf.Tensor) -> tf.Tensor:

        ids = tf.range(0, self.decoder.vocab_size)
        non_blank_ids = tf.squeeze(tf.gather(ids, tf.where(ids != 0)))
        index_adder = 1 if self.blank_id == 0 else 0

        batch_size = tf.shape(encoder_outputs)[0]
        cur_tokens = tf.fill([batch_size * self.beam_size, 1], self.blank_id)
        cur_states = self.decoder.make_initial_states(batch_size *
                                                      self.beam_size)
        sequences_len = tf.zeros([batch_size * self.beam_size, 1],
                                 dtype=tf.int32)

        u_max = tf.cast(tf.multiply(tf.cast(encoder_lengths, dtype=tf.float32),
                                    self.fraction),
                        dtype=encoder_lengths.dtype)

        enc_lengths = tf.expand_dims(
            tf.repeat(encoder_lengths, self.beam_size), 1)
        total_lengths = tf.expand_dims(
            tf.repeat(u_max + encoder_lengths, self.beam_size), 1)
        blanks = tf.fill([batch_size * self.beam_size, 1], self.blank_id)
        inf = tf.fill([batch_size * self.beam_size, 1], -float('inf'))
        hyps = tf.zeros([batch_size * self.beam_size, 1], dtype=tf.int32)
        scores = tf.tile(
            tf.convert_to_tensor([[0.0] + [-float("inf")] *
                                  (self.beam_size - 1)]), [batch_size, 1])
        scores = tf.reshape(scores, [-1, 1])

        ends_flag = tf.fill([batch_size * self.beam_size, 1], False)
        final = []
        i = tf.zeros([batch_size * self.beam_size, 1], dtype=tf.int32)

        while not tf.reduce_all(ends_flag):
            t = i - sequences_len
            skips_flag = tf.greater(t, enc_lengths - 1)
            t = tf.where(skips_flag, 0, t)

            dec, next_states = self.decoder.infer(cur_tokens,
                                                  states=cur_states)
            enc = []
            sqt = tf.split(tf.reshape(t, [-1]), batch_size.numpy())
            for j in range(batch_size):
                enc_dim = [encoder_outputs[j, ts, :] for ts in sqt[j].numpy()]
                enc.extend(enc_dim)
            enc = tf.expand_dims(tf.stack(enc, axis=0), axis=1)

            pred = self.jointer(enc, dec)
            pred = tf.squeeze(pred, axis=[1, 2])
            logp = tf.nn.log_softmax(pred / self.temperature)

            # (B x b, 1)
            blank_values = tf.expand_dims(logp[:, self.blank_id], axis=1)

            # (B x b, 1)
            flag = tf.logical_or(skips_flag, ends_flag)
            finals_flag = tf.equal(t, enc_lengths - 1)
            finals_flag = tf.logical_and(finals_flag, tf.logical_not(flag))
            if tf.reduce_any(finals_flag):
                final_scores = tf.where(finals_flag, scores + blank_values,
                                        -float("inf"))
                final.append((hyps, final_scores, sequences_len))

            # (B x b, b - 1)
            topk_logp_values, topk_logp_indices = tf.math.top_k(
                tf.gather(logp, indices=non_blank_ids, axis=1),
                k=self.beam_size - 1,
                sorted=False
            )
            # (B x b, b)
            topk_logp_values = tf.concat([blank_values, topk_logp_values],
                                         axis=1)
            topk_logp_values = tf.where(tf.tile(flag, [1, self.beam_size]),
                                        inf, topk_logp_values)
            topk_logp_indices = tf.concat(
                [blanks, topk_logp_indices + index_adder], axis=1)

            # (B x b, b)
            scores = scores + topk_logp_values
            if self.use_lm:
                new_scores = []
                lm_scores = []
                for idx, hyp in enumerate(hyps.numpy()):
                    hyp = hyp.tolist()
                    sentence = ' '.join(map(str, hyp))
                    old_score = self.lm.score(sentence, bos=True, eos=False)
                    for score, token in zip(scores[idx].numpy(),
                                            topk_logp_indices[idx].numpy()):
                        if score == -float('inf'):
                            new_scores.append(score)
                            lm_scores.append(-float('inf'))
                            continue
                        new_sentence = ' '.join(map(str, hyp + [int(token)]))
                        new_score = self.lm.score(new_sentence,
                                                  bos=True,
                                                  eos=False)
                        lm_score = (new_score -
                                    old_score) / tf.experimental.numpy.log10(
                                        tf.exp(1.0))
                        total_score = score + self.lmwt * lm_score
                        lm_scores.append(lm_score)
                        new_scores.append(total_score)
                scores = tf.reshape(
                    tf.convert_to_tensor(new_scores),
                    [batch_size * self.beam_size, self.beam_size])
                scores = tf.cast(scores, dtype=tf.float32)
                lm_scores = tf.reshape(
                    tf.convert_to_tensor(lm_scores),
                    [batch_size * self.beam_size, self.beam_size])
            # (B, b x b)
            scores = tf.reshape(scores,
                                [batch_size, self.beam_size * self.beam_size])
            # (B, b)
            topk_scores_values, topk_scores_indices = tf.math.top_k(
                scores, k=self.beam_size, sorted=False)

            # (B x b, 1)
            scores = tf.reshape(topk_scores_values, [-1, 1])
            # (B x b)
            topk_scores_indices = tf.reshape(topk_scores_indices, [-1])

            # (B x b)
            base_indices = tf.repeat(tf.range(0, batch_size), self.beam_size)
            base_indices = base_indices * self.beam_size * self.beam_size
            best_indices = base_indices + topk_scores_indices

            # (B x b x b)
            topk_logp_indices = tf.reshape(topk_logp_indices, [-1])
            # (B x b)
            next_tokens = tf.gather(topk_logp_indices, best_indices)
            # (B x b, 1)
            next_tokens = tf.expand_dims(next_tokens, axis=1)

            _equal = tf.equal(next_tokens, blanks)
            cur_tokens = tf.where(_equal, cur_tokens, next_tokens)

            hyp = tf.where(flag, blanks, next_tokens)
            best_hyp_indices = best_indices // self.beam_size
            best_hyps = tf.gather(hyps, best_hyp_indices, axis=0)
            hyps_mask = tf.where(best_hyps != 0, 1, 0)
            last_nonzero_indices = tf.argmax(tf.multiply(
                hyps_mask, tf.range(1, best_hyps.shape[1] + 1)),
                                             axis=1,
                                             output_type=tf.int32)
            cur_tokens = tf.gather(best_hyps,
                                   last_nonzero_indices,
                                   axis=1,
                                   batch_dims=1)
            cur_tokens = tf.expand_dims(cur_tokens, axis=1)
            cur_tokens = tf.where(_equal, cur_tokens, next_tokens)

            # (B x b, len)
            hyps = tf.concat([best_hyps, hyp], axis=1)
            scores = self.recombine_hypotheses(hyps, scores)

            new_cur_states = []
            cur_states = tf.where(_equal, cur_states, next_states)

            i = tf.where(i < total_lengths - 1, i + 1, i)
            ends_flag = tf.greater_equal(i, total_lengths - 1)
            sequences_len = tf.expand_dims(tf.math.count_nonzero(
                hyps, axis=1, dtype=tf.int32),
                                           axis=1)

        if final:
            # (B x b, len(final))
            scores = tf.concat([f[1] for f in final], axis=1)
            # (B, b x len(final))
            scores = tf.reshape(scores, [batch_size, -1])
            # (B)
            best_index = tf.argmax(scores, axis=1, output_type=tf.int32)
            best_scores = tf.reduce_max(scores, axis=1)

            beam_idx = best_index // len(final)
            final_idx = best_index % len(final)
            base_index = tf.range(batch_size, dtype=tf.int32) * self.beam_size
            beam_idx = beam_idx + base_index

            preds = tf.ragged.constant([[0]])
            for fin_id, beam_id in zip(final_idx, beam_idx):
                pred = tf.expand_dims(final[fin_id][0][beam_id], axis=0)
                preds = tf.concat([preds, pred], axis=0)
            preds = preds[1:]
        else:
            preds = tf.zeros([batch_size, 1], dtype=tf.int32)

        return self.text_decoder.decode(preds), best_scores

    def recombine_hypotheses(self, hyps, scores):
        """
        hyps: (B x b, len)
        scores: (B x b, 1)
        """

        def remove_zeros(tensor):
            num_non_zero = tf.math.count_nonzero(tensor,
                                                 axis=1,
                                                 dtype=tf.int32)
            flat_tensor = tf.reshape(tensor, [-1])
            mask = tf.logical_not(tf.equal(flat_tensor, 0))
            flat_non_zero_tensor = tf.boolean_mask(flat_tensor, mask)
            ragged_tensor = tf.RaggedTensor.from_row_lengths(
                flat_non_zero_tensor, num_non_zero)
            padded_tensor = tf.cast(ragged_tensor.to_tensor(), dtype=tf.int32)

            return padded_tensor

        batch_size = int(hyps.shape[0] / self.beam_size)
        # (B x b, L)
        hyps = remove_zeros(hyps)
        scores = tf.reshape(scores, [-1])
        hyps_by_batch = tf.split(hyps, batch_size, axis=0)
        for i, batch_hyps in enumerate(hyps_by_batch):
            unique_tensor, _ = tf.raw_ops.UniqueV2(x=batch_hyps, axis=[0])
            if tf.shape(unique_tensor)[0] == tf.shape(batch_hyps)[0]:
                continue
            for tensor in unique_tensor:
                tensor = tf.expand_dims(tensor, axis=0)
                # (n, 1)
                matched_idx = tf.where(
                    tf.reduce_sum(batch_hyps - tensor, axis=1) == 0)
                matched_idx = tf.cast(matched_idx + i * self.beam_size,
                                      dtype=tf.int32)
                if matched_idx.shape[0] > 1:
                    matched_scores = tf.squeeze(tf.gather(scores, matched_idx))
                    new_score = tf.math.log(
                        tf.reduce_sum(tf.exp(matched_scores)))
                    new_score = tf.reshape(new_score, [1])
                    inf = tf.repeat(-float('inf'),
                                    tf.shape(matched_scores)[0] - 1)
                    updated_scores = tf.concat([new_score, inf], axis=0)
                    scores = tf.tensor_scatter_nd_update(
                        scores, matched_idx, updated_scores)

        return tf.expand_dims(scores, axis=1)
