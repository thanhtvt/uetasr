import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import \
    PreprocessingLayer
from typing import List

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
                    self.joint_network(enc_out_t, dec_out),
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
