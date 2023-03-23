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
                self.joint_network(enc_out_t, dec_out),
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
