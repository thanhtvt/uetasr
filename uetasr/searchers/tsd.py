import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import \
    PreprocessingLayer
from typing import List, Union

from .hypothesis import Hypothesis
from .base import BaseSearch
from ..utils.common import get_shape


class TSDSearch(BaseSearch):
    """Time synchronous beam search implementation.

    Based on https://ieeexplore.ieee.org/document/9053040

    Args:
        enc_out: Encoder output sequence. (T, D)

    Returns:
        nbest_hyps: N-best hypothesis.

    """

    def __init__(
        self,
        decoder: tf.keras.Model,
        joint_network: tf.keras.Model,
        text_decoder: PreprocessingLayer,
        beam_size: int = 10,
        lm: tf.keras.Model = None,
        lm_weight: float = 0.0,
        max_sym_exp: int = 2,
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
        self.max_sym_exp = max_sym_exp

    def search(self, enc_out: tf.Tensor,
               enc_len: tf.Tensor) -> List[Hypothesis]:
        """ALSD search implementation.

        Args:
            enc_out: Encoder output sequence. (T, D)

        Returns:
            nbest_hyps: N-best hypothesis.
        """

        beam = min(self.beam_size, self.vocab_size)

        beam_state = self.decoder.make_initial_state(beam)

        B = [
            Hypothesis(
                score=0.0,
                yseq=[self.blank_id],
                dec_state=self.decoder.select_state(beam_state, 0),
                lm_state=self.lm.get_initial_state(beam) if self.lm else None)
        ]

        for enc_out_t in enc_out:
            A = []
            C = B

            enc_out_t = tf.expand_dims(enc_out_t, 0)

            for v in range(self.max_sym_exp):
                D = []

                labels = tf.convert_to_tensor([[hyp.yseq[-1]] for hyp in C])
                beam_dec_out, beam_state = self.decoder.infer(labels,
                                                              states=beam_state,
                                                              training=False)
                beam_logp = tf.nn.log_softmax(
                    self.joint_network(enc_out_t, beam_dec_out)
                    / self.softmax_temperature,
                    axis=-1
                )
                beam_logp = tf.squeeze(beam_logp, axis=(0, 1))
                beam_topk = tf.math.top_k(beam_logp[:, 1:], k=beam)
                beam_topk_values = beam_topk.values.numpy()
                beam_topk_indices = beam_topk.indices.numpy()

                seq_A = [h.yseq for h in A]

                for i, hyp in enumerate(C):
                    if hyp.yseq not in seq_A:
                        A.append(
                            Hypothesis(
                                score=hyp.score + float(beam_logp[i, 0]),
                                yseq=hyp.yseq[:],
                                dec_state=hyp.dec_state,
                                lm_state=hyp.lm_state,
                            )
                        )
                    else:
                        dict_pos = seq_A.index(hyp.yseq)

                        A[dict_pos].score = np.logaddexp(
                            A[dict_pos].score,
                            hyp.score + float(beam_logp[i, 0])
                        )

                if v < self.max_sym_exp - 1:
                    if self.lm:
                        raise NotImplementedError("Not support LM for TSD yet!")

                    for i, hyp in enumerate(C):
                        for logp, k in zip(beam_topk_values[i], beam_topk_indices[i] + 1):
                            new_hyp = Hypothesis(
                                score=hyp.score + float(logp),
                                yseq=hyp.yseq[:] + [int(k)],
                                dec_state=self.decoder.select_state(beam_state, i),
                                lm_state=hyp.lm_state,
                            )

                            if self.lm:
                                raise NotImplementedError("Not support LM for ALSD yet!")

                            D.append(new_hyp)

                C = sorted(D, key=lambda x: x.score, reverse=True)[:beam]

            B = sorted(A, key=lambda x: x.score, reverse=True)[:beam]

        return self.sort_nbest(B)
