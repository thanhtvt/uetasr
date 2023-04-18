import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import \
    PreprocessingLayer
from typing import List

from .hypothesis import Hypothesis


class BaseSearch(tf.keras.layers.Layer):
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
        name: str = "base_search_transducer",
        **kwargs,
    ):
        super(BaseSearch, self).__init__(name=name, **kwargs)
        self.decoder = decoder
        self.joint_network = joint_network
        self.text_decoder = text_decoder

        self.beam_size = beam_size
        self.blank_id = text_decoder.pad_id
        self.vocab_size = decoder.vocab_size

        self.lm = lm
        self.lm_weight = lm_weight

        self.nbest = nbest
        self.score_norm = score_norm
        self.softmax_temperature = softmax_temperature

    def infer(
        self,
        encoder_outs: tf.Tensor,
        encoder_lengths: tf.Tensor
    ) -> tf.Tensor:
        """Perform beam search.
        Args:
            encoder_outs: Encoder output sequence. (B, T, D_enc)
            encoder_lengths: Encoder output sequence length. (B)
        Returns:
            nbest_hyps: N-best decoding results
        """

        nbest_hyps, scores = [], []
        for enc_out, enc_len in zip(encoder_outs, encoder_lengths):
            nbest_hyp = self.search(enc_out, enc_len)
            hyp = self.text_decoder.decode(nbest_hyp.yseq)
            score = nbest_hyp.score
            scores.append(score)
            nbest_hyps.append(hyp)
        return tf.convert_to_tensor(nbest_hyps, dtype=tf.string), \
            tf.convert_to_tensor(scores, dtype=tf.float32)

    def sort_nbest(self, hyps: List[Hypothesis]) -> List[Hypothesis]:
        """Sort hypotheses by score or score given sequence length.
        Args:
            hyps: Hypothesis.
        Return:
            hyps: Sorted hypothesis.
        """
        if self.score_norm:
            hyps.sort(key=lambda x: x.score / len(x.yseq), reverse=True)
        else:
            hyps.sort(key=lambda x: x.score, reverse=True)

        return hyps[0]

    def search(self, enc_out: tf.Tensor, enc_len: tf.Tensor) -> List[Hypothesis]:
        raise NotImplementedError
