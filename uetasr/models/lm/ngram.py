import math
import kenlm
import tensorflow as tf


class NgramBase:
    """ Ngram interface """

    def __init__(self, ngram_model: str, vocab_path: str):
        """ Initialize NgramBase

        Args:
            ngram_model: path to ngram model
            vocab_path: path to vocabulary
        """

        token_list = []
        with open(vocab_path, encoding="utf-8") as fin:
            token_list = [x.strip().split()[0] for x in fin]
        self.chardict = [x if x != "<eos>" else "</s>" for x in token_list]
        self.charlen = len(self.chardict)
        self.lm = kenlm.LanguageModel(ngram_model)
        self.lm_state = kenlm.State()

    def init_state(self):
        state = kenlm.State()
        self.lm.BeginSentenceWrite(state)
        return state

    def score_partial_(self, y, next_token, state):
        """Score interface for both full and partial scorer.
        Args:
            y: previous char [1]
            next_token: next token need to be score [V]
            state: previous state
            x: encoded feature
        Returns:
            tuple[tf.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of
                `(n_batch, n_vocab)` and next state list for ys.
        """
        if isinstance(y, tf.Tensor):
            y = y.numpy()
        out_state = kenlm.State()
        ys = self.chardict[y[-1]] if y[-1] != 0 else '<s>'
        self.lm.BaseFullScore(state, ys, out_state)
        scores = []
        for j in next_token:
            score = self.lm.BaseFullScore(
                out_state, self.chardict[j], self.lm_state
            )
            scores.append(math.log(10 ** score.log_prob))

        return scores, out_state


class NgramScorer(NgramBase):
    """ Scorer for Ngram """

    def score(self, y, state):
        """Score interface for both full and partial scorer.
        Args:
            y: previous char [1]
            state: previous state
            x: encoded feature
        Returns:
            tuple[tf.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of
                `(n_batch, n_vocab)` and next state list for ys.
        """
        return self.score_partial_(y, list(range(self.charlen)), state)

    def select_state(self, state, i):
        """Empty select state for scorer interface."""
        return state
