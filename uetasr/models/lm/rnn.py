import tensorflow as tf

from ...utils.common import get_rnn


class RNNLM(tf.keras.Model):
    """ Sequential RNNLM.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_layers: int,
        hidden_dim: int,
        project_dim: int,
        dropout_rate: float = 0.1,
        rnn_type: str = "LSTM",
        use_softmax: bool = False,
        model_path: str = "",
        name: str = "rnnlm",
        **kwargs,
    ):
        super(RNNLM, self).__init__(name=name, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embed = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.use_softmax = use_softmax

        rnn_class = get_rnn(rnn_type)

        self.rnns = []
        for _ in range(num_layers):
            rnn = rnn_class(hidden_dim,
                            return_sequences=True,
                            return_state=True,
                            implementation=1,
                            dropout=0.1)
            fc = tf.keras.layers.Dense(project_dim)
            self.rnns.append((rnn, fc))
        if model_path:
            self.load_weights(model_path).expect_partial()

    def summary(self):
        inputs = tf.keras.Input(shape=(None,), batch_size=None)
        outputs = self.call(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        model.summary()

    def call(self, inputs, training=False, mask=None):
        x = self.embed(inputs, training=training)
        x = self.do(x, training=training)

        for rnn, fc in self.rnns:
            outs = rnn(x, training=training)
            x = outs[0]
            x = fc(x, training=training)
        if self.use_softmax:
            x = tf.nn.log_softmax(x, axis=-1)
        return x

    def score(self, inputs, states=None):
        inputs = tf.expand_dims(inputs, axis=0)  # [1, 1]
        x = self.embed(inputs, training=False)
        x = self.do(x, training=False)
        states = tf.split(states, num_or_size_splits=len(self.rnns), axis=1)
        states = [tf.split(state, num_or_size_splits=2, axis=1) for state in states]
        final_states = []
        for i in range(len(self.rnns)):
            (rnn, fc) = self.rnns[i]
            outs = rnn(x, training=False, initial_state=states[i])
            x, final_state = outs[0], outs[1:]
            x = fc(x, training=False)
            final_states.extend(final_state)
        final_states = tf.concat(final_states, axis=1)
        x = tf.nn.log_softmax(x, axis=-1)
        return x[0][0].numpy(), final_states
