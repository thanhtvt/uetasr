import tensorflow as tf

from typing import Tuple
from ...utils.common import get_rnn


class RNNDecoder(tf.keras.Model):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        project_dim: int = None,
        num_layers: int = 1,
        hidden_dim: int = 256,
        dropout_embed: float = 0.2,
        dropout_rnn: float = 0.1,
        rnn_type: str = 'lstm',
        use_softmax: bool = False,
        name: str = 'rnn_decoder',
        **kwargs,
    ):
        super(RNNDecoder, self).__init__(name=name, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embed = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.dropout_embed = tf.keras.layers.Dropout(dropout_embed)
        self.dropout_rnn = tf.keras.layers.Dropout(dropout_rnn)
        self.use_softmax = use_softmax
        project_dim = project_dim or hidden_dim

        rnn_class = get_rnn(rnn_type)
        self.rnns = []
        for _ in range(num_layers):
            self.rnns.append((
                rnn_class(
                    hidden_dim,
                    return_sequences=True,
                    return_state=True,
                    implementation=1,
                    dropout=0.1,
                    recurrent_dropout=0.1,
                ),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dense(project_dim)
            ))

    def summary(self):
        inputs = tf.keras.Input(shape=(None,), batch_size=None)
        outputs = self(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.summary()

    def call(
        self,
        inputs: tf.Tensor,
        training: bool = False
    ):
        x = self.embed(inputs)
        x = self.dropout_embed(x, training=training)

        for rnn, fc, ln in self.rnns:
            x, *_ = rnn(x, training=training)
            x = fc(x, training=training)
            x = ln(x, training=training)
        x = self.dropout_rnn(x, training=training)

        if self.use_softmax:
            x = tf.nn.softmax(x, axis=-1)
        return x

    def infer(
        self,
        inputs: tf.Tensor,
        states: tf.Tensor = None,
        training: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor]:

        x = self.embed(inputs)
        x = self.dropout_embed(x, training=training)
        states = tf.split(states, num_or_size_splits=len(self.rnns), axis=1)
        states = [
            tf.split(state, num_or_size_splits=2, axis=1) for state in states
        ]
        final_states = []

        for i in range(len(self.rnns)):
            rnn, fc, ln = self.rnns[i]
            x, *fin_states = rnn(x, initial_state=states[i], training=training)
            x = fc(x, training=training)
            # x = tf.keras.activations.tanh(x)
            x = ln(x, training=training)
            final_states.extend(fin_states)
        x = self.dropout_rnn(x, training=training)
        final_states = tf.concat(final_states, axis=1)

        if self.use_softmax:
            x = tf.nn.softmax(x, axis=-1)
        return x, final_states

    def make_initial_states(self, batch_size: int) -> tf.Tensor:
        inputs = tf.random.uniform((batch_size, self.hidden_dim))
        states = [rnn[0].get_initial_state(inputs) for rnn in self.rnns]
        states = tf.nest.flatten(states)
        states = tf.concat(states, axis=1)
        return states

    def select_state(self, states: tf.Tensor, index: int):
        states = tf.split(states, num_or_size_splits=len(self.rnns), axis=1)
        states = [
            tf.split(state, num_or_size_splits=2, axis=1) for state in states
        ]
        fin_states = []
        for i in range(len(self.rnns)):
            fin_states.append([states[i][0][index][tf.newaxis],
                               states[i][1][index][tf.newaxis]])
        fin_states = tf.nest.flatten(fin_states)
        fin_states = tf.concat(fin_states, axis=1)
        return fin_states

    def get_config(self):
        config = super(RNNDecoder, self).get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'hidden_dim': self.hidden_dim,
            'use_softmax': self.use_softmax,
        })
        config.update(self.embed.get_config())
        config.update(self.dropout_embed.get_config())
        config.update(self.dropout_rnn.get_config())
        for rnn, ln, fc in self.rnns:
            config.update(rnn.get_config())
            config.update(ln.get_config())
            config.update(fc.get_config())
        return config
