import numpy as np
import os
import sentencepiece as spm
import tensorflow as tf
import tensorflow_text as tftext
from typing import Union, List, Tuple


class Subword(tf.keras.layers.experimental.preprocessing.PreprocessingLayer):

    def __init__(self,
                 model_prefix: str,
                 data_path: str = "",
                 character_coverage: float = 1.0,
                 model_type: str = "bpe",
                 num_threads: int = 16,
                 unk_id: int = 0,
                 bos_id: int = -1,
                 eos_id: int = -1,
                 pad_id: int = -1,
                 unk_piece: str = '<unk>',
                 bos_piece: str = '<s>',
                 eos_piece: str = '</s>',
                 pad_piece: str = '<pad>',
                 vocab_size: int = 1024,
                 out_type: tf.DType = tf.int32,
                 user_defined_symbols: str = "",
                 name: str = 'subword',
                 **kwargs):

        super().__init__(trainable=False, name=name, **kwargs)

        self.model_prefix = model_prefix
        model_path = model_prefix + '.model'
        self.character_coverage = character_coverage
        self.model_type = model_type
        self.num_threads = num_threads
        self.unk_id = unk_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.unk_piece = unk_piece
        self.bos_piece = bos_piece
        self.eos_piece = eos_piece
        self.pad_piece = pad_piece
        self.vocab_size = vocab_size
        self.user_defined_symbols = user_defined_symbols
        self.use_string = (out_type == tf.string)

        if not os.path.isfile(model_path) and data_path:
            self.adapt(data_path)

        with open(model_path, 'rb') as fin:
            model = fin.read()

        self.tokenizer = tftext.SentencepieceTokenizer(
            model=model, out_type=out_type, nbest_size=0, alpha=1.0,
            reverse=False, add_bos=False, add_eos=False,
            return_nbest=False, name=name
        )

        self.decode.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int32, name='input')
        )

    def adapt(self, data_path):
        spm.SentencePieceTrainer.train(
            input=data_path,
            model_prefix=self.model_prefix,
            vocab_size=self.vocab_size,
            character_coverage=self.character_coverage,
            model_type=self.model_type,
            num_threads=self.num_threads,
            unk_id=self.unk_id,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            pad_id=self.pad_id,
            unk_piece=self.unk_piece,
            bos_piece=self.bos_piece,
            eos_piece=self.eos_piece,
            pad_piece=self.pad_piece,
            user_defined_symbols=self.user_defined_symbols
        )

    def prepend_blank(self, tokens):
        if isinstance(tokens, tf.Tensor):
            blanks = tf.broadcast_to(
                self.pad_id, (tf.shape(tokens)[0], 1))
        else:
            blanks = tf.broadcast_to(
                self.pad_id, (tokens.bounding_shape()[0], 1))
        tokens = tf.concat([blanks, tokens], axis=-1)
        return tokens

    def append_blank(self, tokens):
        if isinstance(tokens, tf.Tensor):
            blanks = tf.broadcast_to(
                self.pad_id, (tf.shape(tokens)[0], 1))
        else:
            blanks = tf.broadcast_to(
                self.pad_id, (tokens.bounding_shape()[0], 1))
        tokens = tf.concat([tokens, blanks], axis=-1)
        return tokens

    def prepend_bos(self, tokens):
        bos = tf.broadcast_to(
                self.bos_id, (tf.shape(tokens)[0], 1))
        tokens = tf.concat([bos, tokens], axis=-1)
        return tokens

    def append_eos(self, tokens):
        eos = tf.broadcast_to(
                self.eos_id, (tf.shape(tokens)[0], 1))
        tokens = tf.concat([tokens, eos], axis=-1)
        return tokens

    def call(
        self,
        text: Union[tf.Tensor, List, np.ndarray],
        use_batch: bool = True,
        preblank: bool = False,
        postblank: bool = False,
        prepend_bos: bool = False,
        append_eos: bool = False,
    ) -> Union[tf.Tensor, Tuple[tf.Tensor]]:
        sparse_tokens = self.tokenizer.tokenize(text)
        if use_batch:
            blank = self.pad_piece if self.use_string else self.pad_id
            dense_tokens = sparse_tokens.to_tensor(default_value=blank)
        else:
            dense_tokens = tf.expand_dims(sparse_tokens, axis=0)

        if preblank:
            dense_tokens = self.prepend_blank(dense_tokens)

        if postblank:
            dense_tokens = self.append_blank(dense_tokens)

        if prepend_bos:
            dense_tokens = self.prepend_bos(dense_tokens)

        if append_eos:
            dense_tokens = self.append_eos(dense_tokens)

        if not use_batch:
            dense_tokens = tf.squeeze(dense_tokens, axis=0)
        return dense_tokens

    @tf.function
    def decode(self,
               dense_tokens: Union[tf.Tensor, np.ndarray]) -> tf.Tensor:
        return self.tokenizer.detokenize(dense_tokens)

    @tf.function
    def id_to_string(
        self,
        dense_tokens: Union[tf.Tensor, np.ndarray]
    ) -> tf.Tensor:
        return self.tokenizer.id_to_string(dense_tokens)
