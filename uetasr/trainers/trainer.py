import csv
import os
import tqdm
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from typing import Union, List

from .base import BaseTrainer
from .optimizers.accumulation import GradientAccumulator
from ..metrics.toggle import ToggleMetrics
from ..models.accumulators import GradientAccumulateModel
from ..utils.common import has_devices


class ASRTrainer(BaseTrainer):

    def __init__(
        self,
        model: tf.keras.Model,
        learning_rate: Union[float, LearningRateSchedule],
        beam_decoder: tf.keras.layers.Layer,
        optimizer: tf.keras.optimizers.Optimizer,
        log_append: bool = False,
        accum_steps: int = 1,
        losses: List[tf.keras.losses.Loss] = [],
        loss_weights: List[float] = [],
        metrics: List[tf.keras.metrics.Metric] = [],
        num_epochs: int = 1,
        jit_compile: bool = False,
        steps_per_execution: int = 1,
        callbacks: List[tf.keras.callbacks.Callback] = [],
        train_num_samples: int = -1,
        dev_num_samples: int = -1,
        pretrained_model: str = "",
    ):
        # Apply gradient accumulation
        if accum_steps > 1 and has_devices("GPU") > 1:
            if has_devices("GPU") > 1:
                optimizer = GradientAccumulator(optimizer, accum_steps)
            elif has_devices("GPU") == 1:  # GA model is not stable multi-gpus
                model = GradientAccumulateModel(accum_steps=accum_steps,
                                                mixed_precision=False,
                                                use_agc=True,
                                                inputs=model.input,
                                                outputs=model.output)

        self.optimizer = optimizer
        self.model = model

        if pretrained_model:
            self.load_model(pretrained_model)

        self.model.compile(loss=losses,
                           loss_weights=loss_weights,
                           optimizer=optimizer,
                           metrics=metrics,
                           steps_per_execution=steps_per_execution,
                           jit_compile=jit_compile)

        self.num_epochs = num_epochs
        self.train_num_samples = train_num_samples
        self.dev_num_samples = dev_num_samples

        self.decoder = beam_decoder
        self.text_decoder = beam_decoder.text_decoder if beam_decoder else None

        # Init callbacks
        self.callbacks = callbacks

        for m in metrics:
            self.callbacks.append(ToggleMetrics(m))

    def train(
        self,
        train_loader: tf.data.Dataset,
        dev_loader: tf.data.Dataset,
        cmvn_loader: tf.data.Dataset = None,
    ):
        if cmvn_loader and self.model.cmvn:
            print("Start compute cmvn...")
            self.model.adapt(cmvn_loader, batch_size=1)
            print("Finish compute cmvn.")

        if self.train_num_samples != -1:
            train_loader = train_loader.repeat().take(self.train_num_samples)

        if self.dev_num_samples != -1:
            dev_loader = dev_loader.repeat().take(self.dev_num_samples)

        self.model.fit(
            train_loader,
            epochs=self.num_epochs,
            validation_data=dev_loader,
            callbacks=self.callbacks,
        )

    def evaluate(
        self,
        test_loader: tf.data.Dataset,
        result_dir: str = "results",
        return_loss: bool = False,
        num_samples: int = -1,
        verbose: bool = True,
    ):
        test_name = test_loader.name
        result_dir = os.path.join(result_dir, test_name)
        os.makedirs(result_dir, exist_ok=True)
        test_loader = test_loader.take(num_samples)

        if return_loss:
            results = self.model.evaluate(test_loader, batch_size=None)
            with open(os.path.join(result_dir, "loss.txt"), "w") as f:
                f.write(str(results))

        list_audio_paths, list_hyps, list_labels = [], [], []
        list_starts, list_ends = [], []
        list_scores = []
        for data in tqdm.tqdm(test_loader, desc="Evaluating"):
            if len(data) == 5:
                audio_paths, (features, features_length,
                              targets), labels, starts, ends = data
            elif len(data) == 3:
                audio_paths, (features, features_length,
                              targets), labels = data
                starts = [0] * features.shape[0]
                ends = [-1] * features.shape[0]
            else:
                (features, features_length, targets), labels = data
                audio_paths = [''] * features.shape[0]
                starts = [0] * features.shape[0]
                ends = [-1] * features.shape[0]

            if self.model.cmvn:
                features = self.model.cmvn(features)

            # Encoder
            mask = tf.sequence_mask(features_length,
                                    maxlen=tf.shape(features)[1])
            mask = tf.expand_dims(mask, axis=1)
            encoder_outputs, encoder_mask = self.model.encoder(features,
                                                               mask,
                                                               training=False)
            encoder_mask = tf.squeeze(encoder_mask, axis=1)
            features_length = tf.math.reduce_sum(
                tf.cast(encoder_mask, dtype=tf.int32),
                axis=1
            )

            hyps, scores = self.decoder.infer(encoder_outputs, features_length)

            if isinstance(labels, tuple):
                labels = labels[0]
            labels = self.text_decoder.decode(labels).numpy()
            labels = [result.decode('utf-8') for result in labels]
            hyps = [result.decode('utf-8') for result in hyps.numpy()]
            scores = [result for result in scores.numpy()]
            if isinstance(audio_paths, tf.Tensor):
                audio_paths = [
                    audio_path.decode('utf-8')
                    for audio_path in audio_paths.numpy()
                ]
            if isinstance(starts, tf.Tensor):
                starts = starts.numpy().tolist()
            if isinstance(ends, tf.Tensor):
                ends = ends.numpy().tolist()

            list_labels.extend(labels)
            list_hyps.extend(hyps)
            list_audio_paths.extend(audio_paths)
            list_starts.extend(starts)
            list_ends.extend(ends)
            list_scores.extend(scores)
            if verbose:
                for hyp, label in zip(hyps, labels):
                    print('pred :', hyp)
                    print('label:', label)
                    print('+' * 5)

        ref_path = os.path.join(result_dir, 'ref.txt')
        hyp_path = os.path.join(result_dir, 'hyp.txt')
        with open(ref_path, 'w', encoding='utf-8') as fref:
            with open(hyp_path, 'w', encoding='utf-8') as fhyp:
                for hyp, label in zip(list_hyps, list_labels):
                    fref.write(label + '\n')
                    fhyp.write(hyp + '\n')

        log_path = os.path.join(result_dir, test_name + '.tsv')
        with open(log_path, 'w', encoding='utf-8') as fout:
            writer = csv.writer(fout, delimiter='\t')
            fout.write('PATH\tSTART\tEND\tDECODED\tSCORE\tTRANSCRIPT\n')
            for audio_path, start, end, hyp, score, label in zip(
                    list_audio_paths, list_starts, list_ends, list_hyps,
                    list_scores, list_labels):
                writer.writerow([audio_path, start, end, hyp, score, label])

    def load_model(self, checkpoint_path: str):
        self.model.load_weights(checkpoint_path).expect_partial()


# class LMTrainer(BaseTrainer):

#     def __init__(
#         self,
#         model: tf.keras.Model,
#         loss: tf.keras.losses.Loss,
#         learning_rate: Union[float, LearningRateSchedule],
#         optim: str = "adam",
#         weight_decay: float = 0.000001,
#         gradient_clipvalue: float = 5.0,
#         num_epochs: int = 100,
#         jit_compile: bool = False,
#         steps_per_execution: int = 1,
#         train_num_samples: int = -1,
#         dev_num_samples: int = -1,
#         tb_log_dir: str = "logs",
#         tb_update_freq: str = "epoch",
#         tb_profile_batch: int = 0,
#         pretrained_model: str = "",
#         checkpoint_path: str = "",
#         ckpt_save_freq: str = "epoch",
#         backup_dir: str = "train_states",
#     ):
#         # Init optimizer
#         if optim == "adam":
#             optimizer = tf.keras.optimizers.Adam(learning_rate,
#                                                  clipvalue=gradient_clipvalue)
#         elif optim == "adamw":
#             optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate,
#                                              weight_decay=weight_decay,
#                                              clipvalue=gradient_clipvalue)
#         elif optim == "lion":
#             optimizer = Lion(learning_rate=learning_rate,
#                              weight_decay=weight_decay,
#                              clipvalue=gradient_clipvalue)
#         else:
#             raise NotImplementedError(f"Optimizer {optim} is not implemented.")

#         self.optimizer = optimizer
#         self.model = model

#         if pretrained_model:
#             self.load_model(pretrained_model)

#         self.model.compile(optimizer=optimizer,
#                            loss=loss,
#                            steps_per_execution=steps_per_execution,
#                            jit_compile=jit_compile)

#         self.num_epochs = num_epochs

#         # Init callbacks
#         tb_callback = tf.keras.callbacks.TensorBoard(
#             log_dir=tb_log_dir,
#             update_freq=tb_update_freq,
#             profile_batch=tb_profile_batch
#         )

#         if not checkpoint_path:
#             checkpoint_path = "lm/checkpoints/ckpt-epoch-{epoch:02d}.ckpt"

#         os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
#         checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#             filepath=checkpoint_path,
#             save_weights_only=True,
#             save_best_only=True,
#             save_freq=ckpt_save_freq
#         )

#         os.makedirs(backup_dir, exist_ok=True)
#         backup_callback = tf.keras.callbacks.BackupAndRestore(backup_dir)
#         self.callbacks = [tb_callback, checkpoint_callback, backup_callback]

#     def train(
#         self,
#         train_loader: tf.data.Dataset,
#         dev_loader: tf.data.Dataset = None
#     ):
#         if self.train_num_samples != -1:
#             train_loader = train_loader.repeat().take(self.train_num_samples)

#         if self.dev_num_samples != -1 and dev_loader is not None:
#             dev_loader = dev_loader.repeat().take(self.dev_num_samples)

#         self.model.fit(train_loader,
#                        epochs=self.num_epochs,
#                        callbacks=self.callbacks,
#                        validation_data=dev_loader)
