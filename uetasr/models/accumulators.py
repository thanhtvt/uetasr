import tensorflow as tf
from ..utils.adaptive_gradient_clip import adaptive_clip_grad


# https://stackoverflow.com/a/66524901
# https://keras.io/guides/customizing_what_happens_in_fit/
# https://github.com/andreped/GradientAccumulator
# adding this avoids needing to use custom_objects when loading model
@tf.keras.utils.register_keras_serializable()
class GradientAccumulateModel(tf.keras.Model):

    def __init__(self,
                 accum_steps: int = 1,
                 mixed_precision: bool = False,
                 use_agc: bool = False,
                 clip_factor: float = 0.01,
                 eps: float = 1e-3,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.accum_steps = tf.constant(accum_steps,
                                       dtype=tf.int32,
                                       name="accum_steps")
        self.accum_step_counter = tf.Variable(
            0,
            dtype=tf.int32,
            trainable=False,
            name="accum_counter",
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
        )
        self.first_call = True
        self.gradient_accumulation = None
        self.reinit_grad_accum()
        self.mixed_precision = mixed_precision
        self.use_agc = use_agc
        self.clip_factor = clip_factor
        self.eps = eps

    def train_step(self, data):
        # need to reinit accumulator for models subclassed from tf.keras.Model
        if self.first_call:
            self.reinit_grad_accum()
            self.first_call = False

        self.accum_step_counter.assign_add(1)

        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        # NOTE that x and y are lists of inputs and outputs,
        # hence this wrapper supports multi-input-output models
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        # Gradient Tape
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # forward pass

            # Compute the loss value.
            # The loss function is configured in `compile()`.
            loss = self.compiled_loss(
                y,
                y_pred,
                sample_weight=sample_weight,
                regularization_losses=self.losses,
            )
            loss = loss / tf.cast(
                self.accum_steps,
                tf.float32)  # MEAN reduction here IMPORTANT! Don't use SUM!

            # scale loss if mixed precision is enabled
            if self.mixed_precision:
                loss = self.optimizer.get_scaled_loss(loss)

        # Calculate batch gradients -> these are scaled gradients if
        # mixed precision is enabled
        gradients = tape.gradient(
            loss,
            self.trainable_variables,
            unconnected_gradients=tf.UnconnectedGradients.ZERO)

        # scale gradients if mixed precision is enabled
        if self.mixed_precision:
            gradients = self.optimizer.get_unscaled_gradients(gradients)

        # apply adaptive gradient clipping
        # -> should be AFTER unscaling gradients
        if self.use_agc:
            gradients = adaptive_clip_grad(self.trainable_variables,
                                           gradients,
                                           clip_factor=self.clip_factor,
                                           eps=self.eps)

        # Accumulate batch gradients
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(gradients[i],
                                                     read_value=False)

        # If accum_step_counter reach the accum_steps
        # then we apply accumulated gradients to update the variables
        # otherwise do nothing
        tf.cond(tf.equal(self.accum_step_counter, self.accum_steps),
                true_fn=self.apply_accu_gradients,
                false_fn=lambda: None)

        # update metrics
        self.compiled_metrics.update_state(y,
                                           y_pred,
                                           sample_weight=sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def apply_accu_gradients(self):
        # apply accumulated gradients
        self.optimizer.apply_gradients(
            zip(self.gradient_accumulation, self.trainable_variables))

        # reset
        self.accum_step_counter.assign(0)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(tf.zeros_like(
                self.trainable_variables[i], dtype=tf.float32),
                                                 read_value=False)

    def reinit_grad_accum(self):
        # reinitialize gradient accumulator
        self.gradient_accumulation = [
            tf.Variable(
                tf.zeros_like(v, dtype=tf.float32),
                trainable=False,
                name="accum_" + str(i),
                synchronization=tf.VariableSynchronization.ON_READ,
                aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
            ) for i, v in enumerate(self.trainable_variables)
        ]
