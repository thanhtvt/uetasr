import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class NoamSchedule(LearningRateSchedule):

    def __init__(self,
                 d_model: int,
                 factor: int = 1,
                 warmup_steps: int = 4000,
                 max_lr: float = None):
        super(NoamSchedule, self).__init__()

        self.d_model = d_model
        self.factor = factor
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)
        lr = tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
        lr = lr / tf.constant(self.factor, dtype=tf.float32)
        if self.max_lr is not None:
            return tf.math.minimum(self.max_lr, lr)
        return lr


class WarmupLRSchedule(LearningRateSchedule):

    def __init__(self,
                 factor: int = 1,
                 warmup_steps: int = 4000,
                 accum_steps: int = 1,
                 max_lr: float = None):
        super(WarmupLRSchedule, self).__init__()

        self.factor = factor
        self.max_lr = max_lr
        self.accum_steps = accum_steps
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step // self.accum_steps + int(self.accum_steps > 1),
                       dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)
        lr = (self.warmup_steps**0.5) * tf.math.minimum(arg1, arg2)
        lr = lr / tf.constant(self.factor, dtype=tf.float32)
        if self.max_lr is not None:
            return tf.math.minimum(self.max_lr, lr)
        return lr


class WarmupCosineDecayLRSchedule(LearningRateSchedule):

    def _init_(self,
               init_lr: float,
               max_lr: float,
               warmup_steps: int,
               delay_steps: int,
               accum_steps: int = 1):
        super(WarmupCosineDecayLRSchedule, self).__init__()
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.delay_steps = delay_steps
        self.accum_steps = accum_steps
        self.decay_fn = tf.keras.optimizers.schedules.CosineDecay(
            max_lr,
            warmup_steps * 3 - (warmup_steps + delay_steps),
            alpha=init_lr / max_lr
        )

    def __call__(self, step):
        global_step_float = tf.cast(
            (step + self.accum_steps - 1) // self.accum_steps,
            dtype=tf.float32
        )
        warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
        delay_steps_float = tf.cast(self.delay_steps, tf.float32)

        warmup_value = global_step_float * \
            (self.max_lr - self.init_lr) / warmup_steps_float
        warmup_learning_rate = tf.math.minimum(self.init_lr + warmup_value,
                                               self.max_lr)
        return tf.cond(
            global_step_float < warmup_steps_float + delay_steps_float,
            lambda: warmup_learning_rate,
            lambda: self.decay_fn(tf.cast(global_step_float, tf.int32) -
                                  self.warmup_steps - self.delay_steps
                                  ))
