import tensorflow as tf


class RMSLayerNormalization(tf.keras.layers.Layer):
    """
    Root Mean Square Layer Normalization

    Paper: https://arxiv.org/pdf/1910.07467.pdf
    Implementation: https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_tensorflow.py
    """

    def __init__(self, epsilon=1e-8, p=-1.0, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.bias = bias
        self.p = p

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.ones_initializer(),
            trainable=True
        )
        if self.bias:
            self.offset = self.add_weight(
                name='offset',
                shape=input_shape[-1:],
                initializer=tf.zeros_initializer(),
                trainable=True
            )
        super().build(input_shape)

    def call(self, x):
        if self.p < 0 or self.p > 1:
            ms = tf.reduce_mean(x ** 2, axis=-1, keepdims=True)
        else:
            layer_size = tf.shape(x)[-1]
            partial_size = tf.cast(tf.cast(layer_size, tf.float32) * self.p,
                                   dtype=tf.int32)
            partial_x, _ = tf.split(
                x, [partial_size, layer_size - partial_size], axis=-1
            )
            ms = tf.reduce_mean(partial_x ** 2, axis=-1, keepdims=True)

        norm_x = self.scale * x * tf.math.rsqrt(ms + self.epsilon)
        if self.bias:
            return norm_x + self.offset
        return norm_x

    def get_config(self):
        config = super(RMSLayerNormalization, self).get_config()
        config.update({
            'epsilon': self.epsilon,
            'p': self.p,
            'bias': self.bias
        })
        return config


@tf.keras.utils.register_keras_serializable()
class AccumBatchNormalization(tf.keras.layers.Layer):
    """ Custom Batch Normaliztion layer with gradient accumulation support.
    Code from: https://github.com/andreped/GradientAccumulator
    """
    def __init__(
        self,
        accum_steps: int = 1,
        momentum: float = 0.99,
        epsilon:float = 1e-3,
        trainable:bool = True,
        **kwargs
    ):
        """ Construct the AccumBatchNormalization layer.

        Args:
            accum_steps (int): Update gradient in every accumulation steps.
            momentum (float): Momentum used in variable update.
            epsilon (float): Small value to aid numerical stability.
            trainable (bool): Whether layer should be updated during training.
                              Different from training/inference mode.
        """
        self.accum_steps = accum_steps
        self.accum_steps_tf = tf.constant(accum_steps,
                                          dtype=tf.int32,
                                          name="accum_steps")
        self.momentum = momentum
        self.epsilon = epsilon
        self.trainable = trainable
        self.accum_step_counter = tf.Variable(
            0, dtype=tf.int32, trainable=False, name="accum_counter",
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
        )
        super().__init__(**kwargs)

    def build(self, input_shape):
        """Builds layer and variables.
        
        Args:
            input_shape: input feature map size.
        """
        self.param_shape = input_shape[-1]

        self.beta = self.add_weight(
            shape=(self.param_shape),
            dtype=self.dtype,
            initializer="zeros",
            trainable=True,
            name="beta",
            experimental_autocast=False,
        )

        self.gamma = self.add_weight(
            shape=(self.param_shape),
            dtype=self.dtype,
            initializer="ones",
            trainable=True,
            name="gamma",
            experimental_autocast=False,
        )

        self.moving_mean = self.add_weight(
            shape=(self.param_shape),
            dtype=self.dtype,
            initializer="zeros",
            trainable=False,
            name="moving_mean",
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.VariableAggregation.MEAN,
            experimental_autocast=False,
        )

        self.moving_variance = self.add_weight(
            shape=(self.param_shape),
            dtype=self.dtype,
            initializer="ones",
            trainable=False,
            name="moving_variance",
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.VariableAggregation.MEAN,
            experimental_autocast=False,
        )

        self.accum_mean = self.add_weight(
            shape=(self.param_shape),
            dtype=self.dtype,
            initializer="zeros",
            trainable=False,
            name="accum_mean",
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.VariableAggregation.MEAN,
            experimental_autocast=False,
        )

        self.accum_variance = self.add_weight(
            shape=(self.param_shape),
            dtype=self.dtype,
            initializer="zeros",  # this should be "zeros" as we use it for accumulation
            trainable=False,
            name="accum_variance",
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.VariableAggregation.MEAN,
            experimental_autocast=False,
        )

    def get_moving_average(self, statistic, new_value):
        """Returns the moving average given a statistic and current estimate.
        
        Args:
            statistic: summary statistic e.g. average across for single feature over multiple samples
            new_value: statistic of single feature for single forward step.
        Returns:
            Updated statistic.
        """
        decay = tf.convert_to_tensor(1.0 - self.momentum, name="decay")
        if decay.dtype != statistic.dtype.base_dtype:
            decay = tf.cast(decay, statistic.dtype.base_dtype)
        delta = (statistic - tf.cast(new_value, statistic.dtype)) * decay
        return statistic.assign_sub(delta)
    
    def update_variables(self, mean, var):
        """Updates the batch normalization variables.
        
        Args:
            mean: average for single feature
            var: variance for single feature
        """
        self.moving_mean.assign(self.get_moving_average(self.moving_mean, mean))
        self.moving_variance.assign(self.get_moving_average(self.moving_variance, var))

        self.reset_accum()
    
    def reset_accum(self):
        """Resets accumulator slots."""
        self.accum_mean.assign(tf.zeros_like(self.accum_mean))
        self.accum_variance.assign(tf.zeros_like(self.accum_variance))

        self.accum_step_counter.assign(0)

    def call(self, inputs, training=None, mask=None):
        """Performs the batch normalization step.
        
        Args:
            inputs: input feature map to apply batch normalization across.
            training: whether layer should be in training mode or not.
            mask: whether to calculate statistics within masked region of feature map.
        Returns:
            Normalized feature map.
        """
        self.inputs_dtype = inputs.dtype.base_dtype
        if self.inputs_dtype in (tf.float16, tf.bfloat16):
            # Do all math in float32 if given 16-bit inputs for numeric
            # stability. In particular, it's very easy for variance to overflow
            # in float16 and for safety we also choose to cast bfloat16 to
            # float32.
            inputs = tf.cast(inputs, self.dtype)

        if training:
            assert len(inputs.shape) in (2, 4)
            if len(inputs.shape) > 2:
                axes = [0, 1, 2]
            else:
                axes = [0]
            
            # step accum count
            self.accum_step_counter.assign_add(1)
            
            # get batch norm statistics
            mean, var = tf.nn.moments(inputs, axes=axes, keepdims=False)

            # scale mean and variance to produce mean later
            mean_scaled = mean / tf.cast(self.accum_steps_tf, mean.dtype)
            var_scaled = var / tf.cast(self.accum_steps_tf, var.dtype)
            
            # accumulate statistics
            self.accum_mean.assign_add(mean_scaled)
            self.accum_variance.assign_add(var_scaled)

            # only update variables after n accumulation steps
            tf.cond(
                tf.equal(self.accum_step_counter, self.accum_steps_tf),
                true_fn=lambda: self.update_variables(self.accum_mean, self.accum_variance),
                false_fn=lambda: None
            )
        else:
            mean, var = self.moving_mean, self.moving_variance

        scale = self.gamma
        offset = self.beta
        
        inv = tf.math.rsqrt(var + self.epsilon)
        if scale is not None:
            inv *= scale
        
        outputs =  inputs * tf.cast(inv, inputs.dtype) + \
            tf.cast(offset - mean * inv if offset is not None else -mean * inv, inputs.dtype)
        
        # need to convert back to float16 after applying batch norm
        if self.inputs_dtype in (tf.float16, tf.bfloat16):
            outputs = tf.cast(outputs, self.dtype)

        return outputs
    
    @property
    def trainable(self):
        """Returns whether layer is trainable.
        
        Returns:
            trainable boolean state.
        """
        return self._trainable

    @trainable.setter
    def trainable(self, value: bool):
        """Sets trainable variable.
        
        Args:
            value: which boolean state to change variable to.
        """
        self._trainable = value
    
    @property
    def _param_dtype(self):
        """Raise parameters of fp16 batch norm to fp32
        
        Returns:
            dtype of params.
        """
        if self.dtype == tf.float16 or self.dtype == tf.bfloat16:
            return tf.float32
        else:
            return self.dtype or tf.float32
    
    def get_config(self):
        """Returns configurations as dict.
        
        Returns:
            Configuration file.
        """
        config = {
            'accum_steps': self.accum_steps,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'trainable': self.trainable,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
