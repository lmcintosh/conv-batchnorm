"""Normalization layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.engine import Layer, InputSpec
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import backend as K
from keras.legacy import interfaces

__all__ = ['ConvBatchNormalization']


class ConvBatchNormalization(Layer):
    """Convolutional batch normalization layer.

    Modification of batch normalization (Ioffe and Szegedy, 2014) that
    performs local spatial normalization while learning a single scale
    parameter across space for each channel.

    Normalize the activations of the previous layer at each batch,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.

    # Arguments
        normalization_axis: Iterable of integers, the axis (or axes)
            that we independently normalize. In vanilla batch norm we typically
            collapse batch and spatial dimensions, and in which case
            normalization_axis=[1] if `data_format="channels_first"`.
            If we want local normalization, normalization_axis=[1, 2, 3]
            performs normalization independently across space, computing
            statistics just over the batch dimension.
        parameter_axis: Iterable of integers, the axis (or axes)
            that we learn independent parameters for centering and scaling after
            normalization. This only applies if scale=True or center=True.
            In vanilla batch norm, this is typically the features axis.
            For instance, after a `Conv2D` layer with `data_format="channels_first"`,
            set `parameter_axis=1` in `ConvBatchNormalization`.
        momentum: Momentum for the moving mean and the moving variance.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        moving_mean_initializer: Initializer for the moving mean.
        moving_variance_initializer: Initializer for the moving variance.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.

    # References
        - [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
    """

    @interfaces.legacy_batchnorm_support
    def __init__(self,
                 normalization_axis=-1,
                 parameter_axis=-1,
                 momentum=0.99,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(ConvBatchNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.normalization_axis = normalization_axis
        self.parameter_axis = parameter_axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = initializers.get(moving_variance_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        parameter_dim = [input_shape[axis] for axis in self.parameter_axis]
        normalization_dim = [input_shape[axis] for axis in self.normalization_axis]

        if parameter_dim is None:
            raise ValueError('Parameter axis ' + str(self.parameter_axis) + 
                            ' of input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        if normalization_dim is None:
            raise ValueError('Normalization axis ' + str(self.normalization_axis) +
                             ' of input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        # Input_spec provides other keras layers what shape to expect,
        # as well as a dictionary mapping integer axes to a specific
        # dimension value.
        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={axis: dim for axis, dim in zip(
                                            self.parameter_axis, parameter_dim)})
        parameter_shape = (*parameter_dim,)
        print('Parameter shape is {}.'.format(parameter_shape))
        normalization_shape = (*normalization_dim,)
        print('Normalization shape is {}.'.format(normalization_shape))

        if self.scale:
            self.gamma = self.add_weight(shape=parameter_shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
            self.broadcast_gamma = K.expand_dims(K.expand_dims(
                        self.gamma)) * K.ones(normalization_shape)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=parameter_shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
            self.broadcast_beta = K.expand_dims(K.expand_dims(
                        self.beta)) * K.ones(normalization_shape)
        else:
            self.beta = None

        self.moving_mean = self.add_weight(
            shape=normalization_shape,
            name='moving_mean',
            initializer=self.moving_mean_initializer,
            trainable=False)
        self.moving_variance = self.add_weight(
            shape=normalization_shape,
            name='moving_variance',
            initializer=self.moving_variance_initializer,
            trainable=False)
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)

        # Prepare broadcasting shape.
        ndim = len(input_shape)
        reduction_axes = [ax for ax in range(len(input_shape)) if ax not in self.normalization_axis]
        broadcast_shape = [1] * len(input_shape)
        for axis in self.normalization_axis:
            broadcast_shape[axis] = input_shape[axis]

        # Determines whether broadcasting is needed.
        needs_broadcasting = (sorted(
                    reduction_axes) != list(range(ndim))[:-1])

        def normalize_inference():
            if needs_broadcasting:
                # In this case we must explicitly broadcast all parameters.
                broadcast_moving_mean = K.reshape(self.moving_mean,
                                                  broadcast_shape)
                broadcast_moving_variance = K.reshape(self.moving_variance,
                                                      broadcast_shape)
                if self.center:
                    broadcast_beta = K.reshape(self.broadcast_beta, broadcast_shape)
                else:
                    broadcast_beta = None
                if self.scale:
                    broadcast_gamma = K.reshape(self.broadcast_gamma,
                                                broadcast_shape)
                else:
                    broadcast_gamma = None
                return K.batch_normalization(
                    inputs,
                    broadcast_moving_mean,
                    broadcast_moving_variance,
                    broadcast_beta,
                    broadcast_gamma,
                    epsilon=self.epsilon)
            else:
                return K.batch_normalization(
                    inputs,
                    self.moving_mean,
                    self.moving_variance,
                    self.broadcast_beta,
                    self.broadcast_gamma,
                    epsilon=self.epsilon)


        # If the learning phase is *static* and set to inference:
        if training in {0, False}:
            return normalize_inference()

        # If the learning is either dynamic, or set to training:
        normed_training, mean, variance = K.normalize_batch_in_training(
            inputs, self.broadcast_gamma, self.broadcast_beta, reduction_axes,
            epsilon=self.epsilon)

        if K.backend() != 'cntk':
            sample_size = K.prod([K.shape(inputs)[axis]
                                  for axis in reduction_axes])
            sample_size = K.cast(sample_size, dtype=K.dtype(inputs))

            # sample variance - unbiased estimator of population variance
            variance *= sample_size / (sample_size - (1.0 + self.epsilon))

        self.add_update([K.moving_average_update(self.moving_mean,
                                                 mean,
                                                 self.momentum),
                         K.moving_average_update(self.moving_variance,
                                                 variance,
                                                 self.momentum)],
                        inputs)

        # Pick the normalized form corresponding to the training phase.
        return K.in_train_phase(normed_training,
                                normalize_inference,
                                training=training)

    def get_config(self):
        config = {
            'normalization_axis': self.normalization_axis,
            'parameter_axis': self.parameter_axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'moving_mean_initializer': initializers.serialize(self.moving_mean_initializer),
            'moving_variance_initializer': initializers.serialize(self.moving_variance_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(ConvBatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
