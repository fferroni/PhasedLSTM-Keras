from __future__ import absolute_import
import numpy as np

from keras import backend as K
from keras import activations, initializations, regularizers
from keras.engine import Layer, InputSpec
from keras.layers.recurrent import Recurrent

class PhasedLSTM(Recurrent):
    '''
    Phased LSTM implementation in Keras. Working only for Theano due to TF limitation of K.switch
    See pull request https://github.com/fchollet/keras/pull/4519
    
    # Arguments
        output_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        forget_bias_init: initialization function for the bias of the forget gate.
            [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
            recommend initializing with ones.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        inner_activation: activation function for the inner cells.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.
        alpha: float between 0 and 1. Leak fraction of time gate.
    # References
    Daniel Neil, Michael Pfeiffer, Shih-Chii Liu,
    Phased LSTM: Accelerating Recurrent Network Training for Long or Event-based Sequences
    https://arxiv.org/abs/1610.09513
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., alpha=0.001, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W, self.dropout_U = dropout_W, dropout_U
        self.alpha = alpha

        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True
        super(PhasedLSTM, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[2]

        if self.stateful:
            self.reset_states()
        else:
            # initial states: 3 all-zero tensors of shape (output_dim)
            # h,c,t
            self.states = [None, None, None]

        self.W = self.init((self.input_dim, 4 * self.output_dim),
                           name='{}_W'.format(self.name))
        self.U = self.inner_init((self.output_dim, 4 * self.output_dim),
                                 name='{}_U'.format(self.name))

        self.b = K.variable(np.hstack((np.zeros(self.output_dim),
                                       K.get_value(self.forget_bias_init((self.output_dim,))),
                                       np.zeros(self.output_dim),
                                       np.zeros(self.output_dim))),
                            name='{}_b'.format(self.name))

        # all three variables (period, phase and r_on) are learnable
        self.timegate = K.variable(np.vstack((np.random.uniform(10, 100, self.output_dim),
                                              np.random.uniform(0, 1000, self.output_dim),
                                              np.zeros(self.output_dim)+0.05)),
                                   name='{}_tgate'.format(self.name))

        self.trainable_weights = [self.W, self.U, self.b, self.timegate]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(self.U)
            self.regularizers.append(self.U_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim)),
                           K.zeros((input_shape[0], self.output_dim))]

    def preprocess_input(self, x):
        return x

    def step(self, x, states):
        h_tm1 = states[0]
        c_tm1 = states[1]
        t_tm1 = states[2]
        B_U = states[3]
        B_W = states[4]
        
        # time related variables, simply add +1 to t for now...starting from 0
        # need to find better way if asynchronous/irregular time input is desired
        # such as slicing input where first index is time and using that instead.
        t = t_tm1 + 1
        self.timegate = K.abs(self.timegate)
        period = self.timegate[0]
        shift = self.timegate[1]
        on_mid = self.timegate[2] * 0.5 * period
        on_end = self.timegate[2] * period
        
        # double mod necessary to make behaviour consistent in Theano and Tensorflow
        phi = ((((t - shift) % period) + period) % period) / period 
        is_up = K.lesser_equal(phi, on_mid)
        is_down = K.greater(phi, on_mid) & K.lesser_equal(phi, on_end)
        k = K.switch(is_up, phi/on_mid, K.switch(is_down, (on_end-phi)/on_mid, self.alpha*phi))
        
        # standard LSTM calculations, sharing dropout B_U and B_W for i,f,c,o to speed things up.
        # add CPU/GPU/MEM options later
        z = K.dot(x * B_W[0], self.W) + K.dot(h_tm1 * B_U[0], self.U) + self.b
        
        z0 = z[:, :self.output_dim]
        z1 = z[:, self.output_dim: 2 * self.output_dim]
        z2 = z[:, 2 * self.output_dim: 3 * self.output_dim]
        z3 = z[:, 3 * self.output_dim:]

        i = self.inner_activation(z0)
        f = self.inner_activation(z1)
        # intermediate cell update
        c_hat = f * c_tm1 + i * self.activation(z2)
        c = k * c_hat + (1 - k) * c_tm1
        o = self.inner_activation(z3)
        # intermediate hidden update
        h_hat = o * self.activation(c_hat)
        h = k * h_hat + (1 - k) * h_tm1
        return h, [h, c, t]

    def get_constants(self, x):
        constants = []
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.output_dim))
            B_U = [K.in_train_phase(K.dropout(ones, self.dropout_U), ones) for _ in range(4)]
            constants.append(B_U)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        if 0 < self.dropout_W < 1:
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, int(input_dim)))
            B_W = [K.in_train_phase(K.dropout(ones, self.dropout_W), ones) for _ in range(4)]
            constants.append(B_W)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        #add 
        return constants

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'alpha': self.alpha,
                  'init': self.init.__name__,
                  'inner_init': self.inner_init.__name__,
                  'forget_bias_init': self.forget_bias_init.__name__,
                  'activation': self.activation.__name__,
                  'inner_activation': self.inner_activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'dropout_W': self.dropout_W,
                  'dropout_U': self.dropout_U}
        base_config = super(PhasedLSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
