'''Keras implementation of the k-sparse autoencoder.
'''
from keras import backend as K
from keras.layers import Layer, Lambda
from keras.callbacks import Callback
import numpy as np


class KSparse(Layer):
    '''k-sparse Keras layer.

    # Arguments
        sparsity_levels: np.ndarray, sparsity levels per epoch calculated by `calculate_sparsity_levels`
    '''

    def __init__(self, sparsity_levels, **kwargs):
        self.sparsity_levels = sparsity_levels
        self.k = K.variable(self.sparsity_levels[0], dtype=K.tf.int32)
        self.uses_learning_phase = True
        super().__init__(**kwargs)

    def call(self, inputs, mask=None):
        def sparse():
            kth_smallest = K.tf.contrib.framework.sort(inputs)[..., K.shape(inputs)[-1] - 1 - self.k]
            return inputs * K.cast(K.greater(inputs, kth_smallest[:, None]), K.floatx())

        return K.in_train_phase(sparse, inputs)

    def get_config(self):
        config = {'k': self.k}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class UpdateSparsityLevel(Callback):
    '''Update sparsity level at the beginning of each epoch.
    '''

    def on_epoch_begin(self, epoch, logs={}):
        l = self.model.get_layer('KSparse')
        K.set_value(l.k, l.sparsity_levels[epoch])


def calculate_sparsity_levels(initial_sparsity, final_sparsity, n_epochs):
    '''Calculate sparsity levels per epoch.

    # Arguments
        initial_sparsity: int
        final_sparsity: int
        n_epochs: int
    '''
    return np.hstack((np.linspace(initial_sparsity, final_sparsity, n_epochs // 2, dtype=np.int),
                      np.repeat(final_sparsity, (n_epochs // 2) + 1)))[:n_epochs]