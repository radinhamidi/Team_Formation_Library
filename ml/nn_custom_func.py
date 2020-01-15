import numpy as np
from keras import backend as K


# Custom Regularizer function
def sparse_reg(activ_matrix):
    p = 0.01
    beta = 30
    p_hat = K.mean(activ_matrix)  # average over the batch samples
    print("p_hat = ", p_hat)
    # KLD = p*(K.log(p)-K.log(p_hat)) + (1-p)*(K.log(1-p)-K.log(1-p_hat))
    KLD = p * (K.log(p / p_hat)) + (1 - p) * (K.log((1 - p) / (1 - p_hat)))
    print("KLD = ", KLD)
    return beta * K.sum(KLD)  # sum over the layer units


def batch_generator(iterable, n=10):
    l = len(iterable)
    for ndx in range(0, l, n):
        batch_length = min(ndx + n, l) - ndx
        yield np.asarray([record[0].todense() for record in iterable[ndx:min(ndx + n, l)]]).reshape(batch_length, -1)

def batch_generator_dense(iterable, n=10):
    l = len(iterable)
    for ndx in range(0, l, n):
        batch_length = min(ndx + n, l) - ndx
        yield np.asarray([record for record in iterable[ndx:min(ndx + n, l)]]).reshape(batch_length, -1)