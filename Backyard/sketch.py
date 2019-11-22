'''
These are our options:
k-sparse
L1 regularization - Done - Didnt work that much
KL Regularization
Custom optimization function
Check CBOW
Replace 0 with -1
'''

from keras import backend as K
from keras import regularizers
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import Evaluation.Evaluator as dblp_eval
from Common.Utils import crossValidate

fax = './x_sampleset.pkl'
fay = './y_sampleset.pkl'
with open(fax, 'rb') as f:
    x = pkl.load(f)
with open(fay, 'rb') as f:
    y = pkl.load(f)

# Variables
train_ratio = 0.7
validation_ratio = 0.2
epochs = 50
batch_sizes = 8
sp = 0.01
b_val = 3  # Controls the acitvity of the hidden layer nodes
encoding_dim = 5000

# Get the train and test as done in DL assignments.
# Train/Validation/Test version
x_train, x_validate, x_test, ids = crossValidate(x, train_ratio, validation_ratio)
y_train = y[ids[0:int(y.__len__() * train_ratio)]]
y_validate = y[ids[int(y.__len__() * train_ratio):int(y.__len__() * (train_ratio + validation_ratio))]]
y_test = y[ids[int(y.__len__() * (train_ratio + validation_ratio)):]]


input_dim = x_train.shape[1]
output_dim = y_train.shape[1]
print("Input/Output dimensions ", input_dim, output_dim)
# this is our input placeholder
input_img = Input(shape=(input_dim,))
lambda_val = 0.001  # Weight decay , refer : https://stackoverflow.com/questions/44495698/keras-difference-between-kernel-and-activity-regularizers

# Custom Regularizer function
def sparse_reg(activ_matrix):
    p = 0.01
    beta = 3
    p_hat = K.mean(activ_matrix)  # average over the batch samples
    print("p_hat = ", p_hat)
    # KLD = p*(K.log(p)-K.log(p_hat)) + (1-p)*(K.log(1-p)-K.log(1-p_hat))
    KLD = p * (K.log(p / p_hat)) + (1 - p) * (K.log((1 - p) / (1 - p_hat)))
    print("KLD = ", KLD)
    return beta * K.sum(KLD)  # sum over the layer units


encoded = Dense(encoding_dim,
                activation='sigmoid',
                kernel_regularizer=regularizers.l2(lambda_val / 2), activity_regularizer=sparse_reg)(input_img)

decoded = Dense(output_dim,
                activation='sigmoid',
                kernel_regularizer=regularizers.l2(lambda_val / 2), activity_regularizer=sparse_reg)(
    encoded)  # Switch to softmax here?

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='sgd', loss='mse')  # What optimizer to use??

# # For predicting the output of the encoded and decoded layers
# encoder = Model(input_img, encoded)  # map input image to decoded image
#
# encoded_input = Input(shape=(encoding_dim,))
# decoder_layer = autoencoder.layers[-1]  # Gets the last layer
# decoder = Model(encoded_input, decoder_layer(encoded_input))


# Removing normalization since using MSE - later
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

autoencoder.fit(x_train, y_train,
                epochs=epochs,
                batch_size=batch_sizes,
                shuffle=True,
                verbose=2,
                validation_data=(x_validate, y_validate))

print(autoencoder.summary())


evaluation_k_set = np.arange(10, 1100, 100)

# Defining evaluation scores holders for train data
p_at_k_all_train = dblp_eval.init_eval_holder(evaluation_k_set) # all p@k of instances in one fold and one k_evaluation_set
p_at_k_overall_train = dblp_eval.init_eval_holder(evaluation_k_set) # overall p@k of instances in one fold and one k_evaluation_set
r_at_k_all_train = dblp_eval.init_eval_holder(evaluation_k_set) # all r@k of instances in one fold and one k_evaluation_set
r_at_k_overall_train = dblp_eval.init_eval_holder(evaluation_k_set) # overall r@k of instances in one fold and one k_evaluation_set

# Defining evaluation scores holders for test data
p_at_k_all = dblp_eval.init_eval_holder(evaluation_k_set) # all p@k of instances in one fold and one k_evaluation_set
p_at_k_overall = dblp_eval.init_eval_holder(evaluation_k_set) # overall p@k of instances in one fold and one k_evaluation_set
r_at_k_all = dblp_eval.init_eval_holder(evaluation_k_set) # all r@k of instances in one fold and one k_evaluation_set
r_at_k_overall = dblp_eval.init_eval_holder(evaluation_k_set) # overall r@k of instances in one fold and one k_evaluation_set

# @k evaluation process for last train batch data
print("Evaluation on last batch of train data.")
for k in evaluation_k_set:
    # p@k evaluation
    print("Evaluating p@k for top {} records.".format(k))
    p_at_k, p_at_k_array = dblp_eval.p_at_k(autoencoder.predict(x_train), y_train, k=k)
    p_at_k_overall_train[k].append(p_at_k)
    p_at_k_all_train[k].append(p_at_k_array)
    # r@k evaluation
    print("Evaluating r@k for top {} records.".format(k))
    r_at_k, r_at_k_array = dblp_eval.r_at_k(autoencoder.predict(x_train), y_train, k=k)
    r_at_k_overall_train[k].append(r_at_k)
    r_at_k_all_train[k].append(r_at_k_array)

    print("For top {} in Train data:\nP@{}:{}\nR@{}:{}".format(k, k, p_at_k, k, r_at_k))

    # @k evaluation process for test data
    print("Evaluation on test data.")
    for k in evaluation_k_set:
        # p@k evaluation
        print("Evaluating p@k for top {} records.".format(k))
        p_at_k, p_at_k_array = dblp_eval.p_at_k(autoencoder.predict(x_test), y_test, k=k)
        p_at_k_overall[k].append(p_at_k)
        p_at_k_all[k].append(p_at_k_array)
        # r@k evaluation
        print("Evaluating r@k for top {} records.".format(k))
        r_at_k, r_at_k_array = dblp_eval.r_at_k(autoencoder.predict(x_test), y_test, k=k)
        r_at_k_overall[k].append(r_at_k)
        r_at_k_all[k].append(r_at_k_array)

        print("For top {} in Test data:\nP@{}:{}\nR@{}:{}".format(k, k, p_at_k, k, r_at_k))

'''
======================CLASSIFICATION OF TEST DATA=========================================================================
'''
# encode and decode some digits
# note that we take them from the *test* set. This is testing that the digits are correctly classified.
# Why this is requried? Why not send the images directly and the encoding happens internally in the model and we get the output? Explore.
# encoded_imgs = encoder.predict(x_train)
# decoded_imgs = decoder.predict(encoded_imgs)
#
# n = 1  # how many digits we will display
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # display original
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(x_train[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
#     # display reconstruction
#     ax = plt.subplot(2, n, i + 1 + n)
#     plt.imshow(decoded_imgs[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()
