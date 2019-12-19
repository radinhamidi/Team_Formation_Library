import DataAccessLayer.load_dblp_data as dblp
import Evaluation.Evaluator as dblp_eval
from keras.layers import Input, Dense
from keras.models import Model
from keras import backend as K
from sklearn.model_selection import KFold
import numpy as np
import time
from Common.Utils import *
import glob
import pickle as pkl
from keras.utils import plot_model
from contextlib import redirect_stdout
from keras import regularizers

######## Definitions
dataset_name = 'DBLP'
method_name = 'KL'
seed = 7
epochs_in_batch = 10
epochs_overall = 20
back_propagation_batch_size = 64
k_fold = 10
k_max = 50
evaluation_k_set = np.arange(1, k_max+1, 1)
training_batch_size = 1000
min_skill_size = 0
min_member_size = 0
encoding_dim = 500  # encoded size
########

# fix random seed for reproducibility
np.random.seed(seed)
print(K.tensorflow_backend._get_available_gpus())

if dblp.preprocessed_dataset_exist():
    dataset = dblp.load_preprocessed_dataset()
else:
    if not dblp.ae_data_exist(file_path='../Dataset/ae_dataset.pkl'):
        dblp.extract_data(filter_journals=True, skill_size_filter=min_skill_size, member_size_filter=min_member_size)
    if not dblp.preprocessed_dataset_exist() and not dblp.Train_Test_indices_exist():
        dblp.dataset_preprocessing(dblp.load_ae_dataset(file_path='../Dataset/ae_dataset.pkl'), seed=seed)
    dataset = dblp.load_preprocessed_dataset()




# 10-fold Cross Validation
cvscores = []

# Defining evaluation scores holders for train data
r_at_k_all_train = dblp_eval.init_eval_holder(
    evaluation_k_set)  # all r@k of instances in one fold and one k_evaluation_set
r_at_k_overall_train = dblp_eval.init_eval_holder(
    evaluation_k_set)  # overall r@k of instances in one fold and one k_evaluation_set

# Defining evaluation scores holders for test data
r_at_k_all = dblp_eval.init_eval_holder(evaluation_k_set)  # all r@k of instances in one fold and one k_evaluation_set
r_at_k_overall = dblp_eval.init_eval_holder(
    evaluation_k_set)  # overall r@k of instances in one fold and one k_evaluation_set

lambda_val = 0.001  # Weight decay , refer : https://stackoverflow.com/questions/44495698/keras-difference-between-kernel-and-activity-regularizers
# Custom Regularizer function
def sparse_reg(activ_matrix):
    p = 0.2
    beta = 50
    p_hat = K.mean(activ_matrix)  # average over the batch samples
    print("p_hat = ", p_hat)
    # KLD = p*(K.log(p)-K.log(p_hat)) + (1-p)*(K.log(1-p)-K.log(1-p_hat))
    KLD = p * (K.log(p / p_hat)) + (1 - p) * (K.log((1 - p) / (1 - p_hat)))
    print("KLD = ", KLD)
    return beta * K.sum(KLD)  # sum over the layer units


time_str = time.strftime("%Y%m%d-%H%M%S")
train_test_indices = dblp.load_Train_Test_indices()
for fold_counter in range(1,k_fold+1):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    train_index = train_test_indices[fold_counter]['Train']
    test_index = train_test_indices[fold_counter]['Test']
    for sample in dataset:
        id = sample[0]
        if id in train_index:
            x_train.append(sample[1])
            y_train.append(sample[2])
        elif id in test_index:
            x_test.append(sample[1])
            y_test.append(sample[2])
    x_train = np.asarray(x_train).reshape(len(x_train), -1)
    y_train = np.asarray(y_train).reshape(len(y_train), -1)
    x_test = np.asarray(x_test).reshape(len(x_test), -1)
    y_test = np.asarray(y_test).reshape(len(y_test), -1)

    print('Fold number {}'.format(fold_counter))
    print('Dataset Size: {}'.format(len(dataset)))
    print('Train Size: {} Test Size: {}'.format(x_train.__len__(), x_test.__len__()))

    input_dim = x_train[0][0].shape[1]
    output_dim = y_train[0][0].shape[1]
    print("Input/Output Dimensions:  ", input_dim, output_dim)
    # this is our input placeholder
    input_img = Input(shape=(input_dim,))

    encoded = Dense(encoding_dim, activation='sigmoid', kernel_regularizer=regularizers.l2(lambda_val / 2),
                    activity_regularizer=sparse_reg)(input_img)
    decoded = Dense(output_dim, activation='sigmoid', kernel_regularizer=regularizers.l2(lambda_val / 2),
                    activity_regularizer=sparse_reg)(encoded)

    autoencoder = Model(inputs=input_img, outputs=decoded)

    autoencoder.compile(optimizer='adagrad', loss='mean_absolute_error')
    # autoencoder.compile(optimizer='adagrad', loss='cross_entropy')

    # Loading model weights
    load_weights_from_file_q = input('Load weights from file? (y/n)')
    if load_weights_from_file_q.lower() == 'y':
        pick_model_weights(autoencoder, dataset_name=dataset_name)
    # x_train = x_train.astype('float32')
    # x_test = x_test.astype('float32')

    more_train_q = input('Train more? (y/n)')
    if more_train_q.lower() == 'y':
        # Training
        print(
            'Starting with back propagation batch size {} for {} times each during {} overall epochs with training batch size {}. Total epochs={}.'
            .format(back_propagation_batch_size, epochs_in_batch, epochs_overall, training_batch_size,
                    epochs_overall * epochs_in_batch))
        # Training
        for epoch_overall in range(epochs_overall):
            batch_counter = 1
            for x_train_batch, y_train_batch in zip(dblp.batch_generator(x_train, training_batch_size),
                                                    dblp.batch_generator(y_train, training_batch_size)):
                print('Overall Epoch: {} | Training batch {} of fold {}.'.format(epoch_overall + 1, batch_counter,
                                                                                 fold_counter))
                autoencoder.fit(x_train_batch, y_train_batch,
                                epochs=epochs_in_batch,
                                batch_size=back_propagation_batch_size,
                                shuffle=True,
                                verbose=2,
                                validation_data=(
                                np.asarray([x_test_record[0].todense() for x_test_record in x_test]).reshape(x_test.__len__(),
                                                                                                          -1),
                                np.asarray([y_test_record[0].todense() for y_test_record in y_test]).reshape(y_test.__len__(),
                                                                                                          -1)))
                batch_counter += 1
                # Cool down GPU
                # time.sleep(300)

    score = autoencoder.evaluate(
        np.asarray([x_test_record[0].todense() for x_test_record in x_test]).reshape(x_test.__len__(), -1),
        np.asarray([y_test_record[0].todense() for y_test_record in y_test]).reshape(y_test.__len__(), -1),
        verbose=2)
    print('Test loss of fold {}: {}'.format(fold_counter, score))
    cvscores.append(score)

    # @k evaluation process for last train batch data
    print("Evaluation on last batch of train data.")
    for k in evaluation_k_set:
        # r@k evaluation
        print("Evaluating r@k for top {} records in fold {}.".format(k, fold_counter))
        r_at_k, r_at_k_array = dblp_eval.r_at_k(autoencoder.predict(x_train_batch), y_train_batch, k=k)
        r_at_k_overall_train[k].append(r_at_k)
        r_at_k_all_train[k].append(r_at_k_array)

        # print("For top {} in Train data:\nP@{}:{}\nR@{}:{}".format(k, k, p_at_k, k, r_at_k))
        print("For top {} in Train data: R@{}:{}".format(k, k, r_at_k))

    # @k evaluation process for test data
    print("Evaluation on test data.")
    for k in evaluation_k_set:
        # r@k evaluation
        print("Evaluating r@k for top {} records in fold {}.".format(k, fold_counter))
        r_at_k, r_at_k_array = dblp_eval.r_at_k(autoencoder.predict(
            np.asarray([x_test_record[0].todense() for x_test_record in x_test]).reshape(x_test.__len__(), -1)),
            np.asarray([y_test_record[0].todense() for y_test_record in y_test]).reshape(y_test.__len__(), -1), k=k)
        r_at_k_overall[k].append(r_at_k)
        r_at_k_all[k].append(r_at_k_array)

        # print("For top {} in Test data:\nP@{}:{}\nR@{}:{}".format(k, k, p_at_k, k, r_at_k))
        print("For top {} in Test data: R@{}:{}".format(k, k, r_at_k))


    # saving model
    # save_model_q = input('Save the models? (y/n)')
    # if save_model_q.lower() == 'y':
    model_json = autoencoder.to_json()

    # model_name = input('Please enter autoencoder model name:')

    with open('../Output/Models/{}_{}_Time{}_Fold{}.json'.format(dataset_name, method_name, time_str, fold_counter), "w") as json_file:
        json_file.write(model_json)

    autoencoder.save_weights(
        "../Output/Models/Weights/{}_{}_Time{}_Fold{}.h5".format(dataset_name, method_name, time_str, fold_counter))

    with open(
            '../Output/Models/{}_{}_Time{}_EncodingDim{}_Fold{}_Loss{}_Epoch{}(overall{}xinner{})_kFold{}_BatchBP{}_BatchTraining{}.txt'
                    .format(dataset_name, method_name, time_str, encoding_dim, fold_counter, int(score * 1000),
                            epochs_in_batch * epochs_overall, epochs_in_batch, epochs_overall,
                            k_fold, back_propagation_batch_size, training_batch_size), 'w') as f:
        with redirect_stdout(f):
            autoencoder.summary()

    # plot_model(autoencoder, '../Output/Models/{}_Time{}_EncodingDim{}_Fold{}_Loss{}_Epoch{}_kFold{}_BatchBP{}_BatchTraining{}.png'
    #            .format(dataset_name, time_str, encoding_dim, fold_counter, int(np.mean(cvscores) * 1000), epoch, k_fold,
    #                    back_propagation_batch_size, training_batch_size))
    # print('Model and its summary and architecture plot are saved.')
    print('Model and its summary are saved.')

    # Deleting model from RAM
    K.clear_session()

    # Saving evaluation data
    dblp_eval.save_record(r_at_k_all_train, '{}_{}_r@k_all_train_Time{}'.format(dataset_name, method_name, time_str))
    dblp_eval.save_record(r_at_k_overall_train, '{}_{}_r@k_train_Time{}'.format(dataset_name, method_name, time_str))

    dblp_eval.save_record(r_at_k_all, '{}_{}_r@k_all_Time{}'.format(dataset_name, method_name, time_str))
    dblp_eval.save_record(r_at_k_overall, '{}_{}_r@k_Time{}'.format(dataset_name, method_name, time_str))

    print('Evaluation records are saved successfully for fold #{}'.format(fold_counter))

    fold_counter += 1
    break

print('Loss for each fold: {}'.format(cvscores))

compare_submit = input('Submit for compare? (y/n)')
if compare_submit.lower() == 'y':
    with open('../Backyard/KL_r_at_k_50.pkl', 'wb') as f:
        pkl.dump(r_at_k_overall, f)