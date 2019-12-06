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
method_name = 'T2V'
seed = 7
epochs = 300
back_propagation_batch_size = 64
k_fold = 10
k_max = 50
evaluation_k_set = np.arange(1, k_max+1, 1)
training_batch_size = 2000
data_size_limit = 1000
train_ratio = 0.7
validation_ratio = 0.15
min_skill_size = 0
min_member_size = 0
encoding_dim = 500  # encoded size
########

# fix random seed for reproducibility
np.random.seed(seed)
print(K.tensorflow_backend._get_available_gpus())

t2v_model = Team2Vec()
t2v_model = load_T2V_model(t2v_model)
embedding_dim = t2v_model.model.vector_size

if dblp.ae_data_exist(file_path='../Dataset/ae_t2v_dim{}_dataset.pkl'.format(embedding_dim)):
    dataset = dblp.load_ae_dataset(file_path='../Dataset/ae_t2v_dim{}_dataset.pkl'.format(embedding_dim))
else:
    if not dblp.ae_data_exist(file_path='../Dataset/ae_dataset.pkl'):
        dblp.extract_data(filter_journals=True, skill_size_filter=min_skill_size, member_size_filter=min_member_size)
        # extract_data(filter_journals=True, size_limit=data_size_limit)
    # t2v_model = Team2Vec()
    # t2v_model = load_T2V_model(t2v_model)
    raw_dataset = dblp.load_ae_dataset(file_path='../Dataset/ae_dataset.pkl')
    dblp.nn_t2v_dataset_generator(t2v_model, raw_dataset, '../Dataset/ae_t2v_dim{}_dataset.pkl'.format(embedding_dim))
    del raw_dataset
    # del t2v_model
    dataset = dblp.load_ae_dataset(file_path='../Dataset/ae_t2v_dim{}_dataset.pkl'.format(embedding_dim))

ids = []
x = []
y = []
for record in dataset:
    ids.append(record[0])
    x.append(record[1])
    y.append(record[2])
del dataset

y_sparse = []
raw_dataset = dblp.load_ae_dataset(file_path='../Dataset/ae_dataset.pkl')
for record in raw_dataset:
    y_sparse.append(record[2])
del raw_dataset

x = np.asarray(x).reshape(x.__len__(), -1)
y = np.asarray(y).reshape(y.__len__(), -1)
y_sparse = np.asarray(y_sparse).reshape(y_sparse.__len__(), -1)

# Train/Validation/Test version
# x_train, x_validate, x_test, ids = crossValidate(x, train_ratio, validation_ratio)
# y_train = y[ids[0:int(y.__len__() * train_ratio)]]
# y_validate = y[ids[int(y.__len__() * train_ratio):int(y.__len__() * (train_ratio + validation_ratio))]]
# y_test = y[ids[int(y.__len__() * (train_ratio + validation_ratio)):]]

# 10-fold Cross Validation
cv = KFold(n_splits=k_fold, random_state=seed, shuffle=True)
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
    p = 0.01
    beta = 3
    p_hat = K.mean(activ_matrix)  # average over the batch samples
    print("p_hat = ", p_hat)
    # KLD = p*(K.log(p)-K.log(p_hat)) + (1-p)*(K.log(1-p)-K.log(1-p_hat))
    KLD = p * (K.log(p / p_hat)) + (1 - p) * (K.log((1 - p) / (1 - p_hat)))
    print("KLD = ", KLD)
    return beta * K.sum(KLD)  # sum over the layer units


fold_counter = 1
time_str = time.strftime("%Y%m%d-%H%M%S")
for train_index, test_index in cv.split(x):
    print('Fold number {}'.format(fold_counter))
    x_train, x_test, y_train, y_test = x[train_index], x[test_index], y[train_index], y[test_index]
    print('Train Size: {} Test Size: {}'.format(x_train.__len__(), x_test.__len__()))

    input_dim = x_train[0].shape[0]
    output_dim = y_train[0].shape[0]
    print("Input/Output Dimensions:  ", input_dim, output_dim)
    # this is our input placeholder
    input_img = Input(shape=(input_dim,))

    encoded = Dense(encoding_dim, activation='sigmoid', kernel_regularizer=regularizers.l2(lambda_val / 2),
                    activity_regularizer=sparse_reg)(input_img)
    decoded = Dense(output_dim, activation='sigmoid', kernel_regularizer=regularizers.l2(lambda_val / 2),
                    activity_regularizer=sparse_reg)(encoded)

    autoencoder = Model(inputs=input_img, outputs=decoded)

    autoencoder.compile(optimizer='adagrad', loss='mse')

    # Loading model weights
    load_weights_from_file_q = input('Load weights from file? (y/n)')
    if load_weights_from_file_q.lower() == 'y':
        pick_model_weights(autoencoder, dataset_name=dataset_name)

    # x_train = x_train.astype('float32')
    # x_test = x_test.astype('float32')

    more_train_q = input('Train more? (y/n)')
    if more_train_q.lower() == 'y':
        # Training
        autoencoder.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=back_propagation_batch_size,
                        shuffle=True,
                        verbose=2,
                        validation_data=(x_test, y_test))
    # Cool down GPU
    # time.sleep(300)

    score = autoencoder.evaluate(x_test, y_test)
    print('Test loss of fold {}: {}'.format(fold_counter, score))
    cvscores.append(score)

    # Team mode evaluation
    # y_train_pred = [[int(candidate[0]) for candidate in t2v_model.get_team_most_similar_by_vector(record, k_max)]
    #           for record in autoencoder.predict(x_train)]
    # y_train_pred = dblp.get_memebrID_by_teamID(y_train_pred)
    #
    # y_test_pred = [[int(candidate[0]) for candidate in t2v_model.get_team_most_similar_by_vector(record, k_max)]
    #           for record in autoencoder.predict(x_test)]
    # y_test_pred = dblp.get_memebrID_by_teamID(y_test_pred)


    # Member mode evaluation
    y_train_pred = [[int(candidate[0]) for candidate in t2v_model.get_member_most_similar_by_vector(record, k_max)]
              for record in autoencoder.predict(x_train)]
    y_test_pred = [[int(candidate[0]) for candidate in t2v_model.get_member_most_similar_by_vector(record, k_max)]
              for record in autoencoder.predict(x_test)]

    # @k evaluation process for last train batch data
    print("Evaluation on last batch of train data.")
    for k in evaluation_k_set:
        print("Evaluating r@k for top {} records in fold {}.".format(k, fold_counter))
        r_at_k, r_at_k_array = dblp_eval.r_at_k_t2v(y_train_pred, y_sparse[train_index], k=k)
        r_at_k_overall_train[k].append(r_at_k)
        r_at_k_all_train[k].append(r_at_k_array)
        print("For top {} in Train data: R@{}:{}".format(k, k, r_at_k))

    # @k evaluation process for test data
    print("Evaluation on test data.")
    for k in evaluation_k_set:
        # r@k evaluation
        print("Evaluating r@k for top {} records in fold {}.".format(k, fold_counter))
        r_at_k, r_at_k_array = dblp_eval.r_at_k_t2v(y_test_pred, y_sparse[test_index], k=k)
        r_at_k_overall[k].append(r_at_k)
        r_at_k_all[k].append(r_at_k_array)
        print("For top {} in Test data: R@{}:{}".format(k, k, r_at_k))


    # for test_instance in x_test:
    #     result = autoencoder.predict(test_instance)

    # saving model
    save_model_q = input('Save the models? (y/n)')
    if save_model_q.lower() == 'y':
        model_json = autoencoder.to_json()

        # model_name = input('Please enter autoencoder model name:')

        with open('../Output/Models/{}_{}_Time{}_Fold{}.json'.format(dataset_name, method_name, time_str, fold_counter), "w") as json_file:
            json_file.write(model_json)

        autoencoder.save_weights(
            "../Output/Models/Weights/{}_{}_Time{}_Fold{}.h5".format(dataset_name, method_name, time_str, fold_counter))

        with open(
                '../Output/Models/{}_{}_Time{}_EncodingDim{}_Fold{}_Loss{}_Epoch{}_kFold{}_BatchBP{}.txt'
                        .format(dataset_name, method_name, time_str, encoding_dim, fold_counter, int(score * 1000),
                                epochs, k_fold, back_propagation_batch_size), 'w') as f:
            with redirect_stdout(f):
                autoencoder.summary()

        # plot_model(autoencoder, '../Output/Models/{}_Time{}_EncodingDim{}_Fold{}_Loss{}_Epoch{}_kFold{}_BatchBP{}_BatchTraining{}.png'
        #            .format(dataset_name, time_str, encoding_dim, fold_counter, int(np.mean(cvscores) * 1000), epoch, k_fold,
        #                    back_propagation_batch_size, training_batch_size))
        # print('Model and its summary and architecture plot are saved.')
        print('Model and its summary are saved.')

    # Deleting model from RAM
    # K.clear_session()

    # Saving evaluation data
    dblp_eval.save_record(r_at_k_all_train, '{}_{}_r@k_all_train_Time{}'.format(dataset_name, method_name, time_str))
    dblp_eval.save_record(r_at_k_overall_train, '{}_{}_r@k_train_Time{}'.format(dataset_name, method_name, time_str))

    dblp_eval.save_record(r_at_k_all, '{}_{}_r@k_all_Time{}'.format(dataset_name, method_name, time_str))
    dblp_eval.save_record(r_at_k_overall, '{}_{}_r@k_Time{}'.format(dataset_name, method_name, time_str))

    print('Evaluation records are saved successfully for fold #{}'.format(fold_counter))

    fold_counter += 1
    break

print('Loss for each fold: {}'.format(cvscores))


with open('../Backyard/T2V_dim{}_member_r_at_k_50.pkl'.format(embedding_dim), 'wb') as f:
    pkl.dump(r_at_k_overall, f)


# with open('../Backyard/T2V_dim{}_team_r_at_k_50.pkl'.format(embedding_dim), 'wb') as f:
#     pkl.dump(r_at_k_overall, f)
