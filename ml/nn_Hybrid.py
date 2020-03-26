from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.layers import Lambda
from keras.losses import mse, binary_crossentropy, mae, kld, categorical_crossentropy
import time
import pickle as pkl
import csv
from keras.callbacks import EarlyStopping
from contextlib import redirect_stdout
from keras.layers import Input, Dense, concatenate
# from keras.layers.merge import Concatenate
from keras.models import Model
import cmn.utils
import dal.load_dblp_data as dblp
import eval.evaluator as dblp_eval
from cmn.utils import *
from ml.nn_custom_func import *


# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5, min_delta=1)

#running settings
dataset_name = 'DBLP'
method_name = 'Hybrid'

#eval settings
k_fold = 10
k_max = 100 #cut_off for recall
evaluation_k_set = np.arange(1, k_max+1, 1)

#nn settings
epochs_in_batch = 15
epochs_overall = 5
back_propagation_batch_size = 64
training_batch_size = 6000
min_skill_size = 0
min_member_size = 0
latent_dim = 50

print(K.tensorflow_backend._get_available_gpus())


t2v_model = Team2Vec()
t2v_model = load_T2V_model(t2v_model)
embedding_dim = t2v_model.model.vector_size

if dblp.ae_data_exist(file_path='../dataset/ae_t2v_dim{}_tSkill_dataset.pkl'.format(embedding_dim)):
    dataset_t2v = dblp.load_ae_dataset(file_path='../dataset/ae_t2v_dim{}_tSkill_dataset.pkl'.format(embedding_dim))
else:
    if not dblp.ae_data_exist(file_path='../dataset/ae_dataset.pkl'):
        dblp.extract_data(filter_journals=True, skill_size_filter=min_skill_size, member_size_filter=min_member_size)
    if not dblp.preprocessed_dataset_exist() or not dblp.train_test_indices_exist():
        dblp.dataset_preprocessing(dblp.load_ae_dataset(file_path='../dataset/ae_dataset.pkl'), seed=seed, kfolds=k_fold)
    preprocessed_dataset = dblp.load_preprocessed_dataset()

    dblp.nn_t2v_dataset_generator(t2v_model, preprocessed_dataset, output_file_path='../dataset/ae_t2v_dim{}_tSkill_dataset.pkl'.format(embedding_dim), mode='skill')
    del preprocessed_dataset
    dataset_t2v = dblp.load_ae_dataset(file_path='../dataset/ae_t2v_dim{}_tSkill_dataset.pkl'.format(embedding_dim))


if dblp.preprocessed_dataset_exist() and dblp.train_test_indices_exist():
    dataset_onehot = dblp.load_preprocessed_dataset()
    train_test_indices = dblp.load_train_test_indices()
else:
    if not dblp.ae_data_exist(file_path='../dataset/ae_dataset.pkl'):
        dblp.extract_data(filter_journals=True, skill_size_filter=min_skill_size, member_size_filter=min_member_size)
    if not dblp.preprocessed_dataset_exist() or not dblp.train_test_indices_exist():
        dblp.dataset_preprocessing(dblp.load_ae_dataset(file_path='../dataset/ae_dataset.pkl'), seed=seed, kfolds=k_fold)
    dataset_onehot = dblp.load_preprocessed_dataset()
    train_test_indices = dblp.load_train_test_indices()

# k_fold Cross Validation
cvscores = []

# Defining evaluation scores holders for train data
r_at_k_all_train = dblp_eval.init_eval_holder(evaluation_k_set)  # all r@k of instances in one fold and one k_evaluation_set
r_at_k_overall_train = dblp_eval.init_eval_holder(evaluation_k_set)  # overall r@k of instances in one fold and one k_evaluation_set

# Defining evaluation scores holders for test data
r_at_k_all = dblp_eval.init_eval_holder(evaluation_k_set)  # all r@k of instances in one fold and one k_evaluation_set
r_at_k_overall = dblp_eval.init_eval_holder(evaluation_k_set)  # overall r@k of instances in one fold and one k_evaluation_set

lambda_val = 0.001  # Weight decay , refer : https://stackoverflow.com/questions/44495698/keras-difference-between-kernel-and-activity-regularizers

load_weights_from_file_q = input('Load weights from file? (y/n)')
more_train_q = input('Train more? (y/n)')

time_str = time.strftime("%Y_%m_%d-%H_%M_%S")
result_output_name = "../output/predictions/{}_output.csv".format(method_name)
with open(result_output_name, 'w') as file:
    writer = csv.writer(file)
    writer.writerow(
        ['Method Name', '# Total Folds', '# Fold Number', '# Predictions', '# Truth', 'Computation Time (ms)',
         'Prediction Indices', 'True Indices'])

for fold_counter in range(1,k_fold+1):
    x_train_onehot, y_train_onehot, x_test_onehot, y_test_onehot = dblp.get_fold_data(fold_counter, dataset_onehot, train_test_indices)
    x_train_t2v, y_train_t2v, x_test_t2v, y_test_t2v = dblp.get_fold_data(fold_counter, dataset_t2v, train_test_indices)

    input_dim_onehot = x_train_onehot[0][0].shape[1]
    input_dim_t2v = x_train_t2v.shape[1]
    output_dim = y_train_onehot[0][0].shape[1]
    print("Input/output Dimensions:  ", input_dim_onehot+input_dim_t2v, output_dim)

    # this is our input placeholder
    # network parameters
    intermediate_dim_encoder_onehot = input_dim_onehot
    intermediate_dim_encoder_t2v = input_dim_t2v
    intermediate_dim_decoder = output_dim

    # VAE model = encoder + decoder
    # build encoder model
    inputs_onehot = Input(shape=(input_dim_onehot,), name='encoder_input_onehot')
    inputs_t2v = Input(shape=(input_dim_t2v,), name='encoder_input_t2v')
    x_onehot = Dense(intermediate_dim_encoder_onehot, activation='relu')(inputs_onehot)
    x_t2v = Dense(intermediate_dim_encoder_t2v, activation='relu')(inputs_t2v)
    x = concatenate([x_onehot, x_t2v])
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model([inputs_onehot, inputs_t2v], [z_mean, z_log_var, z], name='encoder')
    encoder.summary()
    # plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim_decoder, activation='relu')(latent_inputs)
    outputs = Dense(output_dim, activation='sigmoid')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    # plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

    # instantiate VAE model
    outputs = decoder(encoder([inputs_onehot, inputs_t2v])[2])
    autoencoder = Model([inputs_onehot, inputs_t2v], outputs, name='vae_mlp')

    models = (encoder, decoder)

    def vae_loss(y_true, y_pred):
        reconstruction_loss = mse(y_true, y_pred)

        reconstruction_loss *= output_dim
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        return vae_loss
    autoencoder.compile(optimizer='adam', loss=vae_loss)
    autoencoder.summary()
    # autoencoder.compile(optimizer='adagrad', loss='cross_entropy')

    # Loading model weights
    if load_weights_from_file_q.lower() == 'y':
        pick_model_weights(autoencoder, dataset_name=dataset_name)
    # x_train = x_train.astype('float32')
    # x_test = x_test.astype('float32')

    if more_train_q.lower() == 'y':
        print('Starting with back propagation batch size {} for {} times each during {} overall epochs with training batch size {}. Total epochs={}.'
            .format(back_propagation_batch_size,
                    epochs_in_batch,
                    epochs_overall,
                    training_batch_size,
                    epochs_overall * epochs_in_batch))
        # Training
        for epoch_overall in range(epochs_overall):
            batch_counter = 1
            for x_train_batch_onehot, y_train_batch_onehot, x_train_batch_t2v in zip(batch_generator(x_train_onehot, training_batch_size), batch_generator(y_train_onehot, training_batch_size), batch_generator_dense(x_train_t2v, training_batch_size)):
                print('Overall Epoch: {} | Training batch {} of fold {}.'.format(epoch_overall + 1, batch_counter, fold_counter))
                autoencoder.fit([x_train_batch_onehot, x_train_batch_t2v], y_train_batch_onehot,
                                epochs=epochs_in_batch,
                                batch_size=back_propagation_batch_size,
                                callbacks=[es],
                                shuffle=True,
                                verbose=2,
                                validation_data=(
                                [np.asarray([x_test_record[0].todense() for x_test_record in x_test_onehot]).reshape(x_test_onehot.__len__(),-1), x_test_t2v],
                                np.asarray([y_test_record[0].todense() for y_test_record in y_test_onehot]).reshape(y_test_onehot.__len__(),-1)))
                batch_counter += 1
                # Cool down GPU
                # time.sleep(300)

    score = autoencoder.evaluate(
        [np.asarray([x_test_record[0].todense() for x_test_record in x_test_onehot]).reshape(x_test_onehot.__len__(), -1), x_test_t2v],
        np.asarray([y_test_record[0].todense() for y_test_record in y_test_onehot]).reshape(y_test_onehot.__len__(), -1),
        verbose=2)
    print('Test loss of fold {}: {}'.format(fold_counter, score))
    cvscores.append(score)

    # # @k evaluation process for last train batch data
    # print("eval on last batch of train data.")
    # for k in evaluation_k_set:
    #     # r@k evaluation
    #     print("Evaluating r@k for top {} records in fold {}.".format(k, fold_counter))
    #     r_at_k, r_at_k_array = dblp_eval.r_at_k(autoencoder.predict([x_train_batch_onehot, x_train_batch_t2v]), y_train_batch_onehot, k=k)
    #     r_at_k_overall_train[k].append(r_at_k)
    #     r_at_k_all_train[k].append(r_at_k_array)
    #
    #     # print("For top {} in Train data:\nP@{}:{}\nR@{}:{}".format(k, k, p_at_k, k, r_at_k))
    #     print("For top {} in train data: R@{}:{}".format(k, k, r_at_k))


    # @k evaluation process for test data
    print("eval on test data fold #{}".format(fold_counter))
    true_indices = []
    pred_indices = []
    x_test_1 = np.asarray([x_test_record[0].todense() for x_test_record in x_test_onehot]).reshape(x_test_onehot.__len__(), -1)
    x_test_2 = x_test_t2v
    y_test = np.asarray([y_test_record[0].todense() for y_test_record in y_test_onehot]).reshape(y_test_onehot.__len__(), -1)
    with open(result_output_name, 'a+') as file:
        writer = csv.writer(file)
        for sample_x_1, sample_x_2, sample_y in zip(x_test_1, x_test_2, y_test):
            start_time = time.time()
            sample_prediction = autoencoder.predict([np.asmatrix(sample_x_1), np.asmatrix(sample_x_2)])
            end_time = time.time()
            elapsed_time = (end_time - start_time)*1000
            pred_index, true_index = dblp_eval.find_indices(sample_prediction, [sample_y])
            true_indices.append(true_index[0])
            pred_indices.append(pred_index[0])
            writer.writerow([method_name, k_fold, fold_counter, len(pred_index[0][:k_max]), len(true_index[0]),
                             elapsed_time] + pred_index[0][:k_max] + true_index[0])

    # pred_indices, true_indices = dblp_eval.find_indices(autoencoder.predict(
    #         [np.asarray([x_test_record[0].todense() for x_test_record in x_test_onehot]).reshape(x_test_onehot.__len__(), -1), x_test_t2v]),
    #         np.asarray([y_test_record[0].todense() for y_test_record in y_test_onehot]).reshape(y_test_onehot.__len__(), -1))
    # print("eval on test data.")
    # for k in evaluation_k_set:
    #     r@k evaluation
        # print("Evaluating r@k for top {} records in fold {}.".format(k, fold_counter))
        # r_at_k, r_at_k_array = dblp_eval.r_at_k(pred_indices, true_indices, k)
        # r_at_k_overall[k].append(r_at_k)
        # r_at_k_all[k].append(r_at_k_array)

    #     # print("For top {} in Test data:\nP@{}:{}\nR@{}:{}".format(k, k, p_at_k, k, r_at_k))
    #     print("For top {} in test data: R@{}:{}".format(k, k, r_at_k))


    # saving model
    # save_model_q = input('Save the models? (y/n)')
    # if save_model_q.lower() == 'y':
    model_json = autoencoder.to_json()

    # model_name = input('Please enter autoencoder model name:')

    with open('../output/Models/{}_{}_Time{}_Fold{}.json'.format(dataset_name, method_name, time_str, fold_counter), "w") as json_file:
        json_file.write(model_json)

    autoencoder.save_weights(
        "../output/Models/Weights/{}_{}_Time{}_Fold{}.h5".format(dataset_name, method_name, time_str, fold_counter))

    with open('../output/Models/{}_{}_Time{}_EncodingDim{}_Fold{}_Loss{}_Epoch{}(overall{}xinner{})_kFold{}_BatchBP{}_BatchTraining{}.txt'
                    .format(dataset_name, method_name, time_str, latent_dim, fold_counter, int(score * 1000),
                            epochs_in_batch * epochs_overall, epochs_in_batch, epochs_overall,
                            k_fold, back_propagation_batch_size, training_batch_size), 'w') as f:
        with redirect_stdout(f):
            autoencoder.summary()

    # plot_model(autoencoder, '../output/Models/{}_Time{}_EncodingDim{}_Fold{}_Loss{}_Epoch{}_kFold{}_BatchBP{}_BatchTraining{}.png'
    #            .format(dataset_name, time_str, encoding_dim, fold_counter, int(np.mean(cvscores) * 1000), epoch, k_fold,
    #                    back_propagation_batch_size, training_batch_size))
    # print('Model and its summary and architecture plot are saved.')
    print('Model and its summary are saved.')

    # Deleting model from RAM
    K.clear_session()

    # Saving evaluation data
    # cmn.utils.save_record(r_at_k_all_train, '{}_{}_r@k_all_train_Time{}'.format(dataset_name, method_name, time_str))
    # cmn.utils.save_record(r_at_k_overall_train, '{}_{}_r@k_train_Time{}'.format(dataset_name, method_name, time_str))
    #
    # cmn.utils.save_record(r_at_k_all, '{}_{}_r@k_all_Time{}'.format(dataset_name, method_name, time_str))
    # cmn.utils.save_record(r_at_k_overall, '{}_{}_r@k_Time{}'.format(dataset_name, method_name, time_str))
    #
    # print('eval records are saved successfully for fold #{}'.format(fold_counter))

    fold_counter += 1
    # break

print('Loss for each fold: {}'.format(cvscores))

# compare_submit = input('Submit for compare? (y/n)')
# if compare_submit.lower() == 'y':
#     with open('../misc/{}_r_at_k_50.pkl'.format(method_name), 'wb') as f:
#         pkl.dump(r_at_k_overall, f)