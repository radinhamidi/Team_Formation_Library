import pickle as pkl
import time
from keras.layers import Lambda
from keras.losses import mse, binary_crossentropy, mae, kld, categorical_crossentropy
from keras.layers import Input, Dense
from keras.models import Model
from contextlib import redirect_stdout
import cmn.utils
from cmn.utils import *
import dal.load_dblp_data as dblp
import eval.evaluator as dblp_eval
from ml.nn_custom_func import *

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

#settings
dataset_name = 'DBLP'
method_name = 'T2V_user_VAE'

#eval settings
k_fold = 10
k_max = 50
evaluation_k_set = np.arange(1, k_max+1, 1)

#nn settings
epochs = 300
back_propagation_batch_size = 64
min_skill_size = 0
min_member_size = 0
latent_dim = 50

print(K.tensorflow_backend._get_available_gpus())

t2v_model = Team2Vec()
t2v_model = load_T2V_model(t2v_model)
embedding_dim = t2v_model.model.vector_size

if dblp.ae_data_exist(file_path='../dataset/ae_t2v_dim{}_dataset.pkl'.format(embedding_dim)):
    dataset = dblp.load_ae_dataset(file_path='../dataset/ae_t2v_dim{}_dataset.pkl'.format(embedding_dim))
else:
    if not dblp.ae_data_exist(file_path='../dataset/ae_dataset.pkl'):
        dblp.extract_data(filter_journals=True, skill_size_filter=min_skill_size, member_size_filter=min_member_size)
    if not dblp.preprocessed_dataset_exist() or not dblp.train_test_indices_exist():
        dblp.dataset_preprocessing(dblp.load_ae_dataset(file_path='../dataset/ae_dataset.pkl'), seed=seed, kfolds=k_fold, shuffle_at_the_end=True)
    preprocessed_dataset = dblp.load_preprocessed_dataset()

    dblp.nn_t2v_dataset_generator(t2v_model, preprocessed_dataset, output_file_path='../dataset/ae_t2v_dim{}_dataset.pkl'.format(embedding_dim), mode='user')
    del preprocessed_dataset
    dataset = dblp.load_ae_dataset(file_path='../dataset/ae_t2v_dim{}_dataset.pkl'.format(embedding_dim))


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


# k-fold Cross Validation
cvscores = []

# Defining evaluation scores holders for train data
r_at_k_all_train = dblp_eval.init_eval_holder(evaluation_k_set)  # all r@k of instances in one fold and one k_evaluation_set
r_at_k_overall_train = dblp_eval.init_eval_holder(evaluation_k_set)  # overall r@k of instances in one fold and one k_evaluation_set

# Defining evaluation scores holders for test data
r_at_k_all = dblp_eval.init_eval_holder(evaluation_k_set)  # all r@k of instances in one fold and one k_evaluation_set
r_at_k_overall = dblp_eval.init_eval_holder(evaluation_k_set)  # overall r@k of instances in one fold and one k_evaluation_set

lambda_val = 0.001  # Weight decay , refer : https://stackoverflow.com/questions/44495698/keras-difference-between-kernel-and-activity-regularizers

time_str = time.strftime("%Y%m%d-%H%M%S")
train_test_indices = dblp.load_train_test_indices()
for fold_counter in range(1,k_fold+1):
    x_train, y_train, x_test, y_test = dblp.get_fold_data(fold_counter, dataset, train_test_indices)

    train_index = train_test_indices[fold_counter]['Train']
    test_index = train_test_indices[fold_counter]['Test']
    y_sparse_train = []
    y_sparse_test = []
    preprocessed_dataset = dblp.load_preprocessed_dataset()
    for sample in preprocessed_dataset:
        id = sample[0]
        if id in train_index:
            y_sparse_train.append(sample[2])
        elif id in test_index:
            y_sparse_test.append(sample[2])
    y_sparse_train = np.asarray(y_sparse_train).reshape(y_sparse_train.__len__(), -1)
    y_sparse_test = np.asarray(y_sparse_test).reshape(y_sparse_test.__len__(), -1)
    del preprocessed_dataset

    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]
    print("Input/output Dimensions:  ", input_dim, output_dim)

    # this is our input placeholder
    # network parameters
    intermediate_dim_encoder = input_dim
    intermediate_dim_decoder = output_dim

    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=(input_dim,), name='encoder_input')
    x = Dense(intermediate_dim_encoder, activation='relu')(inputs)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
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
    outputs = decoder(encoder(inputs)[2])
    autoencoder = Model(inputs, outputs, name='vae_mlp')

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
    print("eval on last batch of train data.")
    for k in evaluation_k_set:
        print("Evaluating r@k for top {} records in fold {}.".format(k, fold_counter))
        r_at_k, r_at_k_array = dblp_eval.r_at_k_t2v(y_train_pred, y_sparse_train, k=k)
        r_at_k_overall_train[k].append(r_at_k)
        r_at_k_all_train[k].append(r_at_k_array)
        print("For top {} in Train data: R@{}:{}".format(k, k, r_at_k))

    # @k evaluation process for test data
    print("eval on test data.")
    for k in evaluation_k_set:
        # r@k evaluation
        print("Evaluating r@k for top {} records in fold {}.".format(k, fold_counter))
        r_at_k, r_at_k_array = dblp_eval.r_at_k_t2v(y_test_pred, y_sparse_test, k=k)
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

        with open('../output/Models/{}_{}_Time{}_Fold{}.json'.format(dataset_name, method_name, time_str, fold_counter), "w") as json_file:
            json_file.write(model_json)

        autoencoder.save_weights("../output/Models/Weights/{}_{}_Time{}_Fold{}.h5".format(dataset_name, method_name, time_str, fold_counter))

        with open('../output/Models/{}_{}_Time{}_EncodingDim{}_Fold{}_Loss{}_Epoch{}_kFold{}_BatchBP{}.txt'
                        .format(dataset_name, method_name, time_str, embedding_dim, fold_counter, int(score * 1000),
                                epochs, k_fold, back_propagation_batch_size), 'w') as f:
            with redirect_stdout(f):
                autoencoder.summary()

        # plot_model(autoencoder, '../output/Models/{}_Time{}_EncodingDim{}_Fold{}_Loss{}_Epoch{}_kFold{}_BatchBP{}_BatchTraining{}.png'
        #            .format(dataset_name, time_str, encoding_dim, fold_counter, int(np.mean(cvscores) * 1000), epoch, k_fold,
        #                    back_propagation_batch_size, training_batch_size))
        # print('Model and its summary and architecture plot are saved.')
        print('Model and its summary are saved.')

    # Deleting model from RAM
    # K.clear_session()

    # Saving evaluation data
    cmn.utils.save_record(r_at_k_all_train, '{}_{}_r@k_all_train_Time{}'.format(dataset_name, method_name, time_str))
    cmn.utils.save_record(r_at_k_overall_train, '{}_{}_r@k_train_Time{}'.format(dataset_name, method_name, time_str))

    cmn.utils.save_record(r_at_k_all, '{}_{}_r@k_all_Time{}'.format(dataset_name, method_name, time_str))
    cmn.utils.save_record(r_at_k_overall, '{}_{}_r@k_Time{}'.format(dataset_name, method_name, time_str))

    print('eval records are saved successfully for fold #{}'.format(fold_counter))

    fold_counter += 1
    break

print('Loss for each fold: {}'.format(cvscores))


with open('../misc/{}_dim{}_member_r_at_k_50.pkl'.format(method_name, embedding_dim), 'wb') as f:
    pkl.dump(r_at_k_overall, f)


# with open('../misc/T2V_user_VAE_dim{}_team_r_at_k_50.pkl'.format(embedding_dim), 'wb') as f:
#     pkl.dump(r_at_k_overall, f)
