from dal.load_dblp_data import *
from keras.layers import Input, Dense
from keras.models import Model
from keras import backend as K
import numpy as np
from keras.utils.vis_utils import plot_model

print(K.tensorflow_backend._get_available_gpus())

######## Definitions
data_size_limit = 1000
train_ratio = 0.7
validation_ratio = 0.15
########


if ae_data_exist():
    dataset = load_ae_dataset()
else:
    extract_data(size_limit=data_size_limit)
    dataset = load_ae_dataset()

x = []
y = []
for record in dataset:
    x.append(record[0])
    y.append(record[1])
del dataset

x = np.asarray(x)
y = np.asarray(y)

x_train, x_validate, x_test, ids = crossValidate(x, train_ratio, validation_ratio)
y_train = y[ids[0:int(y.__len__() * train_ratio)]]
y_validate = y[ids[int(y.__len__() * train_ratio):int(y.__len__() * (train_ratio + validation_ratio))]]
y_test = y[ids[int(y.__len__() * (train_ratio + validation_ratio)):]]

# this is the size of our encoded representations
encoder_hidden_layer_dim = 3675  # encoder first hidden layer
# encoder_hidden_layer_dim_2 = 4200  # encoder second hidden layer
# encoder_hidden_layer_dim_3 = 2700  # encoder third hidden layer
encoding_dim = 1200  # encoded size
decoder_hidden_layer_dim = 3675  # decoder first hidden layer
# data_dim is input dimension
input_dim = x_train.shape[1]
output_dim = y_train.shape[1]
# this is our input placeholder
input_img = Input(shape=(input_dim,))
# hidden #1 in encoder
# hidden_layer_1 = Dense(encoder_hidden_layer_dim, activation='relu')(input_img)
# hidden #2 in encoder
# hidden_layer_1_2 = Dense(encoder_hidden_layer_dim_2, activation='relu')(hidden_layer_1)
# hidden #3 in encoder
# hidden_layer_1_3 = Dense(encoder_hidden_layer_dim_2, activation='relu')(hidden_layer_1_2)
# "encoded" is the encoded representation of the input
# encoded = Dense(encoding_dim, activation='relu')(hidden_layer_1)
encoded = Dense(encoding_dim, activation='relu')(input_img)
# hidden #2 in decoder
# hidden_layer_2 = Dense(decoder_hidden_layer_dim, activation='relu')(encoded)
# "decoded" is the lossy reconstruction of the input
# decoded = Dense(output_dim, activation='sigmoid')(hidden_layer_2)
decoded = Dense(output_dim, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(inputs=input_img, outputs=decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (32-dimensional) input
# encoded_input = Input(shape=(encoding_dim,))
# retrieve hidden layer
# decoder_hidden = autoencoder.layers[-2]
# retrieve the last layer of the autoencoder model
# decoder_layer = autoencoder.layers[-1]
# create the decoder model
# decoder = Model(encoded_input, decoder_layer(decoder_hidden(encoded_input)))
# decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# print(x_train.shape)
# print(x_test.shape)

# batch_sizes = [128, 64, 32, 4, 1]
batch_sizes = [8]
epochs = [20]
# epochs      = [50, 30, 20, 10, 3]
for epoch, batch_size in zip(epochs, batch_sizes):
    print('*' * 30)
    print('*' * 30)
    print('Starting batch size {} for {} times.'.format(batch_size, epoch))
    # Training
    autoencoder.fit(x_train, y_train,
                    epochs=epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    verbose=2,
                    validation_data=(x_validate, y_validate))
    # Evaluating
    # encoded_imgs = encoder.predict(x_test)
    # decoded_imgs = decoder.predict(encoded_imgs)
    # do a loss compute for model:
    # loses = K.eval(mean_squared_error(x_test, decoded_imgs))
    # lose = np.mean(loses)
    # print('Total loss of model is: {0:.4f}'.format(lose))
    # Cool down GPU
    # time.sleep(300)
    score = autoencoder.evaluate(x_test, y_test, verbose=2)
    print('Test loss:', score[0])

# saving model
save_model_q = input('Save the models? (y/n)')
if save_model_q == 'y':
    # encoder_model_json = encoder.to_json()
    # decoder_model_json = decoder.to_json()
    autoencoder_model_json = autoencoder.to_json()

    # encoder_name = input('Please enter encoder model name:')
    # decoder_name = input('Please enter decoder model name:')
    autoencoder_name = input('Please enter autoencoder model name:')

    # with open('./Models/{}.json'.format(encoder_name), "w") as json_file:
    #     json_file.write(encoder_model_json)
    # with open('./Models/{}.json'.format(decoder_name), "w") as json_file:
    #     json_file.write(decoder_model_json)
    with open('./Models/{}.json'.format(autoencoder_name), "w") as json_file:
        json_file.write(autoencoder_model_json)

    # encoder.save_weights("./Models/weights/{}.h5".format(encoder_name))
    # decoder.save_weights("./Models/weights/{}.h5".format(decoder_name))
    autoencoder.save_weights("./Models/weights/{}.h5".format(autoencoder_name))

plot_model(autoencoder, to_file='./Figures/model_plot.png', show_shapes=True, show_layer_names=True)