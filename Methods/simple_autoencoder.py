import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model
from keras.losses import mean_squared_error
from keras import backend as K
import numpy as np
import time
print(K.tensorflow_backend._get_available_gpus())

data_train = np.load('./dataset/train_x_50.npy')
data_test = np.load('./dataset/test_x_50.npy')
x_train = data_train
x_test = data_test

# this is the size of our encoded representations
encoder_hidden_layer_dim = 3675  # encoder first hidden layer
# encoder_hidden_layer_dim_2 = 4200  # encoder second hidden layer
# encoder_hidden_layer_dim_3 = 2700  # encoder third hidden layer
encoding_dim = 1200  # encoded size
decoder_hidden_layer_dim = 3675  # decoder first hidden layer
# data_dim is input dimension
data_dim = x_train.shape[1]
# this is our input placeholder
input_img = Input(shape=(data_dim,))
# hidden #1 in encoder
hidden_layer_1 = Dense(encoder_hidden_layer_dim, activation='relu')(input_img)
# hidden #2 in encoder
# hidden_layer_1_2 = Dense(encoder_hidden_layer_dim_2, activation='relu')(hidden_layer_1)
# hidden #3 in encoder
# hidden_layer_1_3 = Dense(encoder_hidden_layer_dim_2, activation='relu')(hidden_layer_1_2)
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(hidden_layer_1)
# encoded = Dense(encoding_dim, activation='relu')(input_img)
# hidden #2 in decoder
hidden_layer_2 = Dense(decoder_hidden_layer_dim, activation='relu')(encoded)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(data_dim, activation='sigmoid')(hidden_layer_2)
# decoded = Dense(data_dim, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve hidden layer
decoder_hidden = autoencoder.layers[-2]
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(decoder_hidden(encoded_input)))
# decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

batch_sizes = [128, 64, 32, 4, 1]
epochs      = [50, 30, 20, 10, 3]
for epoch, batch_size in zip(epochs, batch_sizes):
    print('*'*30)
    print('*'*30)
    print('Starting batch size {} for {} times.'.format(batch_size, epoch))
    # Training
    autoencoder.fit(x_train, x_train,
                    epochs=epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    verbose=2,
                    validation_data=(x_test, x_test))
    # Evaluating
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)
    # do a loss compute for model:
    loses = K.eval(mean_squared_error(x_test, decoded_imgs))
    lose = np.mean(loses)
    print('Total loss of model is: {0:.4f}'.format(lose))
    # Cool down GPU
    time.sleep(300)

# Preview
n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(50, 50, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(50, 50, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# saving model
save_model_q = input('Save the models? (y/n)')
if save_model_q == 'y':
    encoder_model_json = encoder.to_json()
    decoder_model_json = decoder.to_json()
    autoencoder_model_json = autoencoder.to_json()

    encoder_name = input('Please enter encoder model name:')
    decoder_name = input('Please enter decoder model name:')
    autoencoder_name = input('Please enter autoencoder model name:')

    with open('./models/{}.json'.format(encoder_name), "w") as json_file:
        json_file.write(encoder_model_json)
    with open('./models/{}.json'.format(decoder_name), "w") as json_file:
        json_file.write(decoder_model_json)
    with open('./models/{}.json'.format(autoencoder_name), "w") as json_file:
        json_file.write(autoencoder_model_json)

    encoder.save_weights("./weights/{}.h5".format(encoder_name))
    decoder.save_weights("./weights/{}.h5".format(decoder_name))
    autoencoder.save_weights("./weights/{}.h5".format(autoencoder_name))
