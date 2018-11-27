from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, UpSampling2D
from keras.datasets import fashion_mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


X_train = x_train.reshape(x_train.shape[0], 28, 28,1)
X_test = x_test.reshape(x_test.shape[0], 28, 28,1)


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# Encoder
autoencoder = Sequential()
autoencoder.add(Conv2D(128, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=(28,28,1),
                   padding='same'))
autoencoder.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=(28,28,1),
                   padding='same'))
autoencoder.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
autoencoder.add(Conv2D(1, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                   padding='same'))
autoencoder.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
autoencoder.add(Dropout(0.25))

# Decoder
autoencoder.add(UpSampling2D(size=(4,4)))
autoencoder.add(Conv2D(1, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                   padding='same'))

# print layer shapes
encoder = Sequential(layers=autoencoder.layers[0:5])

autoencoder.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['accuracy'])
print(autoencoder.summary())

# Training
autoencoder.fit(X_train, X_train, epochs=10, batch_size=32)

# Evaluation
autoencoder.evaluate(X_train, X_train, batch_size=128)
autoencoder.evaluate(X_test, X_test, batch_size=128)


# based on autoencoders.ipynb
n=5
for k in range(n):
    ax = plt.subplot(2, n, k + 1)
    X = X_test[k:k+1,:].reshape((28,28))
    X *= 255
    X = X.astype(int)
    ax = plt.subplot(2, n, k + 1 + n)
    reconstruction = autoencoder.predict(X_test[k:k+1,:])
    reconstruction.resize((28,28))
    reconstruction*=255
    reconstruction = reconstruction.astype(int)
plt.savefig('auto.png')