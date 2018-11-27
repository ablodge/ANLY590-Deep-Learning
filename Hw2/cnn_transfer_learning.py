from keras import Sequential
from keras.applications import VGG16, ResNet50
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, UpSampling2D, ZeroPadding2D
from keras.datasets import fashion_mnist
from keras.utils import np_utils

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


X_train = x_train.reshape(x_train.shape[0], 28, 28,1)
X_test = x_test.reshape(x_test.shape[0], 28, 28,1)


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)


classifier = Sequential()
classifier.add(ResNet50(weights="imagenet",input_shape=(28,28,1)))
classifier.add(Flatten())
classifier.add(Dense(10,activation='softmax'))
classifier.add(Dropout(0.25))

classifier.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
print(classifier.summary())

# Training
classifier.fit(X_train, Y_train, epochs=10, batch_size=32)

# Evaluation
classifier.evaluate(X_train, Y_train, batch_size=128)
classifier.evaluate(X_test, Y_test, batch_size=128)
