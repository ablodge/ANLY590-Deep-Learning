from keras.models import Model, Sequential
from keras.layers import Input, GRU, Dense, Embedding, Bidirectional, Flatten, LSTM, Dropout
from random import shuffle
import numpy as np
from keras.utils import np_utils
from keras_preprocessing import sequence

from keras_preprocessing.text import one_hot

X_Y = [(l.strip(), 1) for l in open('benign-urls.txt', 'r')]
X_Y += [(l.strip(), 0) for l in open('malicious-urls.txt', 'r')]
shuffle(X_Y)

X = [x for x, y in X_Y]
Y = np.array([[y] for x, y in X_Y]).astype('float32')

vocab = list(set([ch for x in X for ch in x]))


def char2vec(text):
    vecs = []
    for ch in text:
        vecs.append(vocab.index(ch))
    return np.array(vecs).astype('float32')


X = np.array([char2vec(x) for x in X])

X_train, X_test = X[0:int(0.8 * len(X))], X[int(0.8 * len(X)):]
Y_train, Y_test = Y[0:int(0.8 * len(X))], Y[int(0.8 * len(X)):]

max_url_length = 100
X_train = sequence.pad_sequences(X_train, maxlen=max_url_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_url_length)

# create the model
classifier = Sequential()
classifier.add(Embedding(len(vocab), 64, input_length=max_url_length))
classifier.add(GRU(16))
classifier.add(Dropout(0.2))
classifier.add(Dense(1, activation='sigmoid'))


classifier.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
print(classifier.summary())

# Training
print(X_train[:5])
print(Y_train[:5])
classifier.fit(X_train, Y_train, epochs=3, batch_size=32)

# Evaluation
classifier.evaluate(X_train, Y_train, batch_size=128)
classifier.evaluate(X_test, Y_test, batch_size=128)



from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt

y_pred_rnn = classifier.predict(X_test).ravel()

fpr_rnn, tpr_rnn, thresholds_rnn = roc_curve(Y_test.ravel(), y_pred_rnn)

# Plot ROC (based on https://hackernoon.com/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier-2ecc6c73115a)
plt.figure(2)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_cnn, tpr_cnn, label='CNN')
plt.plot(fpr_rnn, tpr_rnn, label='RNN')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

