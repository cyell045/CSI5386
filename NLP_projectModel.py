# Instructions to run (copied from readme.md):
# - to run the code, you must be in the virtual environment where the necessary libraries are installed (see the imports at the top of the code file)
#     > source activate <virtual env name> (e.g. NLPproject)
# - navigate to directory containing this file
# - make sure the data file is in the same directory and has the same name as the variable "dataFile" below after the imports
# - set parameters as desired (highlighted in code with 50 * above and below)
# - use python3 to run code
#     > python3 NLP_projectModel.py
# - when prompted, you may enter lyrics to check their main topic, or input "exit" to move on

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import itertools
import os

# fixes bug initializing libiomp5.dylib
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils

# This code was tested with TensorFlow v1.4
print("You have TensorFlow version", tf.__version__)


# **************************************************
# TODO: update each time file changes
# **************************************************
dataFile = "output4topics.tsv"


# data in tab-delimited file with three columns: id, tag, lyrics
data = pd.read_csv(dataFile, sep="\t")


# Split data into train and test
train_size = int(len(data) * .8)
print ("Train size: %d" % train_size)
print ("Test size: %d" % (len(data) - train_size))


train_posts = data['lyrics'][:train_size]
train_tags = data['tags'][:train_size]

test_posts = data['lyrics'][train_size:]
test_tags = data['tags'][train_size:]

# **************************************************
# PARAMETER:
max_words = 1000 # default 1000
# **************************************************

tokenize = text.Tokenizer(num_words=max_words, char_level=False)


tokenize.fit_on_texts(train_posts) # only fit on train
x_train = tokenize.texts_to_matrix(train_posts)
x_test = tokenize.texts_to_matrix(test_posts)


# Use sklearn utility to convert label strings to numbered index
encoder = LabelEncoder()
encoder.fit(train_tags)
y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)


# Converts the labels to a one-hot representation
num_classes = np.max(y_train) + 1
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)


# Inspect the dimensions of our training and test data (this is helpful to debug)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)


# **********
# PARAMETERS:
batch_size = 32
epoch_num = 2 # default = 2 to avoid overfitting
num_neurons = 512 # default = 512
# **********


# Build the model
model = Sequential()
model.add(Dense(num_neurons, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# **************************************************
# PARAMETER:
# - validation_percent is percentage of training data to use for validation
# **************************************************
validation_percent = 0.1 # default = 0.1

# model.fit trains the model
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epoch_num,
                    verbose=1,
                    validation_split=validation_percent)


# Evaluate the accuracy of our trained model
score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])


# Here's how to generate a prediction on individual examples
text_labels = encoder.classes_

for i in range(10):
	# NOTE: x_test contains the one-hot vectors for the test data
	prediction = model.predict(np.array([x_test[i]]))
	predicted_label = text_labels[np.argmax(prediction)]
	print(test_posts.iloc[i][:50], "...")
	print('Actual label:' + test_tags.iloc[i])
	print("Predicted label: " + predicted_label + "\n")

y_softmax = model.predict(x_test)

y_test_1d = []
y_pred_1d = []

for i in range(len(y_test)):
    probs = y_test[i]
    index_arr = np.nonzero(probs)
    one_hot_index = index_arr[0].item(0)
    y_test_1d.append(one_hot_index)

for i in range(0, len(y_softmax)):
    probs = y_softmax[i]
    predicted_index = np.argmax(probs)
    y_pred_1d.append(predicted_index)

# Request lyrics:

# - write predicted label
def predictLabel(lyrics):
	oneHotArray = tokenize.texts_to_matrix([text])
	# print(oneHotArray)
	prediction = model.predict(oneHotArray)
	predicted_label = text_labels[np.argmax(prediction)]
	return predicted_label

exit = False

print("Type exit to stop.")

while (not exit):
	text = input("Lyrics please:\n")
	print(predictLabel(text))
	if (text == "exit"):
		break




# This utility function is from the sklearn docs: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
# The final argument is a bool of whether to display probability (if false, then counts shown)
def plot_confusion_matrix(cm, classes,
                          title='',
                          cmap=plt.cm.Blues,
                          b_probability=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    fmt = 'n'
    if b_probability:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=22)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=18)
    plt.yticks(tick_marks, classes, fontsize=18)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)

    plt.tight_layout()

cnf_matrix = confusion_matrix(y_test_1d, y_pred_1d)
plt.figure(figsize=(5,5))

print('matrix:\n' + str(cnf_matrix))
print("Number of labels: " + str(len(text_labels)))

plot_confusion_matrix(cnf_matrix, classes=text_labels, title="", b_probability=True)
plt.show()