# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 17:12:39 2018

"""
# common modules
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation,Dropout,Embedding
import keras.utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder

# custom modules
import loadgData

#=============
#CONSTANTS
#=============
epochs = 5
batch_size = 32
max_words = 5000
fileName = 'data.csv'
#============

# laoding cleaned data frame and assigning x and y vectors
dataFrame = loadData.loadCleanData()
num_classes = len(dataFrame['category'].drop_duplicates())

X_raw = dataFrame['short_description'].values
Y_raw = dataFrame['category'].values

# transform the training description data into matrix
tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(X_raw)
x_train = tokenizer.texts_to_matrix(X_raw)

# transforming the category
encoder = LabelEncoder()
encoder.fit(Y_raw)
encoded_Y = encoder.transform(Y_raw)
y_train = keras.utils.to_categorical(encoded_Y, num_classes)

# building, compiling, training and saving a model
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)

model.save('text_classifier.h5')
