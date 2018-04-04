# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 20:32:59 2018

@author: camedool

url to download word2vec dictioany: https://docs.google.com/uc?export=download&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM
Example is taken from this source: https://vgpena.github.io/classifying-tweets-with-keras-and-tensorflow/
loading the model - one-hot vector approach?
"""

import json
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
import numpy as np
import time
import pandas as pd
from gensim.models import KeyedVectors as kv

start_time = time.time()
#
#training = np.genfromtxt('data.csv', delimiter=',',
#	skip_header=1, usecols=(8, 13), dtype=None, encoding="utf8")

training = pd.read_csv('data.csv', sep=';', encoding='latin-1')
# create our training data from short description
train_x = training['short_description'].astype(str).values.tolist()

# # get the category name
train_y = training['category_number'].astype(str).values.tolist()

# # only works with the 3000 most popular words found in dataset
max_words = 3000
MAX_SEQUENCE_LENGTH = 300

# # create a new Tokenizer
tokenizer = Tokenizer(num_words=max_words)

 # feed our tweets into the Tokenizer
tokenizer.fit_on_texts(train_x)
sequences = tokenizer.texts_to_sequences(train_x)
sequences = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

print("shape of tensor data:", sequences.shape)
#print("shape of lables:", train_y.shape)


# # Tokenizers come with a convenient list of words and IDs
word_index = tokenizer.word_index
print("Found %s unique tokens." % len(word_index))

EMBEDDING_DIM = 300
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))


# Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
word2vec = kv.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

for word, i in word_index.items():
    try:
        embedding_vector = word2vec[word]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        pass

print("The shape of embedding matrix is {}".format(embedding_matrix.shape))





# # Let's save this out so we can use it later
with open('dictionary.json', 'w') as dictionary_file:
 	json.dump(dictionary, dictionary_file)

def convet_text_to_index_array(text):
 	# one really important thing that 'text_to_word_sequence' does
 	# is make all texts the same length - in this case-> the lenght 
 	# of the longest text in the set.
 	return [dictionary[word] for word in kpt.text_to_word_sequence(text)]

allWordIndices = []
 # for each tweet, change each token to its ID in the Tokenizer's word_index
for text in train_x:
    wordIndices = convet_text_to_index_array(text)
    allWordIndices.append(wordIndices)

 # now we have a list of all tweets converted to index arrays
 # cast as ana array for future usage
allWordIndices = np.array(allWordIndices)

 # create one-hot matrices out of the indexed tweets
train_x = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')

 # treat the labels as categories
train_y = keras.utils.to_categorical(train_y)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

model = Sequential()
model.add(Dense(500, input_shape=(max_words,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(20, activation='softmax'))

#model.compile(loss='categorical_crossentropy',
#              optimizer='adam',
#              metrics=['accuracy'])

#history = model.fit(train_x, train_y,
#                    batch_size=64,
#                    epochs=5,
#                    validation_split=0.1,
#                    shuffle=True)

# # saving the output of the training
# model_json = model.to_json()
# with open('model' + fileSep + 'K.json', 'w') as json_file:
# 	json_file.write(model_json)

# model.save_weights('model' + fileSep +'K.h5')
print("---- It took to make model %s seconds ---" % (time.time() - start_time))

print("predicting...")
def convert_text_to_index_array(text):
    words = kpt.text_to_word_sequence(text)
    wordIndices = []
    for word in words:
        if word in dictionary:
            wordIndices.append(dictionary[word])
        else:
            print("'%s' not in training corpus; ignoring." %(word))
    return wordIndices
testArr = convert_text_to_index_array("ePurchase access")
inputText = tokenizer.sequences_to_matrix([testArr], mode='binary')
# predict which bucket your input belongs in
#pred = model.predict(inputText)

labels =['HRIS - Other','HRIS - Payroll','HRIS - Travel & Expense','HRIS - Integrations','HRIS - Legacy - IJM','HRIS - Legacy - Recruitment-DB','HRIS - Change Management','HRIS - Knowledge Management','HRIS - Time Off','HRIS - Mobile App','HRIS - People Management','HRIS - Legacy - Employee Request- / DQM-','HRIS - Legacy - NAV','HRIS - Master Data Management','HRIS - Legacy - Absence-DB','HRIS - Reporting & Analytics','HRIS - Legacy - AC Staff','HRIS - Access Management','HRIS - CompetencePortal','HRIS - Learning Link',
]

# and print it for the humons
print("%s category; %f%% confidence" % (labels[np.argmax(pred)], pred[0][np.argmax(pred)] * 100))
