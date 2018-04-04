# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 17:12:39 2018

"""

import pandas as pd
import re
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
import json
import numpy as np
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
from sklearn.preprocessing import LabelEncoder


# reading the csv file and creating inital dataFrame
dataFrame = pd.read_csv('data - Copy.csv', encoding='latin-1',)


def replaceSignature(row):
    
    # pattern to high level cleaning
    patternBestRegards = 'Best regards.*'
    patternKindRegards = 'Kind regards.*'
    patternDivider = '________________________________.*'
    patternCid = '\[cid.*\]'
    patternLotus = '\[Notes.*\]'
    
    row['description'] = row['short_description'] + ". " + row['description']      
    row['description'] = re.sub(patternBestRegards, "", row['description'], 0, re.DOTALL|re.I)
    row['description'] = re.sub(patternKindRegards, "", row['description'], 0, re.DOTALL|re.I)
    row['description']  = re.sub(patternDivider, "", row['description'] , 0, re.DOTALL)
    row['description'] = re.sub(r'\d+', '', row['description'])
    row['description'] = re.sub(patternCid, '', row['description'])
    row['description'] = re.sub(patternLotus, '', row['description'])
    row['description'] = row['description'].replace("'", "")
    row['description'] = row['description'].replace('"', '')

    row['description'] = row['description'].strip()
    
    return row

# applying regular expression and concatenating short_description and description columns
dataFrame = dataFrame.apply(replaceSignature, axis=1)

# getting the dataframe with only necessary columns
cleanDataFrame = pd.DataFrame(dataFrame, 
                              columns=['category', 
                                       'subcategory', 
                                       'short_description',
                                       'description', 
                                       'category_id'])

nltk.download('stopwords')
nltk.download('punkt')

customStopWords = ["fw:", "inc", "re:", "``", '"']
stopWords = stopwords.words('english') + list(string.punctuation) + customStopWords
cleanItems = []
for words in cleanDataFrame['description'].astype(str):
    cleanItems.append([word for word in 
                       word_tokenize(words.lower()) if word not in stopWords])

# releasing memory
dataFrame
#train_x = cleanDataFrame['description']
train_x = np.asarray(cleanItems)
train_y = cleanDataFrame['category_id']

max_words = 5000

tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(train_x)

dictionary = tokenizer.word_index
# Let's save this out so we can use it later
with open('dictionary.json', 'w') as dictionary_file:
    json.dump(dictionary, dictionary_file)


def convert_text_to_index_array(text):
    # one really important thing that `text_to_word_sequence` does
    # is make all texts the same length -- in this case, the length
    # of the longest text in the set.
    res = []
    for word in text:
        try:
            res.append(dictionary[word])
        except KeyError:
            pass
    return res
    #return [dictionary[word] for word in kpt.text_to_word_sequence(text)]

allWordIndices = []
# for each tweet, change each token to its ID in the Tokenizer's word_index
for text in train_x:
    wordIndices = convert_text_to_index_array(text)
    allWordIndices.append(wordIndices)

# now we have a list of all tweets converted to index arrays.
# cast as an array for future usage.
allWordIndices = np.asarray(allWordIndices)

# create one-hot matrices out of the indexed tweets
train_x = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')
# treat the labels as categories
encoder = LabelEncoder()
encoder.fit(train_y)

encoded_Y = encoder.transform(train_y)
print(encoded_Y)

y_train = keras.utils.to_categorical(encoded_Y, 20)
train_y = keras.utils.to_categorical(train_y, 20)


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

model = Sequential()
model.add(Dense(512, input_shape=(max_words,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(20, activation='sigmoid'))
#model.add(Dropout(0.5))
#model.add(Dense(20, activation='softmax'))

model.compile(loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy'])

model.fit(train_x, y_train,
  batch_size=32,
  epochs=5,
  verbose=1,
  validation_split=0.1,
  shuffle=True)

model_json = model.to_json()

with open('model.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')

labels = ['HRIS - Learning Link','HRIS - CompetencePortal','HRIS - Access Management','HRIS - Legacy - AC Staff','HRIS - Reporting & Analytics','HRIS - Legacy - Absence-DB','HRIS - Master Data Management','HRIS - Legacy - NAV','HRIS - Legacy - Employee Request- / DQM-','HRIS - People Management','HRIS - Mobile App','HRIS - Time Off','HRIS - Knowledge Management','HRIS - Change Management','HRIS - Legacy - Recruitment-DB','HRIS - Legacy - IJM','HRIS - Integrations','HRIS - Travel & Expense','HRIS - Payroll','HRIS - Other']


def testPrediction(text):
    testArr = convert_text_to_index_array(text)
    input = tokenizer.sequences_to_matrix([testArr], mode='binary')
    # predict which bucket your input belongs in
    pred = model.predict(input)
    # and print it for the humons
    print("%s ; %f%% confidence" % (labels[np.argmax(pred)], pred[0][np.argmax(pred)] * 100))
    return pred
