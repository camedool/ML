# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 00:35:18 2018

"""
from keras.models import load_model
import numpy as np
from keras.preprocessing.text import Tokenizer
import pandas as pd

model = load_model('text_classifier.h5')

def predict(str_query):

    X_raw_test = [str_query]
    x_test = tokenizer.texts_to_matrix(X_raw_test, mode='binary')
    prediction = model.predict(np.array(x_test))
    class_num = np.argmax(prediction[0])
    
    print("%s: %f%% confidence" % (encoder.classes_[class_num], prediction[0][np.argmax(prediction)]*100))
    
    return prediction
