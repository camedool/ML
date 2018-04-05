# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 17:12:39 2018

"""

import pandas as pd
import re
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import string

#====================
# CONSTANTS
predictionColumn="description"

#====================
# STOPWORDS
# download the stopwords and add punkt
nltk.download('stopwords')
nltk.download('punkt')
stopWords = stopwords.words('english') + list(string.punctuation)

# define the custom stopwords (specific for our dataset) and update our stopwords
customStopWords = ["fw", "inc", "re", "``", '"']
stopWords +=  customStopWords
#====================

def cleaningData(row):
    '''
    initial cleaning of read csv data (regular expression)
    '''
    # pattern to high level cleaning
    patternRegards = '(best|kind)* regards,.*'
    patternDivider = '________________________________.*'
    patternCid = '\[cid.*\]'
    patternLotus = r'\[*<*notes://*([\w*|/*]){1,}\]*>*'
    patternUrl = r'https?://([\w*|/*|\.*]){1,}'
    patternDigit = '(\w|\.)*\d+(\w|\.)*'
         
    # short desription cleaning
    row['short_description'] = re.sub(patternRegards, "", row['short_description'], 0, re.DOTALL|re.I)
    row['short_description']  = re.sub(patternDivider, "", row['short_description'] , 0, re.DOTALL)
    row['short_description'] = re.sub(patternDigit, '', row['short_description'])
    row['short_description'] = re.sub(patternCid, '', row['short_description'], 0, re.I)
    row['short_description'] = re.sub(patternLotus, '', row['short_description'], 0, re.I)
    row['short_description'] = re.sub(patternUrl, '', row['short_description'], 0, re.I)
    row['short_description'] = row['short_description'].replace("?", "")
    row['short_description'] = row['short_description'].replace("'", "")
    row['short_description'] = row['short_description'].replace('"', '')
    row['short_description'] = row['short_description'].replace('*', '')
    row['short_description'] = row['short_description'].replace('/', '')    

    row['short_description'] = row['short_description'].strip()
    
    # short_description and description cleaning
    row['description'] = row['short_description'] + ". " + row['description']      
    row['description'] = re.sub(patternRegards, "", row['description'], 0, re.DOTALL|re.I)
    row['description']  = re.sub(patternDivider, "", row['description'] , 0, re.DOTALL)
    row['description'] = re.sub(patternDigit, '', row['description'])
    row['description'] = re.sub(patternCid, '', row['description'], 0, re.I)
    row['description'] = re.sub(patternLotus, '', row['description'], 0, re.I)
    row['description'] = re.sub(patternUrl, '', row['description'], 0, re.I)
    row['description'] = row['description'].replace("?", "")
    row['description'] = row['description'].replace("'", "")
    row['description'] = row['description'].replace('"', '')
    row['description'] = row['description'].replace('*', '')
    row['description'] = row['description'].replace('/', '')
    
    row['description'] = row['description'].strip()
    
    # removing the stopWords and non-ascii
    stopWordsCleaning(row)
    
    return row

def stopWordsCleaning(row, columnName=predictionColumn):
    '''
    creating stopwords and removing from short_description and description
    '''

    str = ''
    for word in word_tokenize(row[columnName].lower()):
        if word not in stopWords:
            try:
                word.encode('ascii')
                str += word + ' '
            except UnicodeEncodeError:
                pass
    row[columnName] = str
    

def loadCleanData(fileName='data.csv'):

    # reading the csv file and creating inital dataFrame
    dataFrame = pd.read_csv(fileName, encoding='latin-1',)
    
    # applying regular expression and concatenating short_description and description columns
    dataFrame = dataFrame.apply(cleaningData, axis=1)
    
    # getting the dataframe with only necessary columns
    cleanDataFrame = pd.DataFrame(dataFrame, 
                                  columns=['category', 
                                           'subcategory', 
                                           'short_description',
                                           'description', 
                                           'category_id'])
    # releasing memory
    del dataFrame
    return cleanDataFrame

    
    
    
    