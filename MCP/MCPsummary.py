#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 19:24:26 2019

@author: sterling
"""
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Dense
from pandas import DataFrame
from matplotlib import pyplot
import string
import re
from os.path import dirname, abspath
# parent directory to allow access to data folder
d = dirname(dirname(abspath(__file__))) + '/'
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text
#Clean a tweet by removing the url, any stop words or non alphabetic, and any words shorter than 1.
def clean_tweet(tweet):
    tweet = tweet.split('T: ')[1]
    tweet = re.sub(r"http\S+", "", tweet)
    tokens = tweet.split()
    #prepare punctuation regex
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    #clean the punctation
    tokens = [re_punc.sub('', w) for w in tokens]
    return tokens
# load doc, clean and return line of tokens
def doc_to_lines(filename, vocab):
    # load doc
    doc = load_doc(filename)
    tweets = doc.split('\n')
    lines = list()
    #clean the tweets in the file
    for t in tweets:
        if t.startswith('C'):
            cleaned = clean_tweet(t)
            cleaned = [w for w in cleaned if w in vocab]
            # clean a tweet
            lines.append(' '.join(cleaned))
    
    return lines
# function to determine if file name starts with a number less than the last testing file
def check_test(filename, max_file):
    ndx = 0
    while(filename[ndx].isdigit()):
        ndx += 1
    return int(filename[:ndx]) <= max_file
# load all docs in a directory for building a vocabulary
def process_docs(directory, vocab, is_train, max_train):
    lines = list()
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip files that do not have the right extension
        if not filename.endswith(".txt"):
            next
        # create the full path of the file to open
        if is_train and check_test(filename, max_train):
            continue
        if not is_train and not check_test(filename, max_train):
            continue
        path = directory + '/' + filename
        doc_lines = doc_to_lines(path, vocab)
        lines += doc_lines
    return lines
#save vocab to file
def save_data(lines, filename):
    #convert list to single string
    data = '\n'.join(lines)
    #open file
    file = open(filename, 'w')
    #write text
    file.write(data)
    file.close()
def load_clean_dataset(vocab, is_train, max_train):
    #load all reviews
    pos = process_docs(d + 'Preprocessing/positive_tweets', vocab, is_train, max_train)
    neg = process_docs(d + 'Preprocessing/negative_tweets', vocab, is_train, max_train)
    #combine pos and neg
    docs = pos + neg
    labels = array([1 for _ in range(len(pos))] + [0 for _ in range(len(neg))])
    return docs, labels
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer
# define the model
def define_model(n_words):
    # define network
    model = Sequential()
    model.add(Dense(50, input_shape=(n_words,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile network
    model.compile(loss='binary_crossentropy', optimizer='adam',  metrics=['accuracy'])
#   Summarize model
    model.summary()
    plot_model(model, to_file='MCPmodel.png', show_shapes=True)
    return model



def train_model(train_docs, test_docs, mode):
    # encode data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_docs)
    Xtrain = tokenizer.texts_to_matrix(train_docs, mode=mode)
    Xtest = tokenizer.texts_to_matrix(test_docs, mode=mode)
    return Xtrain, Xtest
# define vocab
vocab = load_doc(d + 'Preprocessing/vocab.txt')
vocab = set(vocab.split())
#load the documents
train_docs, ytrain = load_clean_dataset(vocab, True, 114)
test_docs, ytest = load_clean_dataset(vocab, False, 114)
Xtrain, Xtest = train_model(train_docs, test_docs,'freq')
n_words = Xtest.shape[1]
model = define_model(n_words)
#fit network
model.fit(Xtrain, ytrain, epochs=10, verbose=2)