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
import json
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text
def load_tweet_doc(filename):
    #open the file as json
    with open(filename) as infile:
        mydata = json.load(infile)
    reviews = [r['full_text'] for r in mydata['statuses']]
    return reviews
def tweet_doc_to_lines(filename, vocab):
#    load doc of tweets
    doc = load_tweet_doc(filename)
    tweets = list()
#    clean tweets
    for t in doc:
        tokens = clean_doc(t)
        tokens = [w for w in tokens if w in vocab]
        tokens = ' '.join(tokens)
        tweets.append(tokens)
    return tweets
# load doc, clean and return line of tokens
def doc_to_line(filename, vocab):
    # load the doc
    doc = load_doc(filename)
    # clean doc
    tokens = clean_doc(doc)
    # filter by vocab
    tokens = [w for w in tokens if w in vocab]
    return ' '.join(tokens)
def clean_doc(doc):
    tokens = doc.split()
    #prepare punctuation regex
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    #clean the punctation
    tokens = [re_punc.sub('', w) for w in tokens]
    #clean out non words
    tokens = [word for word in tokens if word.isalpha()]
    #remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if not word in stop_words]
    #remove short words
    tokens = [w for w in tokens if len(w) > 1]
    return tokens
# load all docs in a directory for building a vocabulary
def process_docs(directory, vocab, is_train):
    lines = list()
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip files that do not have the right extension
        if not filename.endswith(".txt"):
            next
#        # create the full path of the file to open
#        if is_train and filename.startswith("cv9"):
#            continue
#        if not is_train and not filename.startswith("cv9"):
#            continue
        path = directory + '/' + filename
        line = doc_to_line(path, vocab)
        lines.append(line)
    return lines
def process_tweet_docs(directory, vocab, is_train):
    lines = list()
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip files that do not have the right extension
        if not filename.endswith(".txt"):
            next
#        # create the full path of the file to open
#        if is_train and filename.startswith("cv9"):
#            continue
#        if not is_train and not filename.startswith("cv9"):
#            continue
        path = directory + '/' + filename
        line = load_tweet_doc(path)
        lines += line
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
#load data to train model, assumed already pruned to match vocab of classification set
def load_bag_of_words():
    #load all reviews
    pos = load_doc('positive.txt').split()
    neg = load_doc('negative.txt').split()
    #combine pos and neg
    docs = pos + neg
    labels = array([1 for _ in range(len(pos))] + [0 for _ in range(len(neg))])
    return docs, labels
def create_Tokenizer(lines):
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
    return model



def train_model(train_docs, mode):
    # encode data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_docs)
    Xtrain = tokenizer.texts_to_matrix(train_docs, mode=mode)
    return Xtrain
def evaluate_mode(Xtrain, Xtest, ytrain, ytest):
    scores = list();
    n_words = Xtest.shape[1]
    n_runs = 10
    for i in range(n_runs):
        #initialize model
        model = define_model(n_words)
        # fit network
        model.fit(Xtrain, ytrain, epochs=10, verbose=0)
        # evaluate
        loss, acc = model.evaluate(Xtest, ytest, verbose=0)
        scores.append(acc)
        print('%d accuracy: %s'% ((i+1), acc))
    return(scores)
def predict_sentiment(review, vocab, tokenizer, model):
    #prep text
    tokens = clean_doc(review)
    tokens = [w for w in tokens if w in vocab]
    #convert data to processable line
    line = ' '.join(tokens)
    encoded = tokenizer.texts_to_matrix([line], mode='binary')
    yhat = model.predict(encoded, verbose=0)
    percent_pos = yhat[0,0]
    print(yhat)
    if round(percent_pos) == 0:
        return (1-percent_pos), 'NEGATIVE'
    return percent_pos, 'POSITIVE'
#classify tweets and write them to respective files.
def classify_tweets(directory, vocab, tokenizer, model):
    lines = process_tweet_docs(directory, vocab, True)
    positive_directory = 'positive_tweets/'
    negative_directory = 'negative_tweets/'
    negative_file = open(negative_directory + '1_negative_tweets.txt', 'w')
    positive_file = open(positive_directory + '1_positive_tweets.txt', 'w')
    positive_counter = 0
    positive_file_count = 1
    negative_counter = 0
    negative_file_count = 1
    for line in lines:
        result = predict_sentiment(line, vocab, tokenizer, model)
        line = line.replace('\n', ' ')
        if result[1] == 'NEGATIVE':
            negative_file.write('C: (%.3f%%) T: %s\n'% (result[0], line))
            negative_counter += 1
            if negative_counter == 25:
                negative_file.close()
                negative_file_count += 1
                negative_file = open(negative_directory + str(negative_file_count) + '_negative_tweets.txt', 'w')
                negative_counter = 0
        elif result[1] == 'POSITIVE':
            positive_file.write('C: (%.3f%%) T: %s\n'% (result[0], line))
            positive_counter += 1
            if positive_counter == 25:
                positive_file.close()
                positive_file_count += 1
                positive_file = open(positive_directory + str(positive_file_count) + '_positive_tweets.txt', 'w')
                positive_counter = 0
    positive_file.close()
    negative_file.close()
# define vocab
vocab = load_doc('vocab.txt')
vocab = set(vocab.split())
#load the documents
train_docs, ytrain = load_bag_of_words()
tokenizer = create_Tokenizer(train_docs)
Xtrain = train_model(train_docs, 'binary')
n_words = Xtrain.shape[1]
model = define_model(n_words)
model.fit(Xtrain, ytrain, epochs=10, verbose=2)
plot_model(model, to_file='model.png', show_shapes=True)           
# test positive text
#text = load_tweet_doc('EW_data/2EW_tweets.txt')
#for r in text:
#    percent, sentiment = predict_sentiment(r, vocab, tokenizer, model)
#    print('Review: [%s]\nSentiment: %s (%.3f%%)'% (r, sentiment, percent*100))
classify_tweets('EW_data', vocab, tokenizer, model)