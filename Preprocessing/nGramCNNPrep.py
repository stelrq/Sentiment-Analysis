#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 19:24:26 2019

@author: sterling
"""
import string
import re
from pickle import dump
from os import listdir
from numpy import array
from nltk.corpus import stopwords
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
    #clean out non words
    tokens = [word for word in tokens if word.isalpha()]
    #remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if not word in stop_words]
    #remove short words
    tokens = [w for w in tokens if len(w) > 1]
    return tokens
# load doc, clean and return line of tokens
def doc_to_lines(filename):
    # load doc
    doc = load_doc(filename)
    tweets = doc.split('\n')
    lines = list()
    #clean the tweets in the file
    for t in tweets:
        if t.startswith('C'):
            cleaned = clean_tweet(t)
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
def process_docs(directory, is_train, max_train):
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
        doc_lines = doc_to_lines(path)
        lines += doc_lines
    return lines
def load_clean_dataset(is_train, max_train):
    #load all reviews
    pos = process_docs('positive_tweets', is_train, max_train)
    neg = process_docs('negative_tweets', is_train, max_train)
    #combine pos and neg
    docs = pos + neg
    labels = array([1 for _ in range(len(pos))] + [0 for _ in range(len(neg))])
    return docs, labels
# save a dataset to file
def save_dataset(dataset, filename):
    dump(dataset, open(filename, 'wb'))
    print('Saved: %s'% filename)

#load the documents
train_docs, ytrain = load_clean_dataset(True, 114)
# load the testing data
test_docs, ytest = load_clean_dataset(False, 114)
destination = d + 'nGramCNN/'
# save training datasets
save_dataset([train_docs, ytrain], destination + 'ngram_train.pk1')
save_dataset([test_docs, ytest], destination + 'ngram_test.pk1')