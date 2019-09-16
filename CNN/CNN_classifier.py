#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 19:24:26 2019

@author: sterling
"""
import string
import re
from os import listdir
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.models import load_model
from keras.layers.convolutional import MaxPooling1D
import json
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
        tokens = clean_tweet(t)
        tokens = ' '.join(tokens)
        tweets.append(tokens)
    return tweets
def process_tweet_docs(directory, vocab):
    lines = list()
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip files that do not have the right extension
        if not filename.endswith(".txt"):
            next
        path = directory + '/' + filename
        line = load_tweet_doc(path)
        lines += line
    return lines
#Clean a tweet by removing the url, any stop words or non alphabetic, and any words shorter than 1.
def clean_tweet(tweet, vocab):
#    tweet = tweet.split('T: ')[1]
    tweet = re.sub(r"http\S+", "", tweet)
    tokens = tweet.split()
    #prepare punctuation regex
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    #clean the punctation
    tokens = [re_punc.sub('', w) for w in tokens]
    tokens = [w for w in tokens if w in vocab]
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
            cleaned = clean_tweet(t, vocab)
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
def define_model(vocab_size, max_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_length))
    model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile network
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize defined model
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True)
    return model

# integer encode and pad documents
def encode_docs(tokenizer, max_length, docs):
    # integer encode
    encoded = tokenizer.texts_to_sequences(docs)
    # pad sequences
    padded = pad_sequences(encoded, maxlen=max_length, padding='post')
    return padded

# evaluate the sentiment of a tweet as positive or negative
def predict_sentiment(tweet, vocab, tokenizer, max_length, model):
    # prep text
    line = clean_tweet(tweet, vocab)
    # encode and pad tweet
    padded = encode_docs(tokenizer, max_length, [line])
    # predict sentiment
    yhat = model.predict(padded, verbose=0)
    # retreive predicted percentage and label
    percent_pos = yhat[0,0]
    if round(percent_pos) == 0:
        return (1-percent_pos), 'NEGATIVE'
    return percent_pos, 'POSITIVE'

#classify tweets and write them to respective files.
def classify_tweets(directory, vocab, tokenizer, max_length, model):
    lines = process_tweet_docs(directory, vocab)
    positive_directory = 'positive_tweets_testing/'
    negative_directory = 'negative_tweets_testing/'
    negative_file = open(negative_directory + '1_negative_tweets.txt', 'w')
    positive_file = open(positive_directory + '1_positive_tweets.txt', 'w')
    positive_counter = 0
    positive_file_count = 1
    negative_counter = 0
    negative_file_count = 1
    for line in lines:
        result = predict_sentiment(line, vocab, tokenizer, max_length, model)
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
vocab = load_doc(d + 'Preprocessing/vocab.txt')
vocab = set(vocab.split())
#load the documents
train_docs, ytrain = load_clean_dataset(vocab, True, 114)
# load the testing data
test_docs, ytest = load_clean_dataset(vocab, False, 114)
# create tokenizer
tokenizer = create_tokenizer(train_docs)
# define vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary size: %d'% vocab_size)
# calculate max sequence length
max_length = max([len(s.split()) for s in train_docs])
print('Maximum length: %d'% max_length)
# encode data
Xtrain = encode_docs(tokenizer, max_length, train_docs)
Xtest = encode_docs(tokenizer, max_length, test_docs)
# load the model
model = load_model('model.h5')
classify_tweets(d + 'Preprocessing/EW_data_new', vocab, tokenizer, max_length, model)