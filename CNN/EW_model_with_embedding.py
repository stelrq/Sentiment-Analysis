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
from os.path import dirname, abspath
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
def clean_tweet(tweet, vocab):
    tweet = tweet.split('T: ')[1]
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
    model.add(Conv1D(filters=48, kernel_size=8, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile network
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#    # summarize defined model
#    model.summary()
#    plot_model(model, to_file='model.png', show_shapes=True)
    return model

# integer encode and pad documents
def encode_docs(tokenizer, max_length, docs):
    # integer encode
    encoded = tokenizer.texts_to_sequences(docs)
    # pad sequences
    padded = pad_sequences(encoded, maxlen=max_length, padding='post')
    return padded

# evaluate the sentiment of a tweet as positive or negative
def predict_sentiment(tweet, vocab, tokenizer, model):
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
# define vocab
vocab = load_doc(d + 'Preprocessing/vocab.txt')
vocab = set(vocab.split())
#load the documents
train_docs, ytrain = load_clean_dataset(vocab, True, 114)
#create tokenizer
tokenizer = create_tokenizer(train_docs)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary size: %d'% vocab_size)
# calculate the maximum sequence length
max_length = max([len(s.split()) for s in train_docs])
print('Maximum length: %d'% max_length)
# encode data
Xtrain = encode_docs(tokenizer, max_length, train_docs)
# define model
model = define_model(vocab_size, max_length)

# fit network
model.fit(Xtrain, ytrain, epochs=5, verbose=2)
# save the model
model.save('model5pass48filter.h5')