#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 12:01:46 2019

@author: sterling
"""
from os import listdir
from nltk.corpus import stopwords
import string
import re
from collections import Counter
import json
# load labeled vocab docs into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    words = file.read().split()
    # close the file
    file.close()
    return words
labeled_vocab = load_doc('all.txt')
print(labeled_vocab)
labeled_vocab = set(labeled_vocab)
print(labeled_vocab)
# load doc into memory
def load_tweet_doc(filename):
    #open the file as json
    with open(filename) as infile:
        mydata = json.load(infile)
    reviews = [r['full_text'] for r in mydata['statuses']]
    return reviews
#clean an individual tweet
def clean_tweet(tweet):
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
    tokens = [word for word in tokens if word in labeled_vocab]
    #remove short words
    tokens = [w for w in tokens if len(w) > 1]
    return tokens
#clean a set of tweets
def clean_doc(doc):
    cleaned = list()
    for t in doc:
        cleaned.append(clean_tweet(t))
    return cleaned
        
# load doc and add to vocab
def add_doc_to_vocab(filename, vocab):
    # load doc
    doc = load_tweet_doc(filename)
    # clean doc
    tweets = clean_doc(doc)
    # update counts
    for t in tweets:
        vocab.update(t)

def save_list(lines, filename):
    data = ('\n'.join(lines))
    file = open(filename, 'w')
    file.write(data)
    file.close()

# load doc, clean and return line of tokens
def doc_to_line(filename, vocab):
    
    # load the doc
    doc = load_tweet_doc(filename)
    # clean doc
    tokens = clean_doc(doc)
    # filter by vocab
    tokens = [w for w in tokens if w in vocab]
    return ' '.join(tokens)
# load all docs in a directory for building a vocabulary
def process_docs(directory, vocab):
    lines = list()
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip files that do not have the right extension
        if not filename.endswith(".txt"):
            next
        # create the full path of the file to open
        path = directory + '/' + filename
        # add to list
        add_doc_to_vocab(path, vocab)
    return lines
#rewrites file with only valid words
def prune_annotated(tokens, filename):
    my_words = set(tokens)
    to_prune = load_doc(filename)
    pruned = [w for w in to_prune if w in my_words]
    save_list(pruned, filename)
# specify directory to load
directory_EW = 'EW_data'
vocab = Counter()
process_docs(directory_EW, vocab)
# print the size of the vocab
print(len(vocab))
# print the top words in the vocab
print(vocab.most_common(50))
# keep tokens with > 2 occurrence
min_occurrence = 2
tokens = [k for k,c in vocab.items() if c >= min_occurrence]
print(len(tokens))
# save tokens to a vocabulary file
save_list(tokens, 'vocab.txt')
prune_annotated(tokens, 'positive.txt')
prune_annotated(tokens, 'negative.txt')
prune_annotated(tokens, 'neutral.txt')
