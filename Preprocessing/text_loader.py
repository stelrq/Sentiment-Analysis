#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 12:01:46 2019

@author: sterling
"""
import re
import string
# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    lines = file.readlines()
    # close the file
    file.close()
    return lines
#Extra words and seperate them into lists according to pos, neg, neut.
def clean_doc(doc_lines):
    #skip commented lines
    start = 0
    while doc_lines[start][0] == '#':
        start += 1
    pos_words = list()
    neg_words = list()
    neutral_words = list()
    all_words = list()
    for i in range(start, len(doc_lines)):
        line = doc_lines[i].split('#')
        word = line[0]
        rating = line[1].split('\t')[1]
        rating = float(rating.split('\n')[0])
        re_punc = re.compile('[%s]' % re.escape(string.punctuation))
        #clean the punctation
        word = re_punc.sub('', word)
        if len(word) > 1:
            all_words.append(word)
            if rating > 0:
                pos_words.append(word)
            elif rating < 0:
                neg_words.append(word)
            else:
                neutral_words.append(word)
    return pos_words, neg_words, neutral_words, all_words
def save_list(lines, filename):
    data = ('\n'.join(lines))
    file = open(filename, 'w')
    file.write(data)
    file.close()

# load annotated vocabulary
vocab_filename = 'SentiWords_1.1.txt'
vocab = load_doc(vocab_filename)
vocab = clean_doc(vocab)
positive_words = vocab[0]
negative_words = vocab[1]
neutral_words = vocab[2]
all_words = vocab[3]
#save the processed words
save_list (negative_words, 'negative.txt')
save_list(positive_words, 'positive.txt')
save_list(neutral_words, 'neutral.txt')
save_list(all_words, 'all.txt')