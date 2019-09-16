#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 15:07:34 2019

@author: sterling
"""
import json
with open('EW_data/2EW_tweets.txt') as infile:
    mydata = json.load(infile)
print(mydata['statuses'][-1]['id'])

def check_test(filename, max_file):
    ndx = 0
    while(filename[ndx].isdigit()):
        ndx += 1
    return int(filename[:ndx]) <= max_file
print(check_test('100_positive_tweets.txt', 114))
print(check_test('564_positive_tweets.txt', 114))                    