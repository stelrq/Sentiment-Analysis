#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 20:48:27 2019

@author: sterling
"""

import tweepy
import time
import json
#Enter authorizations
consumer_key = 'jyjfLM9rzwimmE0xw30Tm5Dqq' 
consumer_secret = 'Fptyjcggr1e2nnMOOUWKBMbdgrqnYjyscRYePo6h3D2a9EVvhZ'
access_key = '1027293645676769280-jO1P77gG9NsJ0KJCg9THwLBjyy6DNS'
access_secret = '5pvsK3oxrrXn8oeqUVYkyQQt0lLHCoxSA8yBlKa2ZhR6c'
#Set up your authorisations
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
#SetupAPI call 
api = tweepy.API(auth, parser = tweepy.parsers.JSONParser())
def getTweets(directory, searchquery, desired_number, id_start):
    data = api.search(q = searchquery, count = 25, lang = 'en', result_type =
'mixed', max_id = id_start, tweet_mode = 'extended')
    files = 801
    new_files = 1
    with open(directory + str(files) + 'EW_tweets.txt', 'w') as outfile:
            json.dump(data, outfile)
    while (new_files <= desired_number/25):
        time.sleep(5)
        last = data['statuses'][-1]['id']
        data = api.search(q = searchquery, count = 25, lang = 'en', result_type = 'mixed', max_id = last, tweet_mode = 'extended')
        files += 1
        new_files += 1
        with open(directory + str(files) + 'EW_tweets.txt', 'w') as outfile:
            json.dump(data, outfile)
def getNewTweets(directory, searchquery, desired_number, id_start, last_file):
    data = api.search(q = searchquery, count = 25, lang = 'en', result_type =
'mixed', since_id = id_start, tweet_mode = 'extended')
    files = last_file + 1
    last = data['statuses'][-1]['id']
    new_tweets = len(data['statuses'])
    with open(directory + str(files) + 'EW_tweets.txt', 'w') as outfile:
            json.dump(data, outfile)
    while (new_tweets <= desired_number and last > id_start):
        time.sleep(5)
        data = api.search(q = searchquery, count = 25, lang = 'en', result_type = 'mixed', max_id = last, tweet_mode = 'extended')
        last = data['statuses'][-1]['id']
        files += 1
        new_tweets += len(data['statuses'])
        with open(directory + str(files) + 'EW_tweets.txt', 'w') as outfile:
            json.dump(data, outfile)
#Set search query 
searchquery = '"Elizabeth Warren" -filter:retweets'
#location = 'Preprocessing/EW_data/'
#getTweets(location, searchquery, 100, 1158859804904837120)
location2 = 'Preprocessing/EW_data_new/'
getNewTweets(location2, searchquery, 100, 1161338149936590849, 0)
#searchquery = '"Joe Biden" -filter:retweets'
#location = 'JB_data/'
#getTweets(location, searchquery, 20000)