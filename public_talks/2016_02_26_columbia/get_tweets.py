#!/usr/bin/python
# -*- coding: utf_8 -*-

"""Access and query Twitter's API with the simplistic twitter package (`pip install twitter`).
"""

from __future__ import print_function
from __future__ import unicode_literals
import csv
import os
import time

from twitter import OAuth
from twitter import Twitter


def setup_twitter(config_file='config.py'):
    """Setup auth keys and session with Twitter client."""
    config = {}
    execfile(config_file, config)

    twitter_obj = Twitter(auth=OAuth(config["access_key"],
                                 config["access_secret"],
                                 config["consumer_key"],
                                 config["consumer_secret"]))
    return twitter_obj

def search_twitter(twitter_session, query, count=100, status='popular'):
    """Submit query to Twitter API via twitter package."""

    status_options = ['mixed', 'recent', 'popular']
    assert status in status_options, "'status' must be in {}.".format(status_options)

    query = twitter_session.search.tweets(q=query,
                                          lang='en',
                                          result=status,
                                          count=count,
                                          retry=True)

    return query


def parse_twitter_response(twitter_response, min_rts=500, strip_non_ascii=True):
    """Extract requested variables from Twitter API response. Yield each tweet
    one at a time with a generator. Available keys:

    [u'contributors', u'truncated', u'text', u'is_quote_status',
    u'in_reply_to_status_id', u'id', u'favorite_count', u'source',
    u'retweeted', u'coordinates', u'entities', u'in_reply_to_screen_name',
    u'in_reply_to_user_id', u'retweet_count', u'id_str', u'favorited',
    u'retweeted_status', u'user', u'geo', u'in_reply_to_user_id_str',
    u'possibly_sensitive', u'lang', u'created_at',
    u'in_reply_to_status_id_str', u'place', u'metadata']
    """
    for result in twitter_response['statuses']:
        tweet_datetime = result['created_at']

        text = result['text'].encode('utf_8')
        if strip_non_ascii:
            text = ''.join([i if ord(i) < 128 else ' ' for i in text])

        # Strip 'RT ' from head of retweets, redundant
        if text.startswith('RT '):
            text = text[3:]

        # Ch newlines to spaces
        text = ''.join([' ' if c == '\n' else c for c in text])

        rt_count = result['retweet_count']

        yield {'_tweet_datetime': tweet_datetime,
               '_text': text,
               '_rt_count': rt_count}


def search_parse_write_tweets(query_str,
                              total_to_fetch,
                              status,
                              minimum_rts,
                              low_rt_threshold):

    twitter = setup_twitter()
    query_response = search_twitter(twitter_session=twitter,
                                    query=query_disjunction,
                                    count=TWEETS_TO_FETCH,
                                    status=status)

    print("Search complete ({} seconds)".format(query_response["search_metadata"]["completed_in"]))

    tweets_data = parse_twitter_response(query_response, min_rts=minimum_rts)  # yields generator

    fieldnames = []
    if not fieldnames:
        for row in tweets_data:
            fieldnames = row.keys()
            fieldnames_len = len(row.keys())
            break


    # Set up csv writers
    file1 = 'tweets/tweets_popular.csv'
    f1_write_header = False
    if not os.path.isfile(file1):
        f1_write_header = True
    csv_popular_open = open(file1, 'ab')
    csv_popular_writer = csv.DictWriter(csv_popular_open, delimiter=b'|', fieldnames=fieldnames)
    if f1_write_header:
        csv_popular_writer.writeheader()

    file2 = 'tweets/tweets_not_popular.csv'
    f2_write_header = False
    if not os.path.isfile(file2):
        f2_write_header = True
    csv_not_popular_open = open(file2, 'ab')
    csv_not_popular_writer = csv.DictWriter(csv_not_popular_open, delimiter=b'|', fieldnames=fieldnames)
    if f2_write_header:
        csv_not_popular_writer.writeheader()

    # Loop thru generator of dicts, write row to right file
    for tweet_data in tweets_data:
        if tweet_data['rt_count'] >= minimum_rts:
            if len(tweet_data.keys()) == fieldnames_len:
                csv_popular_writer.writerow(tweet_data)
        elif tweet_data['rt_count'] <= low_rt_threshold:
            if len(tweet_data.keys()) == fieldnames_len:
                csv_not_popular_writer.writerow(tweet_data)


if __name__ == '__main__':
    TWEETS_TO_FETCH = 1000
    query_string = 'the a u i me she you he they for rt at tweet'.split(' ')
    query_disjunction = ' OR '.join(query_string)
    #status = 'popular'  # ['mixed', 'recent', 'popular']
    minimum_rts = 500
    low_rt_threshold = 10


    while True:
        time.sleep(60)
        search_parse_write_tweets(query_str=query_disjunction,
                                  total_to_fetch=TWEETS_TO_FETCH,
                                  status='popular',
                                  minimum_rts=minimum_rts,
                                  low_rt_threshold=low_rt_threshold)

        search_parse_write_tweets(query_str=query_disjunction,
                                  total_to_fetch=TWEETS_TO_FETCH,
                                  status='mixed',
                                  minimum_rts=minimum_rts,
                                  low_rt_threshold=low_rt_threshold)
