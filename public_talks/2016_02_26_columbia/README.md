# About

Notebooks for a lecture to Postdocs at [NYC Ascent](http://www.nycascent.org/) by Cesar Koirala, Kyle P. Johnson, and Ken Bame, on 26 February 2016. These focus on some fundamentals of natural language processing (NLP) and how to leverage machine learning for insights into human language.


# Setup

These were created on Windows 7 with the multi-platform Anaconda distribution.

## Software setup
1. Install [Anaconda](https://www.continuum.io/downloads) for your OS.
1. install pandas (from the Anaconda Prompt, `conda install pandas`)
1. Install scikit-learn (`conda install scikit-learn`)

## Get lecture code
1. Install [Git](https://git-scm.com/downloads)
1. With a terminal app (on Windows, Git Bash is strongly recommended), fetch this repo's source: `git clone https://github.com/kylepjohnson/nyc_postdoc_lecture.git`
1. Change into the repo (`cd lecture_nyc_ascent`) and start the Jupyter notebook (`jupyter notebook`)


# Contents

The folder `tweets` contains two .csv files, one of popular tweets (more than 500 retweets) and another on unpopular tweets (with fewer than 10 retweets). These were obtained with the script `get_tweets.py`. To use this file, you will need to [obtain authentication tokens](https://apps.twitter.com/) and add them to `config.py`.

`tweets_to_features.ipynb` is a Jupyter notebook which illustrates some NLP basics (e.g., tokenization, stopword filter) and also shows how to extract features from text (e.g., bag of words). When you run all of its commands, it will create a diectory `feature_tables` which keep several feature tables for the tweets.

The `code_snippets` directory has some simplified code which serve as easy-to-understand examples of what appear in the other notebooks.
