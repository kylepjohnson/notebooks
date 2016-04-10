
# coding: utf-8

# Script to demo scikit for tweet popular/unpopular classification.

# In[1]:

from __future__ import division
from __future__ import print_function
import csv
import datetime as dt
import os
import platform
import sys

import numpy as np
import pandas
from sklearn import preprocessing
from sklearn import svm
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report


# In[2]:

def csv_to_dict_cesar(csv_filename):
    # Let's say, We are intersted in only count features
    count_features = ['_char_count', '_hashtag_count', '_word_count', '_url_count']
    with open(csv_filename) as f:
        features = [({k: int(v) for k, v in row.items() if k in count_features}, row['_popular'])
                    for row in csv.DictReader(f, skipinitialspace=True)]
        X = [f[0] for f in features]
        Y = [f[1] for f in features]
    return (X, Y)


# In[3]:

def csv_to_dict(csv_filename):
    """Open feature table with csv library.
    
    Task: Run with '_rt_count'. See the good results!
    """
    non_numeric_features = ['', '_text', '_urls', '_mentions', '_hashtags', 
                            '_tweet_datetime', '_popular', '_rt_count']
    with open(csv_filename, 'rU') as f:
        rows = csv.DictReader(f, skipinitialspace=True, delimiter='|')
        labels = [row['_popular'] for row in rows]

    features = []
    with open(csv_filename, 'rU') as f:
        rows = csv.DictReader(f, skipinitialspace=True, delimiter='|')
        for row in rows:
            #print(row)
            row_dict = {}
            for k, v in row.items():
                if k not in non_numeric_features:
                    try:
                        row_dict[k] = int(v)
                    # these tries catch a few junk entries
                    except TypeError:
                        row_dict[k] = 0
                    except ValueError:
                        row_dict[k] = 0

            #row_dict = {k: int(v) for k, v in row.items() if k not in non_numeric_features}
            features.append(row_dict)                

    return features, labels


# In[4]:

def csv_to_df(csv_file):
    """Open csv with Pandas DataFrame, then convert to dict 
    and return.
    
    TODO: Fix this.
    """
    
    dataframe = pandas.read_csv(csv_file, 
                                encoding='utf-8', 
                                engine='python', 
                                sep='|',
                                delimiter='|',
                                index_col=0)
    return dataframe


# In[5]:

def train(csv_filename):
    
    print('Loading CSV into dict ...')
    t0 = dt.datetime.utcnow()
    data, target = csv_to_dict(csv_filename)
    print('... finished in {} secs.'.format(dt.datetime.utcnow() - t0))
    print()
    
    print('Loading dict into vectorizer')
    t0 = dt.datetime.utcnow()
    vec = DictVectorizer()
    X = vec.fit_transform(data).toarray()  # change to numpy array
    Y = np.array(target)  # change to numpy array
    print('... finished in {} secs.'.format(dt.datetime.utcnow() - t0))
    print()

    
    '''
    -In case we need to know the features
    '''
    feature_names = vec.get_feature_names()

    '''
    -Dividing the data into train and test
    -random_state is pseudo-random number generator state used for
     random sampling
    '''
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
    
    # write models dir if not present
    models_dir = 'models'
    if not os.path.isdir(models_dir):
        os.mkdir(models_dir)

    '''
    -PREPOCESSING 
    -Here, scaled data has zero mean and unit varience
    -We save the scaler to later use with testing/prediction data
    '''
    print('Scaling data ...')
    t0 = dt.datetime.utcnow()
    scaler = preprocessing.StandardScaler().fit(X_train)
    joblib.dump(scaler, 'models/scaler.pickle')
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print('... finished in {} secs.'.format(dt.datetime.utcnow() - t0))
    print()
    

    '''
    -This is where we define the models
    -Here, I use SVM and Decision tree with pre-defined parameters
    -We can learn these parameters given our data
    '''
    print('Defining and fitting models ...')
    t0 = dt.datetime.utcnow()   
    clf0 = svm.LinearSVC(C=100.)
    clf1 = tree.DecisionTreeClassifier()

    clf0.fit(X_train_scaled, Y_train)
    clf1.fit(X_train_scaled, Y_train)

    joblib.dump(clf0, 'models/svc.pickle')
    joblib.dump(clf1, 'models/tree.pickle')

    print('... finished in {} secs.'.format(dt.datetime.utcnow() - t0))
    print()
    

    Y_prediction_svc = clf0.predict(X_test_scaled)
    print('svc_predictions ', Y_prediction_svc)
    Y_prediction_tree = clf1.predict(X_test_scaled)
    print('tree_predictions ', Y_prediction_tree)
    expected = Y_test
    print('actual_values   ', expected)
    print()

    
    '''
    Classifiation metrics
    (Case 1): SVMs
    '''
    print()
    print('----Linear SVC_report--------------------------')
    print(classification_report(expected, Y_prediction_svc))

    '''
    Classification metrics
    (case 2): Decision tree
    '''
    print()
    print('----Tree_report--------------------------------')
    print(classification_report(expected, Y_prediction_tree))


# In[ ]:

train("feature_tables/all.csv")


# In[ ]:



