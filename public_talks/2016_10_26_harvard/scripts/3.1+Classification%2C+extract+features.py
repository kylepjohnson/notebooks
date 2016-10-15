
# coding: utf-8

# In[1]:

import datetime as dt
import os
import time

from cltk.corpus.greek.tlg.parse_tlg_indices import get_epithet_of_author
from cltk.corpus.greek.tlg.parse_tlg_indices import get_id_author
import pandas
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# # Make vectorizer

# In[26]:

def stream_lemmatized_files(corpus_dir):
    # return all docs in a dir
    user_dir = os.path.expanduser('~/cltk_data/user_data/' + corpus_dir)
    files = os.listdir(user_dir)

    for file in files:
        filepath = os.path.join(user_dir, file)
        with open(filepath) as fo:
            #TODO rm words less the 3 chars long
            yield file[3:-4], fo.read()


# In[3]:

t0 = dt.datetime.utcnow()

map_id_author = get_id_author()

df = pandas.DataFrame(columns=['id', 'author' 'text', 'epithet'])

for _id, text in stream_lemmatized_files('tlg_lemmatized_no_accents_no_stops'):
    author = map_id_author[_id]
    epithet = get_epithet_of_author(_id)
    df = df.append({'id': _id, 'author': author, 'text': text, 'epithet': epithet}, ignore_index=True)

print(df.shape)
print('... finished in {}'.format(dt.datetime.utcnow() - t0))
print('Number of texts:', len(df))


# In[4]:

text_list = df['text'].tolist()

# make a list of short texts to drop
# For pres, get distributions of words per doc
short_text_drop_index = [index if len(text) > 500 else None for index, text in enumerate(text_list) ]  # ~100 words


# In[5]:

t0 = dt.datetime.utcnow()

# TODO: Consdier using generator to CV http://stackoverflow.com/a/21600406

# time & size counts, w/ 50 texts:
# 0:01:15 & 202M @ ngram_range=(1, 3), min_df=2, max_features=500
# 0:00:26 & 80M @ ngram_range=(1, 2), analyzer='word', min_df=2, max_features=5000
# 0:00:24 & 81M @ ngram_range=(1, 2), analyzer='word', min_df=2, max_features=50000

# time & size counts, w/ 1823 texts:
# 0:02:18 & 46MB @ ngram_range=(1, 1), analyzer='word', min_df=2, max_features=500000
# 0:2:01 & 47 @ ngram_range=(1, 1), analyzer='word', min_df=2, max_features=1000000

# max features in the lemmatized data set: 551428
max_features = 100000
ngrams = 2
vectorizer = TfidfVectorizer(ngram_range=(1, ngrams), analyzer='word', 
                             min_df=2, max_features=max_features)
term_document_matrix = vectorizer.fit_transform(text_list)  # input is a list of strings, 1 per document

# save matrix
vector_fp = os.path.expanduser('~/cltk_data/user_data/tfidf_vectorizer_features{0}_ngrams{1}.pickle'.format(max_features, ngrams))
joblib.dump(term_document_matrix, vector_fp)

print('... finished in {}'.format(dt.datetime.utcnow() - t0))


# # Transform term matrix into feature table

# In[6]:

# Put BoW vectors into a new df
term_document_matrix = joblib.load(vector_fp)  # scipy.sparse.csr.csr_matrix


# In[7]:

term_document_matrix.shape


# In[8]:

term_document_matrix_array = term_document_matrix.toarray() 


# In[9]:

dataframe_bow = pandas.DataFrame(term_document_matrix_array, columns=vectorizer.get_feature_names())


# In[10]:

ids_list = df['id'].tolist()


# In[11]:

len(ids_list)


# In[12]:

#dataframe_bow.shape


# In[13]:

dataframe_bow['id'] = ids_list


# In[14]:

authors_list = df['author'].tolist()
dataframe_bow['author'] = authors_list


# In[15]:

epithets_list = df['epithet'].tolist()
dataframe_bow['epithet'] = epithets_list


# In[16]:

# For pres, give distribution of epithets, including None
#dataframe_bow['epithet']


# In[21]:

t0 = dt.datetime.utcnow()

# removes 334
#! remove rows whose epithet = None
# note on selecting none in pandas: http://stackoverflow.com/a/24489602
dataframe_bow = dataframe_bow[dataframe_bow.epithet.notnull()]
dataframe_bow.shape

print('... finished in {}'.format(dt.datetime.utcnow() - t0))


# In[22]:

t0 = dt.datetime.utcnow()

dataframe_bow.to_csv(os.path.expanduser('~/cltk_data/user_data/tlg_bow.csv'))

print('... finished in {}'.format(dt.datetime.utcnow() - t0))


# In[23]:

print('shape:', dataframe_bow.shape)


# In[24]:

#print(dataframe_bow.head(10))


# In[25]:

# write dataframe_bow to disk, for fast reuse while classifying
# 2.3G
fp_df = os.path.expanduser('~/cltk_data/user_data/tlg_bow_df_tfidf_vectorizer_features{0}_ngrams{1}.pickle'.format(max_features, ngrams))
joblib.dump(dataframe_bow, fp_df)

