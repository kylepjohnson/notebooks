import datetime as dt
import os
import time

from cltk.corpus.greek.tlg.parse_tlg_indices import get_epithet_of_author
from cltk.corpus.greek.tlg.parse_tlg_indices import get_id_author
import pandas
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer


def stream_lemmatized_files(corpus_dir):
    # return all docs in a dir
    user_dir = os.path.expanduser('~/cltk_data/user_data/' + corpus_dir)
    files = os.listdir(user_dir)

    for file in files:
        filepath = os.path.join(user_dir, file)
        with open(filepath) as fo:
            #TODO rm words less the 3 chars long
            yield file[3:-4], fo.read()

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


text_list = df['text'].tolist()

# make a list of short texts to drop
# For pres, get distributions of words per doc
short_text_drop_index = [index if len(text) > 500 else None for index, text in enumerate(text_list) ]  # ~100 words


t0 = dt.datetime.utcnow()

# TODO: Consdier using generator to CV http://stackoverflow.com/a/21600406

# time & size counts, w/ 50 texts:
# 0:01:15 & 202M @ ngram_range=(1, 3), min_df=2, max_features=500
# 0:00:26 & 80M @ ngram_range=(1, 2), analyzer='word', min_df=2, max_features=5000
# 0:00:24 & 81M @ ngram_range=(1, 2), analyzer='word', min_df=2, max_features=50000

# time & size counts, w/ 1823 texts:
# 0:02:18 & 46MB @ ngram_range=(1, 1), analyzer='word', min_df=2, max_features=500000
#  &  @ ngram_range=(1, 2), analyzer='word', min_df=2, max_features=500000

max_features = 500000
ngrams = 2
vectorizer = CountVectorizer(ngram_range=(1, ngrams), analyzer='word',
                             min_df=2, max_features=max_features)

term_document_matrix = vectorizer.fit_transform(text_list)  # input is a list of strings, 1 per document

vector_fp = os.path.expanduser('~/cltk_data/user_data/vectorizer_test_features{0}_ngrams{1}.pickle'.format(max_features, ngrams))
joblib.dump(vectorizer, vector_fp)

print('... finished in {}'.format(dt.datetime.utcnow() - t0))
