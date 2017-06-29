
# coding: utf-8

# Following tutorial ["Topic Modeling for Fun and Profit"](http://radimrehurek.com/topic_modeling_tutorial/2%20-%20Topic%20Modeling.html)

# In[31]:


import itertools
import logging
import os
import pickle
import time

from cltk.stop.greek.stops import STOPS_LIST
import gensim
from gensim.corpora.mmcorpus import MmCorpus
from gensim.utils import simple_preprocess
import numpy as np


# In[2]:


logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO  # ipython sometimes messes up the logging setup; restore


# In[3]:


user_dir = os.path.expanduser('~/cltk_data/user_data/lda_tlg/')
try:
    os.makedirs(user_dir)
except FileExistsError:
    pass


# In[4]:


PREPROCESS_DEACCENT = False
STOPS_LIST = [simple_preprocess(stop, deacc=PREPROCESS_DEACCENT)[0] for stop in STOPS_LIST if len(simple_preprocess(stop, deacc=PREPROCESS_DEACCENT)) > 0]
STOPS_LIST = ['τῆϲ', 'τοῖϲ', 'εἰϲ', 'πρὸϲ', 'τοὺϲ', 'τᾶς', 'επι']
STOPS_LIST += ["τηϲ", "τοιϲ", "εϲτι", "προϲ", "ειϲ", "ταϲ", "ωϲ", "τουϲ", "ξυν", 'πρε']  # useful for after rm accents


# In[5]:


TOK_MIN = 3  # rm words shorter than
TOK_MAX = 20  # rm words longer than
DOC_MIN = 50  # drop docs shorter than
def tokenize(text):
    """Tokenize and rm stopwords. The Gensim `simple_preprocess` will work fine
    here becuase the Greek text has already been aggressively cleaned up.
    https://radimrehurek.com/gensim/utils.html#gensim.utils.simple_preprocess
    """
    tokens = [token for token in simple_preprocess(text, deacc=PREPROCESS_DEACCENT, min_len=TOK_MIN, max_len=TOK_MAX)]
    return [token for token in tokens if token not in STOPS_LIST]
    

def iter_tlg(tlg_dir):
    """Stream TLG doc-by-doc."""
    file_names = os.listdir(tlg_dir)
    for file_name in file_names:
        file_path = os.path.join(tlg_dir, file_name)
        with open(file_path) as file_open:
            file_read = file_open.read()
        tokens = tokenize(file_read)
        # ignore very short docs
        # todo: get file length distribution to better know what is short in TLG
        if len(tokens) < DOC_MIN:
            continue
        yield file_name, tokens


# In[6]:


# Take a look at the docs post-processing
# Open corpus iterator
tlg_preprocessed = os.path.expanduser('~/cltk_data/greek/text/tlg/plaintext/')
stream = iter_tlg(tlg_preprocessed)
for title, tokens in itertools.islice(iter_tlg(tlg_preprocessed), 8):
    print(title, tokens[:10])  # print the article title and its first ten tokens


# # Mk word dictionaries

# In[7]:


# Open corpus iterator
doc_stream = (tokens for _, tokens in iter_tlg(tlg_preprocessed))


# In[12]:


no_below = 20
no_above = 0.1


# In[20]:


# store the dictionary, for future reference
dict_name = 'gensim_dict_id2word_tlg_nobelow{}_noabove{}_tokmin{}_tokmax{}_docmin{}_deaccent{}.dict'.format(no_below, 
                                                                                                            no_above, 
                                                                                                            TOK_MIN, 
                                                                                                            TOK_MAX, 
                                                                                                            DOC_MIN, 
                                                                                                            PREPROCESS_DEACCENT)
dict_path = os.path.join(user_dir, dict_name)

try:
    id2word_tlg = gensim.corpora.dictionary.Dictionary.load(dict_path)
except FileNotFoundError:
    t0 = time.time()
    # ~4 min on TLG corpus if rm accents; ~w min if not
    id2word_tlg = gensim.corpora.Dictionary(doc_stream)
    # this cutoff might lose too much info, we'll see
    # ignore words that appear in less than 20 documents or more than 10% documents
    id2word_tlg.filter_extremes(no_below=no_below, no_above=no_above)
    id2word_tlg.save(dict_path)
    print('Time to mk new corpus dictionary:', time.time() - t0)
print(id2word_tlg)


# # Mk vectors
# 
# Now start again with the corpus, turning the actual words into integers from our map.

# In[ ]:


# Illustrate what this BoW space looks like with example doc
doc = "περὶ ποιητικῆς αὐτῆς τε καὶ τῶν εἰδῶν αὐτῆς, ἥν τινα δύναμιν ἕκαστον ἔχει, καὶ πῶς δεῖ συνίστασθαι τοὺς μύθους [10] εἰ μέλλει καλῶς ἕξειν ἡ ποίησις, ἔτι δὲ ἐκ πόσων καὶ ποίων ἐστὶ μορίων, ὁμοίως δὲ καὶ περὶ τῶν ἄλλων ὅσα τῆς αὐτῆς ἐστι μεθόδου, λέγωμεν ἀρξάμενοι κατὰ φύσιν πρῶτον ἀπὸ τῶν πρώτων."
doc = ' '.join(simple_preprocess(doc))
bow = id2word_tlg.doc2bow(tokenize(doc))
print(bow)  # words both in BoW dict and doc
print(id2word_tlg[bow[0][0]])  # map int back to str


# In[ ]:


class TLGCorpus(object):
    def __init__(self, dump_file, dictionary, clip_docs=None):
        """Yield each document in turn, as a list of tokens (unicode strings).
        """
        self.dump_file = dump_file
        self.dictionary = dictionary
        self.clip_docs = clip_docs
    
    def __iter__(self):
        self.titles = []
        for title, tokens in itertools.islice(iter_tlg(self.dump_file), self.clip_docs):
            self.titles.append(title)
            yield self.dictionary.doc2bow(tokens)
    
    def __len__(self):
        return self.clip_docs


# In[ ]:


# make the BoW corpus
# creates a stream of bag-of-words vectors
corpus_bow_tlg = TLGCorpus(tlg_preprocessed, id2word_tlg)

# reduce corpus size for faster testing
# corpus_bow_tlg = gensim.utils.ClippedCorpus(corpus_bow_tlg, 100)

# vector = next(iter(corpus_bow_tlg))
# print(vector)  # print the first vector in the stream
# [(0, 1), (1, 1), (2, 1), ...]

# # what is the most common word in that first article?
# most_index, most_count = max(vector, key=lambda _tuple: _tuple[1])
# print(id2word_tlg[most_index], most_count)  # μιλησιοις 2


# In[ ]:


# Save BoW
# ~4 min on TLG corpus
bow_name = 'gensim_bow_tlg_nobelow{}_noabove{}_tokmin{}_tokmax{}_docmin{}_deaccent{}.mm'.format(no_below, 
                                                                                                no_above, 
                                                                                                TOK_MIN, 
                                                                                                TOK_MAX, 
                                                                                                DOC_MIN, 
                                                                                                PREPROCESS_DEACCENT)
bow_path = os.path.join(user_dir, bow_name)
t0 = time.time()
gensim.corpora.MmCorpus.serialize(bow_path, corpus_bow_tlg)
print('Time to save BoW space:', time.time() - t0)

# Later load saved corpus with:
# corpus_bow_tlg = gensim.corpora.MmCorpus(bow_path)


# # LDA transformation

# In[10]:


# Quick testing using just a part of the corpus

NUM_TOPICS_LIST = [5, 10, 20, 40, 60, 120]
PASSES = 100


# In[ ]:


for num_topics in NUM_TOPICS_LIST:
    print('Beginning training ...')
    print('... {} topics and {} passes ...'.format(num_topics, PASSES))
    t0 = time.time()
    lda_model = gensim.models.LdaMulticore(corpus_bow_tlg, num_topics=num_topics, id2word=id2word_tlg, passes=PASSES)
    
    # save LDA vector space
    lda_space_name = 'gensim_lda_space_tlg_numtopics{}_numpasses{}_nobelow{}_noabove{}_tokmin{}_tokmax{}_docmin{}_deaccent{}.mm'.format(num_topics, 
                                                                                                                                        PASSES, 
                                                                                                                                        no_below, 
                                                                                                                                        no_above, 
                                                                                                                                        TOK_MIN, 
                                                                                                                                        TOK_MAX, 
                                                                                                                                        DOC_MIN, 
                                                                                                                                        PREPROCESS_DEACCENT)
    path_lda = os.path.join(user_dir, lda_space_name)
    gensim.corpora.MmCorpus.serialize(path_lda, lda_model[corpus_bow_tlg])
    
    # save model
    lda_model_name = 'gensim_lda_model_tlg_numtopics{}_numpasses{}_nobelow{}_noabove{}_tokmin{}_tokmax{}_docmin{}_deaccent{}.model'.format(num_topics, 
                                                                                                                                           PASSES, 
                                                                                                                                           no_below, 
                                                                                                                                           no_above, 
                                                                                                                                           TOK_MIN, 
                                                                                                                                           TOK_MAX, 
                                                                                                                                           DOC_MIN, 
                                                                                                                                           PREPROCESS_DEACCENT)
    path_lda = os.path.join(user_dir, lda_model_name)
    lda_model.save(path_lda)
    print('Time to train LDA model space:', time.time() - t0)


# In[ ]:


# # Examples of how to use the model
# lda_model.print_topics(-1)  # print a few most important words for each LDA topic
# # transform text into the bag-of-words space
# bow_vector = id2word_tlg.doc2bow(tokenize(doc))
# print([(id2word_tlg[id], count) for id, count in bow_vector])

# # transform into LDA space
# lda_vector = lda_model[bow_vector]
# print(lda_vector)

# # print the document's single most prominent LDA topic
# print(lda_model.print_topic(max(lda_vector, key=lambda item: item[1])[0]))


# # Evaluation

# ## Word intrusion
# 
# > For each trained topic, they take its first ten words, then substitute one of them with another, randomly chosen word (intruder!) and see whether a human can reliably tell which one it was. If so, the trained topic is topically coherent (good); if not, the topic has no discernible theme (bad)

# In[13]:


for num_topics in NUM_TOPICS_LIST:
    # load model
    lda_model_name = 'gensim_lda_model_tlg_numtopics{}_numpasses{}_nobelow{}_noabove{}_tokmin{}_tokmax{}_docmin{}_deaccent{}.model'.format(num_topics, 
                                                                                                                                           PASSES, 
                                                                                                                                           no_below, 
                                                                                                                                           no_above, 
                                                                                                                                           TOK_MIN, 
                                                                                                                                           TOK_MAX, 
                                                                                                                                           DOC_MIN, 
                                                                                                                                           PREPROCESS_DEACCENT)
    print('Loading model: {} ...'.format(lda_model_name))
    print('... for word intrusion testing ...')
    path_lda = os.path.join(user_dir, lda_model_name)
    lda_model = gensim.models.LdaMulticore.load(path_lda)
    
    # select top 50 words for each of the LDA topics
    print('Top 50 words of each LDA model:')
    top_words = [[word for word, _ in lda_model.show_topic(topicno, topn=50)] for topicno in range(lda_model.num_topics)]
    print(top_words)
    print('')

    # get all top 50 words in all 20 topics, as one large set
    all_words = set(itertools.chain.from_iterable(top_words))
    print("Can you spot the misplaced word in each topic?")

    # for each topic, replace a word at a different index, to make it more interesting
    replace_index = np.random.randint(0, 10, lda_model.num_topics)

    replacements = []
    for topicno, words in enumerate(top_words):
        other_words = all_words.difference(words)
        replacement = np.random.choice(list(other_words))
        replacements.append((words[replace_index[topicno]], replacement))
        words[replace_index[topicno]] = replacement
        print("%i: %s" % (topicno, ' '.join(words[:10])))
    
    print("Actual replacements were:")
    print(list(enumerate(replacements)))
    print('')


# ## Split doc
# 
# > We'll split each document into two parts, and check that 1) topics of the first half are similar to topics of the second 2) halves of different documents are mostly dissimilar

# In[17]:


# evaluate on 1k documents **not** used in LDA training
tlg_preprocessed = os.path.expanduser('~/cltk_data/greek/text/tlg/plaintext/')
doc_stream = (tokens for _, tokens in iter_tlg(tlg_preprocessed))  # generator
test_docs = list(itertools.islice(doc_stream, 100, 200))  # ['πανυ', 'καλως', ...], [...], ...]


# In[18]:


def intra_inter(model, test_docs, num_pairs=10000):
    # split each test document into two halves and compute topics for each half
    part1 = [model[id2word_tlg.doc2bow(tokens[: len(tokens) // 2])] for tokens in test_docs]
    part2 = [model[id2word_tlg.doc2bow(tokens[len(tokens) // 2 :])] for tokens in test_docs]
    
    # print computed similarities (uses cossim)
    print("average cosine similarity between corresponding parts (higher is better):")
    print(np.mean([gensim.matutils.cossim(p1, p2) for p1, p2 in zip(part1, part2)]))

    random_pairs = np.random.randint(0, len(test_docs), size=(num_pairs, 2))
    print("average cosine similarity between {} random parts (lower is better):".format(num_pairs))    
    print(np.mean([gensim.matutils.cossim(part1[i[0]], part2[i[1]]) for i in random_pairs]))


# In[21]:


for num_topics in NUM_TOPICS_LIST:
    # load model
    lda_model_name = 'gensim_lda_model_tlg_numtopics{}_numpasses{}_nobelow{}_noabove{}_tokmin{}_tokmax{}_docmin{}_deaccent{}.model'.format(num_topics, 
                                                                                                                                           PASSES, 
                                                                                                                                           no_below, 
                                                                                                                                           no_above, 
                                                                                                                                           TOK_MIN, 
                                                                                                                                           TOK_MAX, 
                                                                                                                                           DOC_MIN, 
                                                                                                                                           PREPROCESS_DEACCENT)
    print('Loading model: {} ...'.format(lda_model_name))
    print('... for testing split document topic matching ...')
    path_lda = os.path.join(user_dir, lda_model_name)
    lda_model = gensim.models.LdaMulticore.load(path_lda)

    print("LDA results:")
    intra_inter(lda_model, test_docs)
    print('')


# # Transform all docs

# In[30]:


for num_topics in NUM_TOPICS_LIST:
    print('num topics', num_topics)
    # load model
    lda_model_name = 'gensim_lda_model_tlg_numtopics{}_numpasses{}_nobelow{}_noabove{}_tokmin{}_tokmax{}_docmin{}_deaccent{}.model'.format(num_topics, 
                                                                                                                                           PASSES, 
                                                                                                                                           no_below, 
                                                                                                                                           no_above, 
                                                                                                                                           TOK_MIN, 
                                                                                                                                           TOK_MAX, 
                                                                                                                                           DOC_MIN, 
                                                                                                                                           PREPROCESS_DEACCENT)
    print('Loading model: {} ...'.format(lda_model_name))
    print('... scoring topics of all TLG documents ...')
    path_lda = os.path.join(user_dir, lda_model_name)
    lda_model = gensim.models.LdaMulticore.load(path_lda)

    # mk save path name
    scores_name = lda_model_name.rstrip('.model') + '.scores'
    scores_path = os.path.join(user_dir, scores_name)
    doc_topics = ''
    for title, tokens in iter_tlg(tlg_preprocessed):
        #print(title, tokens[:10])  # print the article title and its first ten tokens
        # print(title)
        topic_distribution = str(lda_model[id2word_tlg.doc2bow(tokens)])
        # print(topic_distribution)
        doc_topics += 'title: ' + title + '\n'
        doc_topics += topic_distribution + '\n\n'
    with open(scores_path, 'w') as file_open:
        file_open.write(doc_topics)
    print('')


sys.exit()
# # Visualization
# 
# Following: http://nbviewer.jupyter.org/github/bmabey/pyLDAvis/blob/master/notebooks/pyLDAvis_overview.ipynb

# In[ ]:


lda_model.show_topics()


# In[ ]:


import pyLDAvis.gensim

pyLDAvis.enable_notebook()


# In[ ]:


pyLDAvis.gensim.prepare(lda_model, corpus_bow_tlg, id2word_tlg)


# In[ ]:




