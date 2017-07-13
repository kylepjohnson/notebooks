import itertools
import os
import string

from gensim.utils import simple_preprocess

from cltk.corpus.utils.formatter import cltk_normalize
from cltk.stop.greek.stops import STOPS_LIST as STOPS_LIST_GRK


# configs for all notebooks
working_dir = os.path.expanduser('~/cltk_data/user_data/lda_1kgreek/')
PREPROCESS_DEACCENT = False
TOK_MIN = 3  # rm words shorter than
TOK_MAX = 20  # rm words longer than
DOC_MIN = 50  # drop docs shorter than
remove_ascii = True
no_below = 20
no_above = 0.1


STOPS_LIST_GRK = [simple_preprocess(stop, deacc=PREPROCESS_DEACCENT)[0] for stop in STOPS_LIST_GRK if len(simple_preprocess(stop, deacc=PREPROCESS_DEACCENT)) > 0]
STOPS_LIST_GRK = ['τῆϲ', 'τοῖϲ', 'εἰϲ', 'πρὸϲ', 'τοὺϲ']
STOPS_LIST_GRK += ["τηϲ", "τοιϲ", "εϲτι", "προϲ", "ειϲ", "ταϲ", "ωϲ", "τουϲ", "ξυν", 'ξὺν', 'πρε', 'ἀλλ']  # useful for after rm accents
STOPS_LIST = [cltk_normalize(stop) for stop in STOPS_LIST_GRK]

ascii_str = string.ascii_letters + string.punctuation + string.digits


def mk_working_dir(fp):
    """Make dir if not exists."""
    user_dir = os.path.expanduser(fp)
    try:
        os.makedirs(user_dir)
    except FileExistsError:
        pass


def tokenize(text, rm_ascii=False):
    """Tokenize and rm stopwords. The Gensim `simple_preprocess` will work fine
    here becuase the Greek text has already been aggressively cleaned up.
    https://radimrehurek.com/gensim/utils.html#gensim.utils.simple_preprocess
    """
    if rm_ascii:
        text = [char for char in text if char not in ascii_str]
        text = ''.join(text)
    tokens = [token for token in simple_preprocess(text, deacc=PREPROCESS_DEACCENT, min_len=TOK_MIN, max_len=TOK_MAX)]
    return [token for token in tokens if token not in STOPS_LIST]
    

def iter_docs(docs_dir, rm_ascii=False):
    """Stream files in a dir (TLG, TEI, etc.) doc-by-doc."""
    file_names = os.listdir(docs_dir)
    for file_name in file_names:
        file_path = os.path.join(docs_dir, file_name)
        with open(file_path) as file_open:
            file_read = file_open.read()
        tokens = tokenize(file_read, rm_ascii=rm_ascii)
        tokens = [cltk_normalize(token) for token in tokens]
        # ignore very short docs
        # todo: get file length distribution to better know what is short in TLG
        if len(tokens) < DOC_MIN:
            continue
        yield file_name, tokens


class GenerateCorpus(object):
    def __init__(self, dump_file, dictionary, clip_docs=None):
        """Yield each document in turn, as a list of tokens (unicode strings).
        """
        self.dump_file = dump_file
        self.dictionary = dictionary
        self.clip_docs = clip_docs
    
    def __iter__(self):
        self.titles = []
        for title, tokens in itertools.islice(iter_docs(self.dump_file), self.clip_docs):
            self.titles.append(title)
            yield self.dictionary.doc2bow(tokens)
    
    def __len__(self):
        return self.clip_docs
