from argparse import ArgumentParser
import copy
import os
import gensim
import numpy as np

from nltk import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class EmbeddingExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, word2vec_name=os.path.join(os.path.dirname(__file__), '..', 'data', 'word2vec_main.w2v'),
                 tokenizer=word_tokenize, lowercase=False):
        self.tokenizer = tokenizer
        self.word2vec_name = word2vec_name
        self.lowercase = lowercase

    def fit(self, X, y=None):
        self.check_X(X)
        self.word2vec_ = gensim.models.KeyedVectors.load_word2vec_format(self.word2vec_name, binary=True,
                                                                         unicode_errors='ignore')
        self.word2vec_.init_sims(replace=True)
        self.size_ = len(self.word2vec_[list(self.word2vec_.vocab.keys())[0]])
        lengths = [len(self.tokenizer(sentence)) for sentence in X]
        self.max_sentence_length_ = int(np.mean(lengths) + np.std(lengths))
        return self

    def transform(self, X):
        self.check_X(X)
        check_is_fitted(self, ['max_sentence_length_', 'size_', 'word2vec_'])
        result = np.zeros((len(X), 1, self.max_sentence_length_, self.size_ + 1), dtype=np.float32)
        for line_number, line in enumerate(X):
            for word_number, word in enumerate(self.tokenizer(line.lower() if self.lowercase else line)):
                if word_number >= self.max_sentence_length_:
                    break
                if word in self.word2vec_:
                    result[line_number, 0, word_number, 0:self.size_] = self.word2vec_[word]
                result[line_number, 0, word_number, self.size_] = 1.0
        return result

    def check_X(self, X):
        if (not isinstance(X, list)) and (not isinstance(X, tuple)) and (not isinstance(X, np.ndarray)):
            raise TypeError('X must be a list, but it is a {type}'.format(type=type(X)))
        if isinstance(X, np.ndarray):
            if len(X.shape) != 1:
                raise TypeError('X must be a 1-D array, but it is a {shape}-D array'.format(shape=len(X.shape)))

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.tokenizer = copy.copy(self.tokenizer)
        result.word2vec_name = copy.copy(self.word2vec_name)
        result.lowercase = self.lowercase
        if hasattr(self, 'max_sentence_length_'):
            result.max_sentence_length_ = self.max_sentence_length_
        if hasattr(self, 'size_'):
            result.size_ = self.size_
        if hasattr(self, 'word2vec_'):
            result.word2vec_ = self.word2vec_
        return result

    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        result = cls.__new__(cls)
        result.tokenizer = copy.deepcopy(self.tokenizer)
        result.word2vec_name = copy.copy(self.word2vec_name)
        result.lowercase = self.lowercase
        if hasattr(self, 'max_sentence_length_'):
            result.max_sentence_length_ = self.max_sentence_length_
        if hasattr(self, 'size_'):
            result.size_ = self.size_
        if hasattr(self, 'word2vec_'):
            result.word2vec_ = self.word2vec_
        return result

    def __setstate__(self, state):
        self.tokenizer = copy.deepcopy(state['tokenizer'])
        self.word2vec_name = state['word2vec_name']
        self.lowercase = state['lowercase']
        if ('max_sentence_length_' in state) and ('size_' in state):
            if hasattr(self, 'word2vec_'):
                del self.word2vec_
            self.max_sentence_length_ = state['max_sentence_length_']
            self.size_ = state['size_']
            self.word2vec_ = gensim.models.KeyedVectors.load_word2vec_format(self.word2vec_name, binary=True,
                                                                             unicode_errors='ignore')
            self.word2vec_.init_sims(replace=True)
        return self

    def __getstate__(self):
        result =  {'tokenizer': copy.deepcopy(self.tokenizer), 'word2vec_name': self.word2vec_name,
                   'lowercase': self.lowercase}
        if hasattr(self, 'max_sentence_length_'):
            result['max_sentence_length_'] = self.max_sentence_length_
        if hasattr(self, 'size_'):
            result['size_'] = self.size_
        return result


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-d', '--download', dest='downloaded_w2v_model', type=str, required=True,
                        choices=['small', 'main', 'large'],
                        help='Kind of a word2vec model for downloading (`small` model, `main` model or `large` model).')
    args = parser.parse_args()

    if args.downloaded_w2v_model == 'main':
        model_url = 'http://panchenko.me/data/dsl-backup/w2v-ru/tenth.norm-sz500-w7-cb0-it5-min5.w2v'
        model_name = os.path.join(os.path.dirname(__file__), '..', 'data', 'word2vec_main.w2v')
    elif args.downloaded_w2v_model == 'large':
        model_name = os.path.join(os.path.dirname(__file__), '..', 'data', 'word2vec_large.w2v')
        model_url = 'http://panchenko.me/data/dsl-backup/w2v-ru/all.norm-sz500-w10-cb0-it3-min5.w2v'
    else:
        model_name = os.path.join(os.path.dirname(__file__), '..', 'data', 'word2vec_small.w2v')
        model_url = 'http://panchenko.me/data/dsl-backup/w2v-ru/all.norm-sz100-w10-cb0-it1-min100.w2v'

    import wget
    wget.download(url=model_url, out=model_name, bar=wget.bar_adaptive)
