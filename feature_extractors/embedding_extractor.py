import os
import gensim
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError

class EmbeddingExtractor(BaseEstimator, ClassifierMixin):
    def __init__(self, word2vec_name = os.path.join('data', 'word2vec'), tokenizer = str.split):
        self.tokenize = tokenizer
        self.fitted = False
        self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_name, binary = True, unicode_errors = 'ignore')
        self.size = len(self.word2vec[list(self.word2vec.vocab.keys())[0]])
        
    def fit(self, X, y = None):
        lengths = [len(self.tokenize(sentence)) for sentence in X]
        self.max_sentence_length = int(np.mean(lengths) + np.std(lengths))
        self.fitted = True
        return self
    
    def transform(self, X, y = None):
        if not isinstance(X, list):
            raise TypeError('X must be a list, but it is a {type}'.format(type = type(X)))
        
        if not self.fitted:
            raise NotFittedError
        
        result = np.zeros((len(X), 1, self.max_sentence_length, self.size + 1))
        for line_number, line in enumerate(X):
            for word_number, word in enumerate(self.tokenize(line)):
                if word_number < self.max_sentence_length:
                    try:
                        result[line_number, 0, word_number] = np.hstack([self.word2vec[word], [0]])
                    except KeyError:
                        result[line_number, 0, word_number] = np.hstack([np.zeros(self.size), [1]])
                else:
                    break
        
        return result
    
    def fit_transform(self, X, y = None):
        return self.fit(X).transform(X)
    
    def __setstate__(self, state):
        self.tokenize = state['tokenize']
        self.fitted = state['fitted']
        self.word2vec = state['word2vec']
        self.size = state['size']
        if self.fitted:
            self.max_sentence_length = state['max_sentence_length']
        return self
    
    def __getstate__(self):
        result =  {'tokenize': self.tokenize, 'fitted': self.fitted,
                'word2vec': self.word2vec, 'size': self.size}
        
        if self.fitted:
            result['max_sentence_length'] = self.max_sentence_length
        
        return result