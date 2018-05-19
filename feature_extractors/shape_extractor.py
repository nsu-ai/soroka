import codecs
import copy
import json
import os

from nltk import word_tokenize
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class ShapeExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, config_name=os.path.join(os.path.dirname(__file__), '..', 'data', 'shape_config.json'),
                 tokenizer=word_tokenize, max_number_of_shapes: int=50):
        self.config_name = config_name
        self.tokenizer = tokenizer
        self.max_number_of_shapes = max_number_of_shapes

    def fit(self, X, y=None):
        self.check_X(X)
        self.characters_ = self.load_config(self.config_name)
        if self.max_number_of_shapes < 2:
            raise ValueError('`max_number_of_shapes` is wrong! '
                             '{0} is too small value!'.format(self.max_number_of_shapes))
        vocabulary = dict()
        for cur_text in X:
            for cur_token in self.tokenizer(cur_text):
                cur_shape = self.str_to_shape(cur_token)
                if cur_shape in vocabulary:
                    vocabulary[cur_shape] += 1
                else:
                    vocabulary[cur_shape] = 1
        vocabulary = sorted([(cur_shape, vocabulary[cur_shape]) for cur_shape in vocabulary],
                            key=lambda it: (-it[1], it[0]))
        if len(vocabulary) > self.max_number_of_shapes:
            vocabulary = sorted(vocabulary[ind][0] for ind in range(self.max_number_of_shapes))
        else:
            vocabulary = sorted(vocabulary[ind][0] for ind in range(len(vocabulary)))
        self.vocabulary_ = dict([(vocabulary[ind], ind) for ind in range(len(vocabulary))])
        lengths = [len(self.tokenizer(sentence)) for sentence in X]
        self.max_sentence_length_ = int(np.mean(lengths) + np.std(lengths))
        return self

    def transform(self, X):
        self.check_X(X)
        check_is_fitted(self, ['max_sentence_length_', 'characters_', 'vocabulary_'])
        result = np.zeros((len(X), 1, self.max_sentence_length_, len(self.vocabulary_) + 1), dtype=np.float32)
        for text_ind in range(len(X)):
            tokens = self.tokenizer(X[text_ind])
            n_tokens = min(len(tokens), self.max_sentence_length_)
            for token_ind in range(n_tokens):
                shape = self.str_to_shape(tokens[token_ind])
                if shape in self.vocabulary_:
                    result[text_ind, 0, token_ind, self.vocabulary_[shape]] = 1.0
                result[text_ind, 0, token_ind, len(self.vocabulary_)] = 1.0
        return result

    def check_X(self, X):
        if (not isinstance(X, list)) and (not isinstance(X, tuple)) and (not isinstance(X, np.ndarray)):
            raise TypeError('X must be a list, but it is a {type}'.format(type=type(X)))
        if isinstance(X, np.ndarray):
            if len(X.shape) != 1:
                raise TypeError('X must be a 1-D array, but it is a {shape}-D array'.format(shape=len(X.shape)))

    def str_to_shape(self, src: str) -> str:
        dst = []
        prev_char = None
        for cur_char in src:
            new_char = self.characters_.get(cur_char, 'Unk')
            if prev_char is None:
                dst.append(new_char)
            else:
                if prev_char != new_char:
                    dst.append(new_char)
            prev_char = new_char
        return ''.join(dst)

    def __setstate__(self, state):
        self.tokenizer = copy.deepcopy(state['tokenizer'])
        self.config_name = state['config_name']
        self.max_number_of_shapes = state['max_number_of_shapes']
        if ('max_sentence_length_' in state) and ('characters_' in state) and ('vocabulary_' in state):
            self.max_sentence_length_ = state['max_sentence_length_']
            self.characters_ = copy.deepcopy(state['characters_'])
            self.vocabulary_ = copy.deepcopy(state['vocabulary_'])
        return self

    def __getstate__(self):
        result = {'tokenizer': copy.deepcopy(self.tokenizer), 'config_name': self.config_name,
                  'max_number_of_shapes': self.max_number_of_shapes}
        if hasattr(self, 'max_sentence_length_'):
            result['max_sentence_length_'] = self.max_sentence_length_
        if hasattr(self, 'characters_'):
            result['characters_'] = copy.deepcopy(self.characters_)
        if hasattr(self, 'vocabulary_'):
            result['vocabulary_'] = copy.deepcopy(self.vocabulary_)
        return result

    @staticmethod
    def load_config(config_name: str) -> dict:
        with codecs.open(config_name, mode='r', encoding='utf-8', errors='ignore') as fp:
            data = json.load(fp)
        if not isinstance(data, dict):
            raise IOError('File "{0}" does not contain a `dict` object.')
        all_shapes = set(data.keys())
        all_characters = set()
        prepared_data = dict()
        for cur_shape in all_shapes:
            if not isinstance(cur_shape, str):
                raise IOError('File "{0}" contains a wrong specification of shape. Name of shape must be a string , '
                              'but `{1}` is not a `str` object.'.format(config_name, type(cur_shape)))
            if len(cur_shape) == 0:
                raise IOError('File "{0}" contains a wrong specification of shape. '
                              'Name of shape is empty.'.format(config_name))
            if (len(cur_shape) > 2) or (not cur_shape.isalpha()) or (not cur_shape[0].isupper()):
                raise IOError('File "{0}" contains a wrong specification of shape. '
                              '"{1}" is inadmissible name of shape.'.format(config_name, type(cur_shape)))
            if len(cur_shape) == 2:
                if cur_shape[1].isupper():
                    raise IOError('File "{0}" contains a wrong specification of shape. '
                                  '"{1}" is inadmissible name of shape.'.format(config_name, type(cur_shape)))
            characters_for_shape = set(data[cur_shape])
            for cur in characters_for_shape:
                if len(cur) != 1:
                    raise IOError('File "{0}" contains a wrong specification of shape. '
                                  '"{1}" is not a character.'.format(config_name, cur))
                if cur in all_characters:
                    raise IOError('File "{0}" contains a wrong specification of shape. '
                                  'Character "{1}" is duplicated.'.format(config_name, cur))
                all_characters.add(cur)
                prepared_data[cur] = cur_shape
        return prepared_data
