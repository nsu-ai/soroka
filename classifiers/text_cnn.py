import copy
import os
import pickle
import tempfile
from typing import Union
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score

from classifiers.cnn import CNNClassifier
from feature_extractors import EmbeddingExtractor


class TextCNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, feature_extractor: EmbeddingExtractor, base_estimator: CNNClassifier, batch_size: int=1000,
                 warm_start: bool=False):
        self.feature_extractor = feature_extractor
        self.base_estimator = base_estimator
        self.batch_size = batch_size
        self.warm_start = warm_start

    def fit(self, X, y, **fit_params):
        self.check_X(X)
        self.check_y(y, len(X))
        if 'validation' in fit_params:
            if (not isinstance(fit_params['validation'], tuple)) and (not isinstance(fit_params['validation'], list)):
                raise ValueError('Validation data are specified incorrectly!')
            if len(fit_params['validation']) != 2:
                raise ValueError('Validation data are specified incorrectly!')
            X_val = fit_params['validation'][0]
            y_val = fit_params['validation'][1]
            self.check_X(X_val)
            self.check_y(y_val, len(X_val))
        else:
            X_val = None
            y_val = None
        self._check_params()
        self.feature_extractor.fit(X)
        if self.base_estimator.verbose:
            print('')
            print('Feature extractor has been fitted...')
            print('Caclulated sentence length is {0}.'.format(self.feature_extractor.max_sentence_length_))
            print('')
        n_batches = len(X) // self.batch_size
        while (n_batches * self.batch_size) < len(X):
            n_batches += 1
        warm_start = self.base_estimator.warm_start
        max_epochs_number = self.base_estimator.max_epochs_number
        epochs_before_stopping = self.base_estimator.epochs_before_stopping
        validation_fraction = self.base_estimator.validation_fraction
        eval_metric = self.base_estimator.eval_metric
        fp = tempfile.NamedTemporaryFile()
        tmp_cnn_name = fp.name
        fp.close()
        del fp
        try:
            self.base_estimator.warm_start = self.warm_start
            self.base_estimator.max_epochs_number = 1
            self.base_estimator.epochs_before_stopping = 1
            self.base_estimator.validation_fraction = None
            self.base_estimator.eval_metric = 'Logloss'
            if self.base_estimator.verbose:
                print('')
                print('==========')
                print('Training of TextCNN is started...')
            f1_best = None
            best_epoch = None
            epochs_without_improving = 0
            for cur_epoch in range(max_epochs_number):
                for batch_ind in range(n_batches):
                    start_ind = batch_ind * self.batch_size
                    end_ind = (batch_ind + 1) * self.batch_size
                    cur_X, cur_y = self.__generate_data(start_ind, end_ind, X, y)
                    self.base_estimator.fit(cur_X, cur_y)
                    if not self.base_estimator.warm_start:
                        self.base_estimator.warm_start = True
                if self.base_estimator.verbose:
                    print('')
                    print('Epoch {0} is completed.'.format(cur_epoch + 1))
                    print('==========')
                if (X_val is not None) and (y_val is not None):
                    f1 = f1_score(y_val, self.predict(X_val), average='macro')
                    if self.base_estimator.verbose:
                        print('F1 is {0:.3%}.'.format(f1))
                    if f1_best is None:
                        best_epoch = cur_epoch
                        f1_best = f1
                        with open(tmp_cnn_name, 'wb') as fp:
                            pickle.dump(self.base_estimator, fp)
                    elif f1 > f1_best:
                        f1_best = f1
                        best_epoch = cur_epoch
                        with open(tmp_cnn_name, 'wb') as fp:
                            pickle.dump(self.base_estimator, fp)
                        epochs_without_improving = 0
                    else:
                        epochs_without_improving += 1
                        if epochs_without_improving >= epochs_before_stopping:
                            break
            if self.base_estimator.verbose:
                print('')
                print('Training of TextCNN is finished...')
                if best_epoch is not None:
                    print('Best epoch of training is {0}.'.format(best_epoch + 1))
                print('==========')
                print('')
            if os.path.exists(tmp_cnn_name):
                with open(tmp_cnn_name, 'rb') as fp:
                    del self.base_estimator
                    self.base_estimator = pickle.load(fp)
                    self.base_estimator.n_iter_ = best_epoch
        finally:
            self.base_estimator.validation_fraction = validation_fraction
            self.base_estimator.max_epochs_number = max_epochs_number
            self.base_estimator.epochs_before_stopping = epochs_before_stopping
            self.base_estimator.eval_metric = eval_metric
            self.base_estimator.warm_start = warm_start
            if os.path.isfile(tmp_cnn_name):
                os.remove(tmp_cnn_name)
        return self

    def predict_proba(self, X):
        self.check_X(X)
        self._check_params()
        n_batches = len(X) // self.batch_size
        while (n_batches * self.batch_size) < len(X):
            n_batches += 1
        y = None
        for batch_ind in range(n_batches):
            start_ind = batch_ind * self.batch_size
            end_ind = (batch_ind + 1) * self.batch_size
            cur_X = self.__generate_data(start_ind, end_ind, X)
            cur_y = self.base_estimator.predict_proba(cur_X)
            if y is None:
                y = cur_y.copy()
            else:
                y = np.vstack((y, cur_y))
        return y

    def predict(self, X):
        self.check_X(X)
        self._check_params()
        n_batches = len(X) // self.batch_size
        while (n_batches * self.batch_size) < len(X):
            n_batches += 1
        y = None
        for batch_ind in range(n_batches):
            start_ind = batch_ind * self.batch_size
            end_ind = (batch_ind + 1) * self.batch_size
            cur_X = self.__generate_data(start_ind, end_ind, X)
            cur_y = self.base_estimator.predict(cur_X)
            if y is None:
                y = cur_y.copy()
            else:
                y = np.concatenate((y, cur_y))
        return y

    def _check_params(self):
        if not isinstance(self.feature_extractor, EmbeddingExtractor):
            raise ValueError('Type of `feature_extractor` is wrong! Expected `feature_extractors.embedding_extractor.'
                             'EmbeddingExtractor`, but got `{0}`.'.format(type(self.feature_extractor)))
        if not isinstance(self.base_estimator, CNNClassifier):
            raise ValueError('Type of `base_estimator` is wrong! Expected `classifiers.cnn.CNNClassifier`, '
                             'but got `{0}`.'.format(type(self.base_estimator)))
        if 'conv' not in self.base_estimator.layers:
            raise ValueError('Structure of `base_estimator` is wrong! Convolutional layers are not specified!')
        for layer_ind in range(len(self.base_estimator.layers['conv'])):
            if len(self.base_estimator.layers['conv'][layer_ind]) != 2:
                raise ValueError('Convolutional layer {0} of `base_estimator` is wrong!'.format(layer_ind))
            if len(self.base_estimator.layers['conv'][layer_ind][1]) != 2:
                raise ValueError('Convolutional layer {0} of `base_estimator` is wrong!'.format(layer_ind))
            if (self.base_estimator.layers['conv'][layer_ind][1][0] < 1) or \
                    (self.base_estimator.layers['conv'][layer_ind][1][1] > 0):
                raise ValueError('Convolutional layer {0} of `base_estimator` is wrong!'.format(layer_ind))
        if not isinstance(self.batch_size, int):
            raise ValueError('Type of `batch_size` is wrong! Expected `int`, but '
                             'got `{0}`.'.format(type(self.batch_size)))
        if self.batch_size < 1:
            raise ValueError('Value of `batch_size` is wrong! Expected `a positive integer value, but '
                             'got `{0}`.'.format(self.batch_size))
        if self.batch_size < self.base_estimator.batch_size:
            raise ValueError('Value of `batch_size` less than value of `base_estimator.batch_size`!')
        if not isinstance(self.warm_start, int):
            raise ValueError('Type of `warm_start` is wrong! Expected `bool`, but '
                             'got `{0}`.'.format(type(self.warm_start)))
        if self.base_estimator.validation_fraction is not None:
            warnings.warn('`base_estimator.validation_fraction` is not considered. For early stopping we can use '
                          'additional argument `validation` in the `fit`.')

    def __generate_data(self, start_ind, end_ind, X, y=None):
        if end_ind > len(X):
            end_ind = len(X)
        if y is None:
            return self.feature_extractor.transform(X[start_ind:end_ind])
        return self.feature_extractor.transform(X[start_ind:end_ind]), y[start_ind:end_ind]

    @staticmethod
    def check_X(X: Union[list, tuple, np.ndarray]):
        if (not isinstance(X, list)) and (not isinstance(X, tuple)) and (not isinstance(X, np.ndarray)):
            raise ValueError('Type of `X` is wrong! '
                             'Expected `list`, `tuple` or `numpy.ndarray, but got `{0}`'.format(type(X)))
        if isinstance(X, np.ndarray):
            if len(X.shape) != 1:
                raise ValueError('Structure of `X` is wrong! Expected a 1-D array, but got a '
                                 '{0}-D array.'.format(len(X.shape)))

    @staticmethod
    def check_y(y: Union[list, tuple, np.ndarray], n: int):
        if (not isinstance(y, list)) and (not isinstance(y, tuple)) and (not isinstance(y, np.ndarray)):
            raise ValueError('Type of `X` is wrong! '
                             'Expected `list`, `tuple` or `numpy.ndarray, but got `{0}`'.format(type(y)))
        if isinstance(y, np.ndarray):
            if len(y.shape) != 1:
                raise ValueError('Structure of `X` is wrong! Expected a 1-D array, but got a '
                                 '{0}-D array.'.format(len(y.shape)))
        if len(y) != n:
            raise ValueError('`y` does not correspond to `X`! {0} != {1}.'.format(len(y), n))

    def get_params(self, deep=True):
        """ Получить словарь управляющих параметров нейросети.

        Данный метод используется внутри sklearn.pipeline.Pipeline, sklearn.model_selection.GridSearchCV и пр.
        Соответствено, если мы хотим насладиться всей мощью scikit-learn и использовать наш класс там, то данный метод
        нужно корректно реализовать.

        :return словарь управляющих параметров нейросети (без параметров, настроенных по итогам обучения).

        """
        return {'feature_extractor': copy.deepcopy(self.feature_extractor) if deep else self.feature_extractor,
                'base_estimator': copy.deepcopy(self.base_estimator) if deep else self.base_estimator,
                'batch_size': self.batch_size, 'warm_start': self.warm_start}

    def set_params(self, **parameters):
        """ Установить новые значения управляющих параметров нейросети из словаря.

        Данный метод используется внутри sklearn.pipeline.Pipeline, sklearn.model_selection.GridSearchCV и пр.
        Соответствено, если мы хотим насладиться всей мощью scikit-learn и использовать наш класс там, то данный метод
        нужно корректно реализовать.

        :param parameters: Названия и значения устанавливаемых параметров, заданные словарём.

        :return self

        """
        for parameter, value in parameters.items():
            self.__setattr__(parameter, value)
        return self

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.base_estimator = copy.copy(self.base_estimator)
        result.feature_extractor = copy.copy(self.feature_extractor)
        result.batch_size= self.batch_size
        result.warm_start = self.warm_start
        return result

    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        result = cls.__new__(cls)
        result.base_estimator = copy.deepcopy(self.base_estimator)
        result.feature_extractor = copy.deepcopy(self.feature_extractor)
        result.batch_size = self.batch_size
        result.warm_start = self.warm_start
        return result

    def __getstate__(self):
        """ Нужно для сериализации через pickle. """
        state = {
            'batch_size': self.batch_size,
            'warm_start': self.warm_start,
            'feature_extractor': self.feature_extractor,
            'base_estimator': self.base_estimator
        }
        return state

    def __setstate__(self, state):
        """ Нужно для десериализации через pickle. """
        self.batch_size = state['batch_size']
        self.warm_start = state['warm_start']
        if hasattr(self, 'base_estimator'):
            del self.base_estimator
        if hasattr(self, 'feature_extractor'):
            del self.feature_extractor
        self.base_estimator = state['base_estimator']
        self.feature_extractor = state['feature_extractor']