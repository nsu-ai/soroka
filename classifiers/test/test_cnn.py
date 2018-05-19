import copy
import os
import pickle
import re
import tempfile
import unittest

import numpy
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_random_state
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from classifiers.cnn import CNNClassifier
from classifiers.utils import split_train_data


class DigitsPreprocessor(BaseEstimator, TransformerMixin):
    """ Простенький препроцессор изображений цифр, заданных 64-мерными векторами, в матрицы 1x8x8 для LeNet. """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if len(X.shape) != 2:
            raise ValueError('`X` is wrong! We expect a 2-D array!')
        if X.shape[1] != 64:
            raise ValueError('`X` is wrong! Number of image features must be equal to 64!')
        n_samples = X.shape[0]
        return X.reshape(n_samples, 1, 8, 8)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)


class DigitsScaler(BaseEstimator, TransformerMixin):
    """ Нормализатор изображений цифр, переводящий яркость пикселей из диапазона [0, max] в диапазон [0.0, 1.0]. """
    def fit(self, X, y=None):
        self.max_val_ = numpy.abs(X).max()
        return self

    def transform(self, X):
        check_is_fitted(self, ['max_val_'])
        if self.max_val_ > 0:
            return X / self.max_val_
        return X

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        if getattr(self, 'max_val_', None) is not None:
            result.max_val_ = self.max_val_
        return result

    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        result = cls.__new__(cls)
        if getattr(self, 'max_val_', None) is not None:
            result.max_val_ = self.max_val_
        return result

    def __getstate__(self):
        """ Нужно для сериализации через pickle. """
        return dict() if getattr(self, 'max_val_', None) is None else {'max_val_': self.max_val_}

    def __setstate__(self, state):
        """ Нужно для десериализации через pickle. """
        if 'max_val_' in state:
            self.max_val_ = state['max_val_']


class TestCNNClassifier(unittest.TestCase):
    def setUp(self):
        fp = tempfile.NamedTemporaryFile()
        self.classifier_name = fp.name
        fp.close()
        del fp
        self.random_state = check_random_state(0)

    def tearDown(self):
        if os.path.isfile(self.classifier_name):
            os.remove(self.classifier_name)
        del self.classifier_name

    def test_create_positive01(self):
        """ Проверить создание свёрточной нейронной сети. """
        layers = {'conv': ((32, (3, 3)), (64, (2, 2))), 'pool': ((2, 2), (2, 2)), 'dense': (100, 80)}
        dropout = 0.5
        learning_rate = 1e-3
        max_epochs_number = 1000
        epochs_before_stopping = 10
        validation_fraction = 0.1
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-08
        batch_size = 50
        batch_norm = True
        verbose = False
        warm_start = False
        eval_metric = 'ROC-AUC'
        cnn = CNNClassifier(layers=layers, dropout=dropout, learning_rate=learning_rate,
                            max_epochs_number=max_epochs_number, epochs_before_stopping=epochs_before_stopping,
                            validation_fraction=validation_fraction, beta1=beta1, beta2=beta2, epsilon=epsilon,
                            batch_size=batch_size, batch_norm=batch_norm, verbose=verbose, warm_start=warm_start,
                            random_state=None, eval_metric=eval_metric)
        self.assertIsInstance(cnn, CNNClassifier)
        self.assertEqual(layers, cnn.layers)
        self.assertAlmostEqual(dropout, cnn.dropout)
        self.assertAlmostEqual(learning_rate, cnn.learning_rate)
        self.assertEqual(max_epochs_number, cnn.max_epochs_number)
        self.assertEqual(epochs_before_stopping, cnn.epochs_before_stopping)
        self.assertAlmostEqual(validation_fraction, cnn.validation_fraction)
        self.assertAlmostEqual(beta1, cnn.beta1)
        self.assertAlmostEqual(beta2, cnn.beta2)
        self.assertAlmostEqual(epsilon, cnn.epsilon)
        self.assertEqual(batch_size, cnn.batch_size)
        self.assertTrue(cnn.batch_norm)
        self.assertFalse(cnn.verbose)
        self.assertFalse(cnn.warm_start)
        self.assertIsNone(cnn.random_state)
        self.assertEqual(eval_metric, cnn.eval_metric)

    def test_fit_predict_positive01(self):
        """ Проверить обучение и эксплуатацию свёрточной сети в составе пайплайна (на примере MNIST). """
        layers = {'conv': ((32, (3, 3)), (64, (2, 2))), 'pool': ((2, 2), (2, 2)), 'dense': (100, 80)}
        cls = Pipeline(steps=[
            ('normalizer', DigitsScaler()),
            ('reshaper', DigitsPreprocessor()),
            ('classifier', CNNClassifier(layers=layers, verbose=True, max_epochs_number=100, epochs_before_stopping=10,
                                         validation_fraction=0.2, batch_size=100, learning_rate=2e-3,
                                         random_state=self.random_state))
        ])
        X_train, y_train, X_test, y_test = self.load_digits()
        cls.fit(X_train, y_train)
        self.assertEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], cls.named_steps['classifier'].classes_list_)
        self.assertEqual((1, 8, 8), cls.named_steps['classifier'].input_size_)
        self.assertIsInstance(cls.named_steps['classifier'].n_iter_, int)
        self.assertGreaterEqual(cls.named_steps['classifier'].n_iter_, 3)
        self.assertLessEqual(cls.named_steps['classifier'].n_iter_, 100)
        self.assertIsInstance(cls.named_steps['classifier'].loss_value_, float)
        self.assertGreaterEqual(cls.named_steps['classifier'].loss_value_, 0.00)
        self.assertLessEqual(cls.named_steps['classifier'].loss_value_, 1.10)
        self.assertIsNotNone(cls.named_steps['classifier'].cnn_)
        self.assertIsNotNone(cls.named_steps['classifier'].predict_fn_)
        y_pred = cls.predict(X_test)
        self.assertIsInstance(y_pred, numpy.ndarray)
        self.assertEqual(y_test.shape, y_pred.shape)
        self.assertGreater(accuracy_score(y_test, y_pred), 0.93)
        probabilities = cls.predict_proba(X_test)
        self.assertIsInstance(probabilities, numpy.ndarray)
        self.assertEqual((X_test.shape[0], 10), probabilities.shape)
        for sample_ind in range(probabilities.shape[0]):
            additional_err_msg = 'Distribution of class probabilities is wrong for sample {0}.'.format(sample_ind)
            sum_of_probabilities = 0.0
            for class_ind in range(10):
                sum_of_probabilities += probabilities[sample_ind][class_ind]
                self.assertGreaterEqual(probabilities[sample_ind][class_ind], 0.0, msg=additional_err_msg)
                self.assertLessEqual(probabilities[sample_ind][class_ind], 1.0, msg=additional_err_msg)
            self.assertAlmostEqual(sum_of_probabilities, 1.0, places=3, msg=additional_err_msg)
        log_probabilities = cls.predict_log_proba(X_test)
        self.assertIsInstance(log_probabilities, numpy.ndarray)
        self.assertEqual(probabilities.shape, log_probabilities.shape)
        for sample_ind in range(probabilities.shape[0]):
            for class_ind in range(10):
                self.assertAlmostEqual(numpy.log(probabilities[sample_ind][class_ind]),
                                       log_probabilities[sample_ind][class_ind])

    def test_fit_predict_positive02(self):
        """ Проверить обучение и эксплуатацию свёрточной сети с вычислением early stopping на отдельном датасете. """
        X_train, y_train, X_test, y_test = self.load_digits()
        preprocessor = Pipeline(steps=[
            ('normalizer', DigitsScaler()),
            ('reshaper', DigitsPreprocessor())
        ])
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)
        layers = {'conv': ((32, (3, 3)), (64, (2, 2))), 'pool': ((2, 2), (2, 2)), 'dense': (100, 80)}
        cls = CNNClassifier(layers=layers, verbose=True, max_epochs_number=100, epochs_before_stopping=10,
                            batch_size=100, batch_norm=False, learning_rate=2e-3, random_state=self.random_state)
        cls.fit(X_train, y_train, validation=(X_test, y_test))
        self.assertEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], cls.classes_list_)
        self.assertEqual((1, 8, 8), cls.input_size_)
        self.assertIsInstance(cls.n_iter_, int)
        self.assertGreaterEqual(cls.n_iter_, 3)
        self.assertLessEqual(cls.n_iter_, 100)
        self.assertIsInstance(cls.loss_value_, float)
        self.assertGreaterEqual(cls.loss_value_, 0.00)
        self.assertLessEqual(cls.loss_value_, 1.10)
        self.assertIsNotNone(cls.cnn_)
        self.assertIsNotNone(cls.predict_fn_)
        y_pred = cls.predict(X_test)
        self.assertIsInstance(y_pred, numpy.ndarray)
        self.assertEqual(y_test.shape, y_pred.shape)
        self.assertGreater(accuracy_score(y_test, y_pred), 0.93)
        probabilities = cls.predict_proba(X_test)
        self.assertIsInstance(probabilities, numpy.ndarray)
        self.assertEqual((X_test.shape[0], 10), probabilities.shape)
        for sample_ind in range(probabilities.shape[0]):
            additional_err_msg = 'Distribution of class probabilities is wrong for sample {0}.'.format(sample_ind)
            sum_of_probabilities = 0.0
            for class_ind in range(10):
                sum_of_probabilities += probabilities[sample_ind][class_ind]
                self.assertGreaterEqual(probabilities[sample_ind][class_ind], 0.0, msg=additional_err_msg)
                self.assertLessEqual(probabilities[sample_ind][class_ind], 1.0, msg=additional_err_msg)
            self.assertAlmostEqual(sum_of_probabilities, 1.0, places=3, msg=additional_err_msg)
        log_probabilities = cls.predict_log_proba(X_test)
        self.assertIsInstance(log_probabilities, numpy.ndarray)
        self.assertEqual(probabilities.shape, log_probabilities.shape)
        for sample_ind in range(probabilities.shape[0]):
            for class_ind in range(10):
                self.assertAlmostEqual(numpy.log(probabilities[sample_ind][class_ind]),
                                       log_probabilities[sample_ind][class_ind])

    def test_fit_predict_positive03(self):
        """ Проверить обучение и эксплуатацию свёрточной сети с вычислением early stopping по метрике F1. """
        X_train, y_train, X_test, y_test = self.load_digits()
        preprocessor = Pipeline(steps=[
            ('normalizer', DigitsScaler()),
            ('reshaper', DigitsPreprocessor())
        ])
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)
        layers = {'conv': ((32, (3, 3)), (64, (2, 2))), 'pool': ((2, 2), (2, 2)), 'dense': (100, 80)}
        cls = CNNClassifier(layers=layers, verbose=True, max_epochs_number=100, epochs_before_stopping=10,
                            batch_size=100, batch_norm=False, learning_rate=2e-3, random_state=self.random_state,
                            eval_metric='F1')
        cls.fit(X_train, y_train, validation=(X_test, y_test))
        self.assertEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], cls.classes_list_)
        self.assertEqual((1, 8, 8), cls.input_size_)
        self.assertIsInstance(cls.n_iter_, int)
        self.assertGreaterEqual(cls.n_iter_, 3)
        self.assertLessEqual(cls.n_iter_, 100)
        self.assertIsInstance(cls.loss_value_, float)
        self.assertGreaterEqual(cls.loss_value_, 0.90)
        self.assertLessEqual(cls.loss_value_, 1.00)
        self.assertIsNotNone(cls.cnn_)
        self.assertIsNotNone(cls.predict_fn_)
        y_pred = cls.predict(X_test)
        self.assertIsInstance(y_pred, numpy.ndarray)
        self.assertEqual(y_test.shape, y_pred.shape)
        self.assertGreater(accuracy_score(y_test, y_pred), 0.93)
        probabilities = cls.predict_proba(X_test)
        self.assertIsInstance(probabilities, numpy.ndarray)
        self.assertEqual((X_test.shape[0], 10), probabilities.shape)
        for sample_ind in range(probabilities.shape[0]):
            additional_err_msg = 'Distribution of class probabilities is wrong for sample {0}.'.format(sample_ind)
            sum_of_probabilities = 0.0
            for class_ind in range(10):
                sum_of_probabilities += probabilities[sample_ind][class_ind]
                self.assertGreaterEqual(probabilities[sample_ind][class_ind], 0.0, msg=additional_err_msg)
                self.assertLessEqual(probabilities[sample_ind][class_ind], 1.0, msg=additional_err_msg)
            self.assertAlmostEqual(sum_of_probabilities, 1.0, places=3, msg=additional_err_msg)
        log_probabilities = cls.predict_log_proba(X_test)
        self.assertIsInstance(log_probabilities, numpy.ndarray)
        self.assertEqual(probabilities.shape, log_probabilities.shape)
        for sample_ind in range(probabilities.shape[0]):
            for class_ind in range(10):
                self.assertAlmostEqual(numpy.log(probabilities[sample_ind][class_ind]),
                                       log_probabilities[sample_ind][class_ind])

    def test_fit_predict_positive04(self):
        """ Проверить обучение и эксплуатацию свёрточной сети с вычислением early stopping по метрике ROC-AUC. """
        X_train, y_train_, X_test, y_test_ = self.load_digits()
        preprocessor = Pipeline(steps=[
            ('normalizer', DigitsScaler()),
            ('reshaper', DigitsPreprocessor())
        ])
        y_train = numpy.empty(y_train_.shape, dtype=y_train_.dtype)
        for sample_ind in range(y_train_.shape[0]):
            if y_train_[sample_ind] in {0, 2, 4, 6, 8}:
                y_train[sample_ind] = 1
            else:
                y_train[sample_ind] = 0
        y_test = numpy.empty(y_test_.shape, dtype=y_test_.dtype)
        for sample_ind in range(y_test_.shape[0]):
            if y_test_[sample_ind] in {0, 2, 4, 6, 8}:
                y_test[sample_ind] = 1
            else:
                y_test[sample_ind] = 0
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)
        layers = {'conv': ((32, (3, 3)), (64, (2, 2))), 'pool': ((2, 2), (2, 2)), 'dense': (100, 80)}
        cls = CNNClassifier(layers=layers, verbose=True, max_epochs_number=100, epochs_before_stopping=10,
                            batch_size=100, batch_norm=False, learning_rate=2e-3, random_state=self.random_state,
                            eval_metric='ROC-AUC')
        cls.fit(X_train, y_train, validation=(X_test, y_test))
        self.assertEqual([0, 1], cls.classes_list_)
        self.assertEqual((1, 8, 8), cls.input_size_)
        self.assertIsInstance(cls.n_iter_, int)
        self.assertGreaterEqual(cls.n_iter_, 3)
        self.assertLessEqual(cls.n_iter_, 100)
        self.assertIsInstance(cls.loss_value_, float)
        self.assertGreaterEqual(cls.loss_value_, 0.90)
        self.assertLessEqual(cls.loss_value_, 1.00)
        self.assertIsNotNone(cls.cnn_)
        self.assertIsNotNone(cls.predict_fn_)
        y_pred = cls.predict(X_test)
        self.assertIsInstance(y_pred, numpy.ndarray)
        self.assertEqual(y_test.shape, y_pred.shape)
        self.assertGreater(accuracy_score(y_test, y_pred), 0.93)
        probabilities = cls.predict_proba(X_test)
        self.assertIsInstance(probabilities, numpy.ndarray)
        self.assertEqual((X_test.shape[0], 2), probabilities.shape)
        for sample_ind in range(probabilities.shape[0]):
            additional_err_msg = 'Distribution of class probabilities is wrong for sample {0}.'.format(sample_ind)
            sum_of_probabilities = 0.0
            for class_ind in range(2):
                sum_of_probabilities += probabilities[sample_ind][class_ind]
                self.assertGreaterEqual(probabilities[sample_ind][class_ind], 0.0, msg=additional_err_msg)
                self.assertLessEqual(probabilities[sample_ind][class_ind], 1.0, msg=additional_err_msg)
            self.assertAlmostEqual(sum_of_probabilities, 1.0, places=3, msg=additional_err_msg)
        log_probabilities = cls.predict_log_proba(X_test)
        self.assertIsInstance(log_probabilities, numpy.ndarray)
        self.assertEqual(probabilities.shape, log_probabilities.shape)
        for sample_ind in range(probabilities.shape[0]):
            for class_ind in range(2):
                self.assertAlmostEqual(numpy.log(probabilities[sample_ind][class_ind]),
                                       log_probabilities[sample_ind][class_ind])

    def test_fit_negative01(self):
        """ Проверить обучение, если валидационное множество задано неверно (не списком или кортежем, а словарём). """
        layers = {'conv': ((32, (3, 3)), (64, (2, 2))), 'pool': ((2, 2), (2, 2)), 'dense': (100, 80)}
        cls = Pipeline(steps=[
            ('normalizer', DigitsScaler()),
            ('reshaper', DigitsPreprocessor()),
            ('classifier', CNNClassifier(layers=layers, verbose=False, max_epochs_number=10, epochs_before_stopping=5,
                                         random_state=self.random_state))
        ])
        X_train, y_train, X_test, y_test = self.load_digits()
        true_err_msg = re.escape('Validation data are specified incorrectly!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            cls.fit(X_train, y_train, classifier__validation={'X': X_test, 'y': y_test})

    def test_fit_negative02(self):
        """ Проверить обучение, если входные примеры для обучения - не 4-мерная матрица. """
        layers = {'conv': ((32, (5, 5)), (64, (3, 3))), 'pool': ((2, 2), (2, 2)), 'dense': (300, 100)}
        cls = CNNClassifier(layers=layers, verbose=False, max_epochs_number=10, epochs_before_stopping=5,
                            random_state=self.random_state)
        X_train, y_train, _, _ = self.load_digits()
        true_err_msg = re.escape('`X` must be a 4-D array (samples, input maps, rows of input map, columns of input '
                                 'map)!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            cls.fit(X_train, y_train)

    def test_fit_negative03(self):
        """ Проверить обучение, если некоторые метки классов для обучения отрицательны. """
        layers = {'conv': ((32, (3, 3)), (64, (2, 2))), 'pool': ((2, 2), (2, 2)), 'dense': (100, 80)}
        cls = Pipeline(steps=[
            ('normalizer', DigitsScaler()),
            ('reshaper', DigitsPreprocessor()),
            ('classifier', CNNClassifier(layers=layers, verbose=False, max_epochs_number=10, epochs_before_stopping=5,
                                         random_state=self.random_state))
        ])
        X_train, y_train, _, _ = self.load_digits()
        y_train[2] = -1
        y_train[10] = -1
        true_err_msg = re.escape('Target values must be non-negative integer numbers!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            cls.fit(X_train, y_train)

    def test_fit_negative04(self):
        """ Проверить обучение, если некоторые метки классов не в диапазоне [0, N), где N - общее число классов. """
        layers = {'conv': ((32, (3, 3)), (64, (2, 2))), 'pool': ((2, 2), (2, 2)), 'dense': (100, 80)}
        cls = Pipeline(steps=[
            ('normalizer', DigitsScaler()),
            ('reshaper', DigitsPreprocessor()),
            ('classifier', CNNClassifier(layers=layers, verbose=False, max_epochs_number=10, epochs_before_stopping=5,
                                         random_state=self.random_state))
        ])
        X_train, y_train, _, _ = self.load_digits()
        y_train[2] = 12
        y_train[10] = 17
        true_err_msg = re.escape('Target values must be non-negative integer numbers!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            cls.fit(X_train, y_train)

    def test_fit_negative05(self):
        """ Проверить обучение, если число входных примеров для обучения не равно числу меток классов. """
        layers = {'conv': ((32, (3, 3)), (64, (2, 2))), 'pool': ((2, 2), (2, 2)), 'dense': (100, 80)}
        cls = CNNClassifier(layers=layers, verbose=False, max_epochs_number=10, epochs_before_stopping=5,
                            random_state=self.random_state)
        X_train, y_train, _, _ = self.load_digits()
        with self.assertRaises(ValueError):
            cls.fit(X_train.reshape((X_train.shape[0], 1, 8, 8)), y_train[0:-1])

    def test_fit_negative06(self):
        """ Проверить обучение, если входные примеры валидационного множества не соответствуют примерам обучающего. """
        layers = {'conv': ((32, (3, 3)), (64, (2, 2))), 'pool': ((2, 2), (2, 2)), 'dense': (100, 80)}
        cls = CNNClassifier(layers=layers, verbose=False, max_epochs_number=10, epochs_before_stopping=5,
                            random_state=self.random_state)
        X_train, y_train, X_test, y_test = self.load_digits()
        true_err_msg = re.escape('Validation inputs do not correspond to train inputs!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            cls.fit(X_train.reshape((X_train.shape[0], 1, 8, 8)), y_train, validation=(
                X_test.reshape((X_test.shape[0], 1, 4, 16)),
                y_test
            ))

    def test_fit_negative07(self):
        """ Проверить обучение, если метки классов валидационного множества не соответствуют меткам обучающего. """
        layers = {'conv': ((32, (3, 3)), (64, (2, 2))), 'pool': ((2, 2), (2, 2)), 'dense': (100, 80)}
        cls = CNNClassifier(layers=layers, verbose=False, max_epochs_number=10, epochs_before_stopping=5,
                            random_state=self.random_state)
        X_train, y_train, X_test, y_test = self.load_digits()
        for sample_ind in range(y_test.shape[0]):
            if y_test[sample_ind] == y_train.max():
                y_test[sample_ind] = y_train.max() - 1
        true_err_msg = re.escape('Validation targets do not correspond to train targets!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            cls.fit(X_train.reshape((X_train.shape[0], 1, 8, 8)), y_train, validation=(
                X_test.reshape((X_test.shape[0], 1, 8, 8)),
                y_test
            ))

    def test_fit_negative08(self):
        """ Проверить обучение, если доля примеров для контрольного множества (validation_fraction) слишком мала. """
        layers = {'conv': ((32, (3, 3)), (64, (2, 2))), 'pool': ((2, 2), (2, 2)), 'dense': (100, 80)}
        cls = CNNClassifier(layers=layers, verbose=False, max_epochs_number=10, epochs_before_stopping=5,
                            validation_fraction=0.000001, random_state=self.random_state)
        X_train, y_train, _, _ = self.load_digits()
        true_err_msg = re.escape('Train data cannot be split into train and validation subsets!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            cls.fit(X_train.reshape((X_train.shape[0], 1, 8, 8)), y_train)

    def test_fit_negative09(self):
        """ Проверить обучение, если доля примеров для контрольного множества (validation_fraction) слишком велика. """
        layers = {'conv': ((32, (3, 3)), (64, (2, 2))), 'pool': ((2, 2), (2, 2)), 'dense': (100, 80)}
        cls = CNNClassifier(layers=layers, verbose=False, max_epochs_number=10, epochs_before_stopping=5,
                            validation_fraction=0.999999, random_state=self.random_state)
        X_train, y_train, _, _ = self.load_digits()
        true_err_msg = re.escape('Train data cannot be split into train and validation subsets!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            cls.fit(X_train.reshape((X_train.shape[0], 1, 8, 8)), y_train)

    def test_fit_negative10(self):
        """ Проверить обучение, если рецептивное поле для первого свёрточного слоя задано неверно. """
        layers = {'conv': ((32, (40, 3)), (64, (2, 2))), 'pool': ((2, 2), (2, 2)), 'dense': (100, 80)}
        cls = Pipeline(steps=[
            ('normalizer', DigitsScaler()),
            ('reshaper', DigitsPreprocessor()),
            ('classifier', CNNClassifier(layers=layers, verbose=False, max_epochs_number=10, epochs_before_stopping=5,
                                         random_state = self.random_state))
        ])
        X_train, y_train, _, _ = self.load_digits()
        true_err_msg = re.escape('Convolution layer 1: (-31, 6) is wrong size of feature map!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            cls.fit(X_train, y_train)

    def test_fit_negative11(self):
        """ Проверить обучение, если рецептивное поле для второго свёрточного слоя задано неверно. """
        layers = {'conv': ((32, (3, 3)), (64, (2, 10))), 'pool': ((2, 2), (2, 2)), 'dense': (100, 80)}
        cls = Pipeline(steps=[
            ('normalizer', DigitsScaler()),
            ('reshaper', DigitsPreprocessor()),
            ('classifier', CNNClassifier(layers=layers, verbose=False, max_epochs_number=10, epochs_before_stopping=5,
                                         random_state=self.random_state))
        ])
        X_train, y_train, _, _ = self.load_digits()
        true_err_msg = re.escape('Convolution layer 2: (2, -6) is wrong size of feature map!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            cls.fit(X_train, y_train)

    def test_fit_negative12(self):
        """ Проверить обучение, если рецептивное поле для первого слоя пулинга задано неверно. """
        layers = {'conv': ((32, (3, 3)), (64, (2, 2))), 'pool': ((32, 2), (2, 2)), 'dense': (100, 80)}
        cls = Pipeline(steps=[
            ('normalizer', DigitsScaler()),
            ('reshaper', DigitsPreprocessor()),
            ('classifier', CNNClassifier(layers=layers, verbose=False, max_epochs_number=10, epochs_before_stopping=5,
                                         random_state=self.random_state))
        ])
        X_train, y_train, _, _ = self.load_digits()
        true_err_msg = re.escape('Pooling layer 1: (0, 3) is wrong size of feature map!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            cls.fit(X_train, y_train)

    def test_fit_negative13(self):
        """ Проверить обучение, если рецептивное поле для второго слоя пулинга задано неверно. """
        layers = {'conv': ((32, (3, 3)), (64, (2, 2))), 'pool': ((2, 2), (2, 12)), 'dense': (100, 80)}
        cls = Pipeline(steps=[
            ('normalizer', DigitsScaler()),
            ('reshaper', DigitsPreprocessor()),
            ('classifier', CNNClassifier(layers=layers, verbose=False, max_epochs_number=10, epochs_before_stopping=5,
                                         random_state=self.random_state))
        ])
        X_train, y_train, _, _ = self.load_digits()
        true_err_msg = re.escape('Pooling layer 2: (1, 0) is wrong size of feature map!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            cls.fit(X_train, y_train)

    def test_fit_negative14(self):
        """ Проверить обучение, если слои свёрточной нейросети заданы не словарём, а, например, кортежем. """
        layers = (((32, (3, 3)), (64, (2, 2))), ((2, 2), (2, 2)), (100, 80))
        cls = Pipeline(steps=[
            ('normalizer', DigitsScaler()),
            ('reshaper', DigitsPreprocessor()),
            ('classifier', CNNClassifier(layers=layers, verbose=False, max_epochs_number=10, epochs_before_stopping=5,
                                         random_state=self.random_state))
        ])
        X_train, y_train, _, _ = self.load_digits()
        true_err_msg = re.escape('Structure of hidden layers must be dictionary consisting from three items!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            cls.fit(X_train, y_train)

    def test_fit_negative15(self):
        """ Проверить обучение, если в словаре слоёв свёрточной нейросети не указан один из ключей. """
        layers = {'conv': ((32, (3, 3)), (64, (2, 2))), 'pool': ((2, 2), (2, 2))}
        cls = Pipeline(steps=[
            ('normalizer', DigitsScaler()),
            ('reshaper', DigitsPreprocessor()),
            ('classifier', CNNClassifier(layers=layers, verbose=False, max_epochs_number=10, epochs_before_stopping=5,
                                         random_state=self.random_state))
        ])
        X_train, y_train, _, _ = self.load_digits()
        true_err_msg = re.escape('Structure of hidden layers must be dictionary consisting from three items!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            cls.fit(X_train, y_train)

    def test_fit_negative16(self):
        """ Проверить обучение, если в словаре слоёв свёрточной нейросети указан лишний ключ. """
        layers = {'conv': ((32, (3, 3)), (64, (2, 2))), 'pool': ((2, 2), (2, 2)), 'dense': (100, 80),
                  'additional': ('Hello', ',', 'world', '!')}
        cls = Pipeline(steps=[
            ('normalizer', DigitsScaler()),
            ('reshaper', DigitsPreprocessor()),
            ('classifier', CNNClassifier(layers=layers, verbose=False, max_epochs_number=10, epochs_before_stopping=5,
                                         random_state=self.random_state))
        ])
        X_train, y_train, _, _ = self.load_digits()
        true_err_msg = re.escape('Structure of hidden layers must be dictionary consisting from three items!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            cls.fit(X_train, y_train)

    def test_fit_negative17(self):
        """ Проверить обучение, если в словаре слоёв неправильно назван ключ для слоёв свёртки. """
        layers = {'c': ((32, (3, 3)), (64, (2, 2))), 'pool': ((2, 2), (2, 2)), 'dense': (100, 80)}
        cls = Pipeline(steps=[
            ('normalizer', DigitsScaler()),
            ('reshaper', DigitsPreprocessor()),
            ('classifier', CNNClassifier(layers=layers, verbose=False, max_epochs_number=10, epochs_before_stopping=5,
                                         random_state=self.random_state))
        ])
        X_train, y_train, _, _ = self.load_digits()
        true_err_msg = re.escape('Description of convolution layers (`conv` key) cannot be found in the `layers` dict!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            cls.fit(X_train, y_train)

    def test_fit_negative18(self):
        """ Проверить обучение, если в словаре слоёв неправильно назван ключ для слоёв пулинга. """
        layers = {'conv': ((32, (3, 3)), (64, (2, 2))), 'pooling': ((2, 2), (2, 2)), 'dense': (100, 80)}
        cls = Pipeline(steps=[
            ('normalizer', DigitsScaler()),
            ('reshaper', DigitsPreprocessor()),
            ('classifier', CNNClassifier(layers=layers, verbose=False, max_epochs_number=10, epochs_before_stopping=5,
                                         random_state=self.random_state))
        ])
        X_train, y_train, _, _ = self.load_digits()
        true_err_msg = re.escape('Description of pooling layers (`pool` key) cannot be found in the `layers` dict!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            cls.fit(X_train, y_train)

    def test_fit_negative19(self):
        """ Проверить обучение, если в словаре слоёв неправильно назван ключ для полносвязных скрытых слоёв. """
        layers = {'conv': ((32, (3, 3)), (64, (2, 2))), 'pool': ((2, 2), (2, 2)), 'mlp': (100, 80)}
        cls = Pipeline(steps=[
            ('normalizer', DigitsScaler()),
            ('reshaper', DigitsPreprocessor()),
            ('classifier', CNNClassifier(layers=layers, verbose=False, max_epochs_number=10, epochs_before_stopping=5,
                                         random_state=self.random_state))
        ])
        X_train, y_train, _, _ = self.load_digits()
        true_err_msg = re.escape('Description of dense layers (`dense` key) cannot be found in the `layers` dict!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            cls.fit(X_train, y_train)

    def test_fit_negative20(self):
        """ Проверить обучение, если в словаре слоёв значение для полносвязных скрытых слоёв- не кортеж и не список. """
        layers = {'conv': ((32, (3, 3)), (64, (2, 2))), 'pool': ((2, 2), (2, 2)), 'dense': {100, 80}}
        cls = Pipeline(steps=[
            ('normalizer', DigitsScaler()),
            ('reshaper', DigitsPreprocessor()),
            ('classifier', CNNClassifier(layers=layers, verbose=False, max_epochs_number=10, epochs_before_stopping=5,
                                         random_state=self.random_state))
        ])
        X_train, y_train, _, _ = self.load_digits()
        true_err_msg = re.escape('Structure of dense layers must be list or tuple!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            cls.fit(X_train, y_train)

    def test_fit_negative22(self):
        """ Проверить обучение, если в словаре слоёв значение для полносвязных скрытых слоёв - пустой кортеж. """
        layers = {'conv': ((32, (3, 3)), (64, (2, 2))), 'pool': ((2, 2), (2, 2)), 'dense': tuple()}
        cls = Pipeline(steps=[
            ('normalizer', DigitsScaler()),
            ('reshaper', DigitsPreprocessor()),
            ('classifier', CNNClassifier(layers=layers, verbose=False, max_epochs_number=10, epochs_before_stopping=5,
                                         random_state=self.random_state))
        ])
        X_train, y_train, _, _ = self.load_digits()
        true_err_msg = re.escape('List of dense layers is empty!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            cls.fit(X_train, y_train)

    def test_fit_negative23(self):
        """ Проверить обучение, если в словаре слоёв значение для слоёв свёртки - не кортеж и не список. """
        layers = {'conv': {(32, (3, 3)), (64, (2, 2))}, 'pool': ((2, 2), (2, 2)), 'dense': (100, 80)}
        cls = Pipeline(steps=[
            ('normalizer', DigitsScaler()),
            ('reshaper', DigitsPreprocessor()),
            ('classifier', CNNClassifier(layers=layers, verbose=False, max_epochs_number=10, epochs_before_stopping=5,
                                         random_state=self.random_state))
        ])
        X_train, y_train, _, _ = self.load_digits()
        true_err_msg = re.escape('Structure of convolution layers must be list or tuple!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            cls.fit(X_train, y_train)

    def test_fit_negative24(self):
        """ Проверить обучение, если в словаре слоёв значение для слоёв свёртки - пустой кортеж. """
        layers = {'conv': tuple(), 'pool': ((2, 2), (2, 2)), 'dense': (100, 80)}
        cls = Pipeline(steps=[
            ('normalizer', DigitsScaler()),
            ('reshaper', DigitsPreprocessor()),
            ('classifier', CNNClassifier(layers=layers, verbose=False, max_epochs_number=10, epochs_before_stopping=5,
                                         random_state=self.random_state))
        ])
        X_train, y_train, _, _ = self.load_digits()
        true_err_msg = re.escape('List of convolution layers is empty!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            cls.fit(X_train, y_train)

    def test_fit_negative25(self):
        """ Проверить обучение, если в словаре слоёв значение для слоёв пулинга - не кортеж и не список. """
        layers = {'conv': ((32, (3, 3)), (64, (2, 2))), 'pool': {(2, 2), (2, 2)}, 'dense': (100, 800)}
        cls = Pipeline(steps=[
            ('normalizer', DigitsScaler()),
            ('reshaper', DigitsPreprocessor()),
            ('classifier', CNNClassifier(layers=layers, verbose=False, max_epochs_number=10, epochs_before_stopping=5,
                                         random_state=self.random_state))
        ])
        X_train, y_train, _, _ = self.load_digits()
        true_err_msg = re.escape('Structure of pooling layers must be list or tuple!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            cls.fit(X_train, y_train)

    def test_fit_negative26(self):
        """ Проверить обучение, если в словаре слоёв значение для слоёв пулинга - пустой кортеж. """
        layers = {'conv': ((32, (3, 3)), (64, (2, 2))), 'pool': tuple(), 'dense': (100, 80)}
        cls = Pipeline(steps=[
            ('normalizer', DigitsScaler()),
            ('reshaper', DigitsPreprocessor()),
            ('classifier', CNNClassifier(layers=layers, verbose=False, max_epochs_number=10, epochs_before_stopping=5,
                                         random_state=self.random_state))
        ])
        X_train, y_train, _, _ = self.load_digits()
        true_err_msg = re.escape('List of pooling layers is empty!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            cls.fit(X_train, y_train)

    def test_fit_negative27(self):
        """ Проверить обучение, если в словаре слоёв число слоёв пулинга не соответствует числу слоёв свёртки. """
        layers = {'conv': ((32, (3, 3)), (64, (2, 2))), 'pool': ((2, 2),), 'dense': (100, 80)}
        cls = Pipeline(steps=[
            ('normalizer', DigitsScaler()),
            ('reshaper', DigitsPreprocessor()),
            ('classifier', CNNClassifier(layers=layers, verbose=False, max_epochs_number=10, epochs_before_stopping=5,
                                         random_state=self.random_state))
        ])
        X_train, y_train, _, _ = self.load_digits()
        true_err_msg = re.escape('Number of convolution layers must be equal to number of pooling layers!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            cls.fit(X_train, y_train)

    def test_fit_negative28(self):
        """ Проверить обучение, если в словаре слоёв первый слой свёртки задан некорректно. """
        X_train, y_train, _, _ = self.load_digits()
        true_err_msg = re.escape('Structure of convolution layer 1 is wrong!')
        # если в первом слое свёртки число карт признаков нулевое
        layers = {'conv': ((0, (3, 3)), (64, (2, 2))), 'pool': ((2, 2), (2, 2)), 'dense': (100, 80)}
        cls = Pipeline(steps=[
            ('normalizer', DigitsScaler()),
            ('reshaper', DigitsPreprocessor()),
            ('classifier', CNNClassifier(layers=layers, verbose=False, max_epochs_number=10, epochs_before_stopping=5,
                                         random_state=self.random_state))
        ])
        with self.assertRaisesRegex(ValueError, true_err_msg):
            cls.fit(X_train, y_train)
        # если в первом слое свёртки размер рецептивного поля задан не кортежем и не списком
        layers = {'conv': ((32, {3, 3}), (64, (2, 2))), 'pool': ((2, 2), (2, 2)), 'dense': (100, 80)}
        cls = Pipeline(steps=[
            ('normalizer', DigitsScaler()),
            ('reshaper', DigitsPreprocessor()),
            ('classifier', CNNClassifier(layers=layers, verbose=False, max_epochs_number=10, epochs_before_stopping=5,
                                         random_state=self.random_state))
        ])
        with self.assertRaisesRegex(ValueError, true_err_msg):
            cls.fit(X_train, y_train)
        # если в первом слое свёртки структура самого слоя задана не кортежем и не списком
        layers = {'conv': ({32, (3, 3)}, (64, (2, 2))), 'pool': ((2, 2), (2, 2)), 'dense': (100, 80)}
        cls = Pipeline(steps=[
            ('normalizer', DigitsScaler()),
            ('reshaper', DigitsPreprocessor()),
            ('classifier', CNNClassifier(layers=layers, verbose=False, max_epochs_number=10, epochs_before_stopping=5,
                                         random_state=self.random_state))
        ])
        with self.assertRaisesRegex(ValueError, true_err_msg):
            cls.fit(X_train, y_train)
        # если в первом слое свёртки структура самого слоя задана не 2-элементной, а 3-элементной последовательностью
        layers = {'conv': ((32, (3, 3), 4), (64, (2, 2))), 'pool': ((2, 2), (2, 2)), 'dense': (100, 80)}
        cls = Pipeline(steps=[
            ('normalizer', DigitsScaler()),
            ('reshaper', DigitsPreprocessor()),
            ('classifier',
             CNNClassifier(layers=layers, verbose=False, max_epochs_number=10, epochs_before_stopping=5,
                           random_state=self.random_state))
        ])
        with self.assertRaisesRegex(ValueError, true_err_msg):
            cls.fit(X_train, y_train)
        # если в первом слое свёртки структура самого слоя задана не 2-элементной, а 1-элементной последовательностью
        layers = {'conv': ((32,), (64, (2, 2))), 'pool': ((2, 2), (2, 2)), 'dense': (100, 80)}
        cls = Pipeline(steps=[
            ('normalizer', DigitsScaler()),
            ('reshaper', DigitsPreprocessor()),
            ('classifier', CNNClassifier(layers=layers, verbose=False, max_epochs_number=10, epochs_before_stopping=5,
                                         random_state=self.random_state))
        ])
        with self.assertRaisesRegex(ValueError, true_err_msg):
            cls.fit(X_train, y_train)

    def test_fit_negative29(self):
        """ Проверить обучение, если в словаре слоёв второй слой пулинга задан некорректно. """
        X_train, y_train, _, _ = self.load_digits()
        true_err_msg = re.escape('Structure of pooling layer 2 is wrong!')
        # если во втором слое пулинга размер рецептивного поля задан не кортежем и не списком
        layers = {'conv': ((32, (3, 3)), (64, (2, 2))), 'pool': ((2, 2), {2, 2}), 'dense': (100, 80)}
        cls = Pipeline(steps=[
            ('normalizer', DigitsScaler()),
            ('reshaper', DigitsPreprocessor()),
            ('classifier', CNNClassifier(layers=layers, verbose=False, max_epochs_number=10, epochs_before_stopping=5,
                                         random_state=self.random_state))
        ])
        with self.assertRaisesRegex(ValueError, true_err_msg):
            cls.fit(X_train, y_train)
        # если во втором слое пулинга структура самого слоя задана не 2-элементной, а 3-элементной последовательностью
        layers = {'conv': ((32, (3, 3)), (64, (2, 2))), 'pool': ((2, 2), (2, 2, 2)), 'dense': (100, 80)}
        cls = Pipeline(steps=[
            ('normalizer', DigitsScaler()),
            ('reshaper', DigitsPreprocessor()),
            ('classifier',
             CNNClassifier(layers=layers, verbose=False, max_epochs_number=10, epochs_before_stopping=5,
                           random_state=self.random_state))
        ])
        with self.assertRaisesRegex(ValueError, true_err_msg):
            cls.fit(X_train, y_train)
        # если в втором слое пулинга структура самого слоя задана не 2-элементной, а 1-элементной последовательностью
        layers = {'conv': ((32, (3, 3)), (64, (2, 2))), 'pool': ((2, 2), (2,)), 'dense': (100, 80)}
        cls = Pipeline(steps=[
            ('normalizer', DigitsScaler()),
            ('reshaper', DigitsPreprocessor()),
            ('classifier', CNNClassifier(layers=layers, verbose=False, max_epochs_number=10, epochs_before_stopping=5,
                                         random_state=self.random_state))
        ])
        with self.assertRaisesRegex(ValueError, true_err_msg):
            cls.fit(X_train, y_train)

    def test_fit_negative30(self):
        """ Проверить обучение, если задана неизвестная метрика для early stopping. """
        layers = {'conv': ((32, (3, 3)), (64, (2, 2))), 'pool': ((2, 2), (2, 2)), 'dense': (100, 80)}
        cls = CNNClassifier(layers=layers, verbose=False, max_epochs_number=10, epochs_before_stopping=5,
                            eval_metric='bla-bla-bla', random_state=self.random_state)
        X_train, y_train, _, _ = self.load_digits()
        true_err_msg = re.escape('"bla-bla-bla" is unknown metric for evaluation and early stopping! '
                                 'We expect "Logloss", "F1" or "ROC-AUC".')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            cls.fit(X_train.reshape((X_train.shape[0], 1, 8, 8)), y_train)

    def test_fit_negative31(self):
        """ Проверить обучение, если метрика для early stopping вообще не строка. """
        layers = {'conv': ((32, (3, 3)), (64, (2, 2))), 'pool': ((2, 2), (2, 2)), 'dense': (100, 80)}
        cls = CNNClassifier(layers=layers, verbose=False, max_epochs_number=10, epochs_before_stopping=5,
                            eval_metric=3, random_state=self.random_state)
        X_train, y_train, _, _ = self.load_digits()
        true_err_msg = re.escape('Metric for evaluation and early stopping must be a string value!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            cls.fit(X_train.reshape((X_train.shape[0], 1, 8, 8)), y_train)

    def test_fit_negative32(self):
        """ Проверить обучение, если метрика для early stopping - это ROC-AUC, а задача классификации не бинарна. """
        layers = {'conv': ((32, (3, 3)), (64, (2, 2))), 'pool': ((2, 2), (2, 2)), 'dense': (100, 80)}
        cls = CNNClassifier(layers=layers, verbose=False, max_epochs_number=10, epochs_before_stopping=5,
                            eval_metric='ROC-AUC', random_state=self.random_state)
        X_train, y_train, _, _ = self.load_digits()
        true_err_msg = re.escape('You can not use `ROC-AUC` metric for early stopping '
                                 'if number of classes is greater than 2.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            cls.fit(X_train.reshape((X_train.shape[0], 1, 8, 8)), y_train)

    def test_predict_negative01(self):
        """ Проверить распознавание, если свёрточная сеть ещё не обучена. """
        layers = {'conv': ((32, (3, 3)), (64, (2, 2))), 'pool': ((2, 2), (2, 2)), 'dense': (100, 80)}
        cls = CNNClassifier(layers=layers, verbose=False, max_epochs_number=10, epochs_before_stopping=5,
                            random_state=self.random_state)
        X_train, _, _, _ = self.load_digits()
        with self.assertRaises(NotFittedError):
            _ = cls.predict(X_train)

    def test_predict_negative02(self):
        """ Проверить распознавание, если распознаваемые примеры не соответствуют тем, на которых сеть обучалась. """
        layers = {'conv': ((32, (3, 3)), (64, (2, 2))), 'pool': ((2, 2), (2, 2)), 'dense': (100, 80)}
        cls = CNNClassifier(layers=layers, verbose=False, max_epochs_number=10, epochs_before_stopping=5,
                            batch_size=50, random_state=self.random_state)
        X_train, y_train, X_test, y_test = self.load_digits()
        cls.fit(X_train.reshape((X_train.shape[0], 1, 8, 8)), y_train)
        true_err_msg = re.escape(
            'Samples of `X` do not correspond to the input structure! Got {0}, expected {1}'.format(
                (1, 4, 16), (1, 8, 8)
            )
        )
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _ = cls.predict(X_test.reshape((X_test.shape[0], 1, 4, 16)))

    def test_predict_negative03(self):
        """ Проверить распознавание, если распознаваемые примеры заданы не 4-мерной матрицей. """
        layers = {'conv': ((32, (3, 3)), (64, (2, 2))), 'pool': ((2, 2), (2, 2)), 'dense': (100, 80)}
        cls = CNNClassifier(layers=layers, verbose=False, max_epochs_number=10, epochs_before_stopping=5,
                            batch_size=50, random_state=self.random_state)
        X_train, y_train, X_test, y_test = self.load_digits()
        cls.fit(X_train.reshape((X_train.shape[0], 1, 8, 8)), y_train)
        true_err_msg = re.escape('`X` must be 4D array (samples, input maps, rows of input map, columns of input map)!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _ = cls.predict(X_test)

    def test_pickling_unpickling_positive01(self):
        """ Проверить сериализацию свёрточной сети в бинарный файл и десериализацию её оттуда средствами pickle. """
        X_train, y_train, X_test, y_test = self.load_digits()
        preprocessor = Pipeline(steps=[
            ('normalizer', DigitsScaler()),
            ('reshaper', DigitsPreprocessor())
        ])
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)
        layers = {'conv': ((32, (3, 3)), (64, (2, 2))), 'pool': ((2, 2), (2, 2)), 'dense': (100, 80)}
        cls1 = CNNClassifier(layers=layers, verbose=False, max_epochs_number=10, epochs_before_stopping=5,
                             batch_size=50, validation_fraction=0.2, random_state=self.random_state)
        cls1.fit(X_train, y_train)
        y_pred_1 = cls1.predict(X_test)
        with open(self.classifier_name, 'wb') as fp1:
            pickle.dump(cls1, fp1)
        del cls1
        with open(self.classifier_name, 'rb') as fp2:
            cls2 = pickle.load(fp2)
        y_pred_2 = cls2.predict(X_test)
        self.assertEqual(y_pred_1.tolist(), y_pred_2.tolist())

    def test_copying_positive01(self):
        """ Проверить поверхностное и глубокое копирование обученной свёрточной сети. """
        X_train, y_train, X_test, y_test = self.load_digits()
        preprocessor = Pipeline(steps=[
            ('normalizer', DigitsScaler()),
            ('reshaper', DigitsPreprocessor())
        ])
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)
        layers = {'conv': ((32, (3, 3)), (64, (2, 2))), 'pool': ((2, 2), (2, 2)), 'dense': (100, 80)}
        cls = CNNClassifier(layers=layers, verbose=False, max_epochs_number=10, epochs_before_stopping=5,
                            batch_size=50, validation_fraction=0.2, random_state=self.random_state)
        cls.fit(X_train, y_train)
        y_pred_1 = cls.predict(X_test)
        cls_copy = copy.copy(cls)
        self.assertFalse(cls_copy is cls)
        y_pred_2 = cls_copy.predict(X_test)
        self.assertEqual(y_pred_1.tolist(), y_pred_2.tolist())
        cls_deepcopy = copy.deepcopy(cls)
        self.assertFalse(cls_deepcopy is cls)
        y_pred_3 = cls_deepcopy.predict(X_test)
        self.assertEqual(y_pred_1.tolist(), y_pred_3.tolist())

    def test_fit_with_warm_start(self):
        """ Проверить дообучение свёрточной сети с warm_start, установленным в True. """
        layers = {'conv': ((32, (3, 3)), (64, (2, 2))), 'pool': ((2, 2), (2, 2)), 'dense': (100, 80)}
        cnn = CNNClassifier(layers=layers, verbose=False, max_epochs_number=10, epochs_before_stopping=5,
                            batch_size=50, random_state=self.random_state)
        cls1 = Pipeline(steps=[
            ('normalizer', DigitsScaler()),
            ('reshaper', DigitsPreprocessor()),
            ('classifier', cnn)
        ])
        X_train, y_train, _, _ = self.load_digits()
        cls1.fit(X_train, y_train)
        old_train_loss = cls1.named_steps['classifier'].loss_value_
        cnn.max_epochs_number = 5
        cnn.warm_start = True
        cls2 = Pipeline(steps=[
            ('normalizer', DigitsScaler()),
            ('reshaper', DigitsPreprocessor()),
            ('classifier', cnn)
        ])
        cls2.fit(X_train, y_train)
        new_train_loss = cls2.named_steps['classifier'].loss_value_
        self.assertGreater(old_train_loss, new_train_loss)

    def load_digits(self):
        """ Загрузить DIGITS и разбить его на обучающее и тестовое множества. """
        digits_dataset = load_digits()
        indices_for_training, indices_for_testing = split_train_data(
            digits_dataset.target, int(0.2 * digits_dataset.target.shape[0]),
            self.random_state
        )
        X_train = digits_dataset.data[indices_for_training]
        y_train = digits_dataset.target[indices_for_training]
        X_test = digits_dataset.data[indices_for_testing]
        y_test = digits_dataset.target[indices_for_testing]
        return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    unittest.main(verbosity=2)

