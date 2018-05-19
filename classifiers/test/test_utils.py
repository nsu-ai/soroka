import re
import unittest

import numpy
from sklearn.utils.validation import check_random_state

from classifiers.utils import split_train_data, iterate


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.random_state = check_random_state(0)
        self.y = numpy.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 2, 1, 1, 1, 1, 0], dtype=numpy.int32)

    def tearDown(self):
        del self.random_state
        del self.y

    def test_split_train_data_positive01(self):
        """ Корректность разделения на обучающее и тестовое множества на корректных данных при заданном рандом-сиде. """
        n_validation = 6
        true_indices_for_training = [18, 1, 19, 8, 10, 17, 6, 13, 4, 2, 5, 14, 9, 7]
        true_indices_for_testing = [16, 11, 3, 0, 15, 12]
        indices_for_training, indices_for_testing = split_train_data(self.y, n_validation, self.random_state)
        self.assertIsInstance(indices_for_training, numpy.ndarray)
        self.assertIsInstance(indices_for_testing, numpy.ndarray)
        self.assertEqual((n_validation,), indices_for_testing.shape)
        self.assertEqual((self.y.shape[0] - n_validation,), indices_for_training.shape)
        self.assertEqual(true_indices_for_training, indices_for_training.tolist())
        self.assertEqual(true_indices_for_testing, indices_for_testing.tolist())

    def test_split_train_data_negative01(self):
        """ Корректность разделения на обучающее и тестовое множества, если n_validation нулевое. """
        n_validation = 0
        true_err_msg = re.escape('Train data cannot be split into train and validation subsets!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _ = split_train_data(self.y, n_validation, self.random_state)

    def test_split_train_data_negative02(self):
        """ Корректность разделения на обучающее и тестовое множества, если n_validation равно числу всех примеров. """
        n_validation = self.y.shape[0]
        true_err_msg = re.escape('Train data cannot be split into train and validation subsets!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _ = split_train_data(self.y, n_validation, self.random_state)

    def test_split_train_data_negative03(self):
        """ Корректность разделения на обучающее и тестовое множества, если n_validation меньше числа классов. """
        n_validation = 2
        true_err_msg = re.escape('Train data cannot be split into train and validation subsets!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _ = split_train_data(self.y, n_validation, self.random_state)

    def test_split_train_data_negative04(self):
        """ Корректность разделения на обучающее и тестовое множества, если n_validation слишком велико. """
        n_validation = self.y.shape[0] - 2
        true_err_msg = re.escape('Train data cannot be split into train and validation subsets!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _ = split_train_data(self.y, n_validation, self.random_state)

    def test_iterate_positive01(self):
        """ Входные данные корректны, длина датасета делится на размер батча нацело, итерируемся последовательно. """
        source_indices = numpy.arange(3, self.y.shape[0] + 3, 1, dtype=numpy.int32)
        batch_size = 5
        true_batches = [
            numpy.arange(3 + 0, 3 + 5, 1, dtype=numpy.int32),
            numpy.arange(3 + 5, 3 + 10, 1, dtype=numpy.int32),
            numpy.arange(3 + 10, 3 + 15, 1, dtype=numpy.int32),
            numpy.arange(3 + 15, 3 + 20, 1, dtype=numpy.int32)
        ]
        calculated_batches = list(iterate(source_indices, batchsize=batch_size, shuffle=False,
                                          random_state=self.random_state))
        self.assertEqual(len(true_batches), len(calculated_batches))
        for batch_ind in range(len(true_batches)):
            self.assertIsInstance(calculated_batches[batch_ind], numpy.ndarray)
            self.assertEqual((batch_size,), calculated_batches[batch_ind].shape)
            self.assertEqual(true_batches[batch_ind].tolist(), calculated_batches[batch_ind].tolist())

    def test_iterate_positive02(self):
        """ Входные данные корректны, длина датасета НЕ делится на размер батча нацело, итерируемся последовательно. """
        source_indices = numpy.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
                                     dtype=numpy.int32)
        batch_size = 3
        true_batches = [
            numpy.array([3, 4, 5], dtype=numpy.int32),
            numpy.array([6, 7, 8], dtype=numpy.int32),
            numpy.array([9, 10, 11], dtype=numpy.int32),
            numpy.array([12, 13, 14], dtype=numpy.int32),
            numpy.array([15, 16, 17], dtype=numpy.int32),
            numpy.array([18, 19, 20], dtype=numpy.int32),
            numpy.array([21, 22, 3], dtype=numpy.int32)
        ]
        calculated_batches = list(iterate(source_indices, batchsize=batch_size, shuffle=False,
                                          random_state=self.random_state))
        self.assertEqual(len(true_batches), len(calculated_batches))
        for batch_ind in range(len(true_batches)):
            self.assertIsInstance(calculated_batches[batch_ind], numpy.ndarray)
            self.assertEqual((batch_size,), calculated_batches[batch_ind].shape)
            self.assertEqual(true_batches[batch_ind].tolist(), calculated_batches[batch_ind].tolist())

    def test_iterate_positive03(self):
        """ Входные данные корректны, длина датасета делится на размер батча нацело, итерируемся случайно. """
        source_indices = numpy.arange(3, self.y.shape[0] + 3, 1, dtype=numpy.int32)
        batch_size = 5
        true_batches = [
            numpy.array([21, 4, 22, 11, 13], dtype=numpy.int32),
            numpy.array([20, 9, 16, 7, 5], dtype=numpy.int32),
            numpy.array([8, 17, 12, 10, 19], dtype=numpy.int32),
            numpy.array([14, 6, 3, 18, 15], dtype=numpy.int32),
        ]
        calculated_batches = list(iterate(source_indices, batchsize=batch_size, shuffle=True,
                                          random_state=self.random_state))
        self.assertEqual(len(true_batches), len(calculated_batches))
        for batch_ind in range(len(true_batches)):
            self.assertIsInstance(calculated_batches[batch_ind], numpy.ndarray)
            self.assertEqual((batch_size,), calculated_batches[batch_ind].shape)
            self.assertEqual(true_batches[batch_ind].tolist(), calculated_batches[batch_ind].tolist())

    def test_iterate_positive04(self):
        """ Входные данные корректны, длина датасета НЕ делится на размер батча нацело, итерируемся случайно. """
        source_indices = numpy.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
                                     dtype=numpy.int32)
        batch_size = 3
        true_batches = [
            numpy.array([21, 4, 22], dtype=numpy.int32),
            numpy.array([11, 13, 20], dtype=numpy.int32),
            numpy.array([9, 16, 7], dtype=numpy.int32),
            numpy.array([5, 8, 17], dtype=numpy.int32),
            numpy.array([12, 10, 19], dtype=numpy.int32),
            numpy.array([14, 6, 3], dtype=numpy.int32),
            numpy.array([18, 15, 21], dtype=numpy.int32)
        ]
        calculated_batches = list(iterate(source_indices, batchsize=batch_size, shuffle=True,
                                          random_state=self.random_state))
        self.assertEqual(len(true_batches), len(calculated_batches))
        for batch_ind in range(len(true_batches)):
            self.assertIsInstance(calculated_batches[batch_ind], numpy.ndarray)
            self.assertEqual((batch_size,), calculated_batches[batch_ind].shape)
            self.assertEqual(true_batches[batch_ind].tolist(), calculated_batches[batch_ind].tolist())

    def test_iterate_negative01(self):
        """ Входные данные некорректны: индексы не являются numpy.ndarray-массивом. """
        source_indices = numpy.arange(3, self.y.shape[0] + 3, 1, dtype=numpy.int32).tolist()
        batch_size = 5
        true_err_msg = re.escape('`indices` must be a numpy.ndarray!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _ = list(iterate(source_indices, batchsize=batch_size, shuffle=False, random_state=self.random_state))

    def test_iterate_negative02(self):
        """ Входные данные некорректны: индексы являются numpy.ndarray-массивом, но двумерным, а не одномерным. """
        source_indices = numpy.array(
            [
                [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
                [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]
            ],
            dtype=numpy.int32
        )
        batch_size = 5
        true_err_msg = re.escape('`indices` must be a 1-D array!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _ = list(iterate(source_indices, batchsize=batch_size, shuffle=False, random_state=self.random_state))

    def test_iterate_negative03(self):
        """ Входные данные некорректны: индексы заданы 1-мерным numpy.ndarray-массивом, но как вещественные числа. """
        source_indices = numpy.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
                                     dtype=numpy.float32)
        batch_size = 5
        true_err_msg = re.escape('Items of the `indices` array must be integer numbers!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _ = list(iterate(source_indices, batchsize=batch_size, shuffle=False, random_state=self.random_state))


if __name__ == '__main__':
    unittest.main(verbosity=2)
