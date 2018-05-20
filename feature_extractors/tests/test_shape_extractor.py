import os
import pickle
import unittest

import numpy as np

from feature_extractors import ShapeExtractor
from sklearn.exceptions import NotFittedError


class TestShapeExtractor(unittest.TestCase):
    def setUp(self):
        self.config_path = os.path.join(os.path.dirname(__file__), 'testdata', 'shape_config.json')
        self.tmp_extractor_name = os.path.join(os.path.dirname(__file__), 'testdata', 'tmp_shape_extractor.pkl')
        self.fe = ShapeExtractor(config_name=self.config_path, tokenizer=str.split, max_number_of_shapes=100)

    def tearDown(self):
        if os.path.isfile(self.tmp_extractor_name):
            os.remove(self.tmp_extractor_name)
        if hasattr(self, 'fe'):
            del self.fe

    def test_fit_positive01(self):
        X_train = ['АбвUp 123, 34+2 , рол hj №23 цена 34$', 'Hello , world !', 'Мама мыла раму % папа#2 (кто это) ,']
        true_characters = {
            "А": "Cu", "Б": "Cu", "В": "Cu", "Г": "Cu", "Д": "Cu", "Е": "Cu", "Ё": "Cu", "Ж": "Cu", "З": "Cu",
            "И": "Cu", "Й": "Cu", "К": "Cu", "Л": "Cu", "М": "Cu", "Н": "Cu", "О": "Cu", "П": "Cu", "Р": "Cu",
            "С": "Cu", "Т": "Cu", "У": "Cu", "Ф": "Cu", "Х": "Cu", "Ц": "Cu", "Ч": "Cu", "Ш": "Cu", "Щ": "Cu",
            "Ъ": "Cu", "Ы": "Cu", "Ь": "Cu", "Э": "Cu", "Ю": "Cu", "Я": "Cu", "а": "Cl", "б": "Cl", "в": "Cl",
            "г": "Cl", "д": "Cl", "е": "Cl", "ё": "Cl", "ж": "Cl", "з": "Cl", "и": "Cl", "й": "Cl", "к": "Cl",
            "л": "Cl", "м": "Cl", "н": "Cl", "о": "Cl", "п": "Cl", "р": "Cl", "с": "Cl", "т": "Cl", "у": "Cl",
            "ф": "Cl", "х": "Cl", "ц": "Cl", "ч": "Cl", "ш": "Cl", "щ": "Cl", "ъ": "Cl", "ы": "Cl", "ь": "Cl",
            "э": "Cl", "ю": "Cl", "я": "Cl", "A": "Lu", "B": "Lu", "C": "Lu", "D": "Lu", "E": "Lu", "F": "Lu",
            "G": "Lu", "H": "Lu", "I": "Lu", "J": "Lu", "K": "Lu", "L": "Lu", "M": "Lu", "N": "Lu", "O": "Lu",
            "P": "Lu", "Q": "Lu", "R": "Lu", "S": "Lu", "T": "Lu", "U": "Lu", "V": "Lu", "W": "Lu", "X": "Lu",
            "Y": "Lu", "Z": "Lu", "a": "Ll", "b": "Ll", "c": "Ll", "d": "Ll", "e": "Ll", "f": "Ll", "g": "Ll",
            "h": "Ll", "i": "Ll", "j": "Ll", "k": "Ll", "l": "Ll", "m": "Ll", "n": "Ll", "o": "Ll", "p": "Ll",
            "q": "Ll", "r": "Ll", "s": "Ll", "t": "Ll", "u": "Ll", "v": "Ll", "w": "Ll", "x": "Ll", "y": "Ll",
            "z": "Ll", "0": "D", "1": "D", "2": "D", "3": "D", "4": "D", "5": "D", "6": "D", "7": "D", "8": "D",
            "9": "D", ",": "P", ".": "P", "/": "P", "\\": "P", ";": "P", ":": "P", "!": "P", "?": "P", "-": "Da",
            "–": "Da", "—": "Da", "‒": "Da", "―": "Da", "‐": "Da", "‑": "Da", "\"": "Q", "'": "Q", "`": "Q", "«": "Q",
            "»": "Q", "„": "Q", "“": "Q", "”": "Q", "‘": "Q", "’": "Q", "『": "Q", "』": "Q", "「": "Q", "」": "Q",
            "(": "Bl", "[": "Bl", "{": "Bl", "<": "Bl", ")": "Br", "]": "Br", "}": "Br", ">": "Br", "@": "E", "$": "M",
            "€": "M", "₽": "M", "¥": "M", "₣": "M", "£": "M", "№": "No"
        }
        true_vocabulary= {
            "BlCl": 0,
            "Cl": 1,
            "ClBr": 2,
            "ClUnkD": 3,
            "CuCl": 4,
            "CuClLuLl": 5,
            "DM": 6,
            "DP": 7,
            "DUnkD": 8,
            "Ll": 9,
            "LuLl": 10,
            "NoD": 11,
            "P": 12,
            "Unk": 13
        }
        self.assertTrue(hasattr(self.fe, 'tokenizer'))
        self.assertTrue(hasattr(self.fe, 'config_name'))
        self.assertTrue(hasattr(self.fe, 'max_number_of_shapes'))
        self.assertFalse(hasattr(self.fe, 'characters_'))
        self.assertFalse(hasattr(self.fe, 'vocabulary_'))
        self.assertFalse(hasattr(self.fe, 'max_sentence_length_'))
        self.assertEqual(self.fe.config_name, self.config_path)
        self.assertEqual(self.fe.max_number_of_shapes, 100)
        self.fe.fit(X_train)
        self.assertTrue(hasattr(self.fe, 'characters_'))
        self.assertTrue(hasattr(self.fe, 'vocabulary_'))
        self.assertTrue(hasattr(self.fe, 'max_sentence_length_'))
        self.assertIsInstance(self.fe.characters_, dict)
        self.assertIsInstance(self.fe.vocabulary_, dict)
        self.assertIsInstance(self.fe.max_sentence_length_, int)
        self.assertEqual(self.fe.max_sentence_length_, 9)
        self.assertEqual(self.fe.characters_, true_characters)
        self.assertEqual(self.fe.vocabulary_, true_vocabulary)

    def test_fit_positive02(self):
        X_train = ['АбвUp 123, 34+2 , рол hj №23 цена 34$', 'Hello , world !', 'Мама мыла раму % папа#2 (кто это) ,']
        true_characters = {
            "А": "Cu", "Б": "Cu", "В": "Cu", "Г": "Cu", "Д": "Cu", "Е": "Cu", "Ё": "Cu", "Ж": "Cu", "З": "Cu",
            "И": "Cu", "Й": "Cu", "К": "Cu", "Л": "Cu", "М": "Cu", "Н": "Cu", "О": "Cu", "П": "Cu", "Р": "Cu",
            "С": "Cu", "Т": "Cu", "У": "Cu", "Ф": "Cu", "Х": "Cu", "Ц": "Cu", "Ч": "Cu", "Ш": "Cu", "Щ": "Cu",
            "Ъ": "Cu", "Ы": "Cu", "Ь": "Cu", "Э": "Cu", "Ю": "Cu", "Я": "Cu", "а": "Cl", "б": "Cl", "в": "Cl",
            "г": "Cl", "д": "Cl", "е": "Cl", "ё": "Cl", "ж": "Cl", "з": "Cl", "и": "Cl", "й": "Cl", "к": "Cl",
            "л": "Cl", "м": "Cl", "н": "Cl", "о": "Cl", "п": "Cl", "р": "Cl", "с": "Cl", "т": "Cl", "у": "Cl",
            "ф": "Cl", "х": "Cl", "ц": "Cl", "ч": "Cl", "ш": "Cl", "щ": "Cl", "ъ": "Cl", "ы": "Cl", "ь": "Cl",
            "э": "Cl", "ю": "Cl", "я": "Cl", "A": "Lu", "B": "Lu", "C": "Lu", "D": "Lu", "E": "Lu", "F": "Lu",
            "G": "Lu", "H": "Lu", "I": "Lu", "J": "Lu", "K": "Lu", "L": "Lu", "M": "Lu", "N": "Lu", "O": "Lu",
            "P": "Lu", "Q": "Lu", "R": "Lu", "S": "Lu", "T": "Lu", "U": "Lu", "V": "Lu", "W": "Lu", "X": "Lu",
            "Y": "Lu", "Z": "Lu", "a": "Ll", "b": "Ll", "c": "Ll", "d": "Ll", "e": "Ll", "f": "Ll", "g": "Ll",
            "h": "Ll", "i": "Ll", "j": "Ll", "k": "Ll", "l": "Ll", "m": "Ll", "n": "Ll", "o": "Ll", "p": "Ll",
            "q": "Ll", "r": "Ll", "s": "Ll", "t": "Ll", "u": "Ll", "v": "Ll", "w": "Ll", "x": "Ll", "y": "Ll",
            "z": "Ll", "0": "D", "1": "D", "2": "D", "3": "D", "4": "D", "5": "D", "6": "D", "7": "D", "8": "D",
            "9": "D", ",": "P", ".": "P", "/": "P", "\\": "P", ";": "P", ":": "P", "!": "P", "?": "P", "-": "Da",
            "–": "Da", "—": "Da", "‒": "Da", "―": "Da", "‐": "Da", "‑": "Da", "\"": "Q", "'": "Q", "`": "Q", "«": "Q",
            "»": "Q", "„": "Q", "“": "Q", "”": "Q", "‘": "Q", "’": "Q", "『": "Q", "』": "Q", "「": "Q", "」": "Q",
            "(": "Bl", "[": "Bl", "{": "Bl", "<": "Bl", ")": "Br", "]": "Br", "}": "Br", ">": "Br", "@": "E", "$": "M",
            "€": "M", "₽": "M", "¥": "M", "₣": "M", "£": "M", "№": "No"
        }
        true_vocabulary= {
            "BlCl": 0,
            "Cl": 1,
            "Ll": 2,
            "P": 3
        }
        self.assertTrue(hasattr(self.fe, 'tokenizer'))
        self.assertTrue(hasattr(self.fe, 'config_name'))
        self.assertTrue(hasattr(self.fe, 'max_number_of_shapes'))
        self.assertFalse(hasattr(self.fe, 'characters_'))
        self.assertFalse(hasattr(self.fe, 'vocabulary_'))
        self.assertFalse(hasattr(self.fe, 'max_sentence_length_'))
        self.assertEqual(self.fe.config_name, self.config_path)
        self.assertEqual(self.fe.max_number_of_shapes, 100)
        self.fe.max_number_of_shapes = 4
        self.fe.fit(X_train)
        self.assertTrue(hasattr(self.fe, 'characters_'))
        self.assertTrue(hasattr(self.fe, 'vocabulary_'))
        self.assertTrue(hasattr(self.fe, 'max_sentence_length_'))
        self.assertIsInstance(self.fe.characters_, dict)
        self.assertIsInstance(self.fe.vocabulary_, dict)
        self.assertIsInstance(self.fe.max_sentence_length_, int)
        self.assertEqual(self.fe.max_sentence_length_, 9)
        self.assertEqual(self.fe.characters_, true_characters)
        self.assertEqual(self.fe.vocabulary_, true_vocabulary)

    def test_transform_positive01(self):
        EPS = 1e-5
        X_src = ['АбвUp 123, 34+2 , рол hj №23 цена 34$', 'Hello , world !', 'Мама мыла раму % папа#2 (кто это) ,']
        true_data = [
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0],
                    [0.0, 1.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                ],
            ],
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0]
                ],
            ],
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0]
                ],
            ],
        ]
        true_X_transformed = np.array(true_data, dtype=np.float32)
        self.fe.max_number_of_shapes = 4
        X_transformed = self.fe.fit_transform(X_src)
        self.assertIsInstance(X_transformed, np.ndarray)
        self.assertEqual(X_transformed.shape, true_X_transformed.shape)
        for text_ind in range(true_X_transformed.shape[0]):
            for token_ind in range(true_X_transformed.shape[2]):
                for feature_ind in range(true_X_transformed.shape[3]):
                    self.assertAlmostEqual(X_transformed[text_ind][0][token_ind][feature_ind],
                                           true_X_transformed[text_ind][0][token_ind][feature_ind], delta=EPS,
                                           msg='Text {0}, token {1}, feature {2}'.format(text_ind, token_ind,
                                                                                         feature_ind))

    def test_transform_negative01(self):
        X = ['АбвUp 123, 34+2 , рол hj №23 цена 34$', 'Hello , world !', 'Мама мыла раму % папа#2 (кто это) ,']
        with self.assertRaises(NotFittedError):
            self.fe.transform(X)
        self.fe.fit(X)
        with self.assertRaises(TypeError):
            _ = self.fe.transform(1)

    def test_pickling_unpickling_positive01(self):
        X = ['АбвUp 123, 34+2 , рол hj №23 цена 34$', 'Hello , world !', 'Мама мыла раму % папа#2 (кто это) ,']
        output1 = self.fe.fit_transform(X)
        with open(self.tmp_extractor_name, 'wb') as f:
            pickle.dump(self.fe, f)
        with open(self.tmp_extractor_name, 'rb') as f:
            fe2 = pickle.load(f)
        output2 = fe2.transform(X)
        del fe2
        EPS = 1e-5
        self.assertEqual(output1.shape, output2.shape)
        for text_ind in range(output1.shape[0]):
            for token_ind in range(output1.shape[2]):
                for feature_ind in range(output1.shape[3]):
                    self.assertAlmostEqual(
                        output1[text_ind, 0, token_ind, feature_ind], output2[text_ind, 0, token_ind, feature_ind],
                        delta=EPS, msg='Sentence {0}, token {1}, feature {2}'.format(text_ind, token_ind, feature_ind)
                    )


if __name__ == '__main__':
    unittest.main(verbosity=2)