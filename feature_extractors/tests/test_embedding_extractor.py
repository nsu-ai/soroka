import os
import pickle
import unittest

import gensim
import numpy as np
import shutil
import urllib.request

from feature_extractors import EmbeddingExtractor
from sklearn.exceptions import NotFittedError


class TestEmbeddingExtractor(unittest.TestCase):
    def setUp(self):
        self.model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'word2vec_small.w2v')
        if not os.path.isfile(self.model_path):
            model_url = 'http://panchenko.me/data/dsl-backup/w2v-ru/all.norm-sz100-w10-cb0-it1-min100.w2v'
            with urllib.request.urlopen(model_url) as response, open(self.model_path, 'wb') as model_fp:
                shutil.copyfileobj(response, model_fp)
        self.tmp_extractor_name = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'tmp_extractor.pkl')
        self.fe = EmbeddingExtractor(word2vec_name=self.model_path, lowercase=True)
    
    def tearDown(self):
        if os.path.isfile(self.tmp_extractor_name):
            os.remove(self.tmp_extractor_name)
        if hasattr(self, 'fe'):
            del self.fe
    
    def test_fit_positive01(self):
        X_train = ['Ваш банк полный ацтой!', 'Ваш магаз — нормас', 'Мне пофиг на ваш ресторан']
        self.assertTrue(hasattr(self.fe, 'tokenizer'))
        self.assertTrue(hasattr(self.fe, 'word2vec_name'))
        self.assertTrue(hasattr(self.fe, 'lowercase'))
        self.assertFalse(hasattr(self.fe, 'word2vec_'))
        self.assertFalse(hasattr(self.fe, 'size_'))
        self.assertFalse(hasattr(self.fe, 'max_sentence_length_'))
        self.assertEqual(self.fe.word2vec_name, self.model_path)
        self.assertTrue(self.fe.lowercase)
        self.fe.fit(X_train)
        self.assertTrue(hasattr(self.fe, 'word2vec_'))
        self.assertTrue(hasattr(self.fe, 'size_'))
        self.assertTrue(hasattr(self.fe, 'max_sentence_length_'))
        self.assertIsInstance(self.fe.word2vec_, gensim.models.KeyedVectors)
        self.assertIsInstance(self.fe.size_, int)
        self.assertIsInstance(self.fe.max_sentence_length_, int)
        self.assertEqual(self.fe.size_, 100)
        self.assertGreater(self.fe.max_sentence_length_, 0)
    
    def test_fit_negative01(self):
        X_train = ['Ваш банк полный ацтой!', 'Ваш магаз — нормас', 'Мне пофиг на ваш ресторан']
        emb = EmbeddingExtractor(word2vec_name='this_path_is_not_expected_to_exist', lowercase=True)
        with self.assertRaises(FileNotFoundError):
            emb.fit(X_train)
    
    def test_transform_positive01(self):
        EPS = 1e-5
        X = np.array(['Ваш банк полный ацтой!', 'Ваш магаз — нормас', 'Мне пофиг на ваш ресторан'])
        X_transformed = self.fe.fit_transform(X)
        self.assertIsInstance(X_transformed, np.ndarray)
        self.assertEqual(len(X_transformed.shape), 4)
        self.assertEqual(X_transformed.shape[0], len(X))
        self.assertEqual(X_transformed.shape[1], 1)
        self.assertEqual(X_transformed.shape[2], self.fe.max_sentence_length_)
        self.assertEqual(X_transformed.shape[3], self.fe.size_ + 1)
        for sentence_ind in range(len(X)):
            n = min(self.fe.max_sentence_length_, len(self.fe.tokenizer(X[sentence_ind])))
            all_zeros = True
            for token_ind in range(n):
                self.assertAlmostEqual(X_transformed[sentence_ind, 0, token_ind, self.fe.size_], 1.0, delta=EPS,
                                       msg='Sentence {0}, token {1} from {2}'.format(sentence_ind, token_ind, n))
                if np.sum(np.square(X_transformed[sentence_ind, 0, token_ind, 0:self.fe.size_])) > EPS:
                    all_zeros = False
            self.assertFalse(all_zeros, msg='Sentence {0}'.format(sentence_ind,))
            for token_ind in range(n, self.fe.max_sentence_length_):
                self.assertAlmostEqual(X_transformed[sentence_ind, 0, token_ind].tolist(), [0.0] * (self.fe.size_+ 1),
                                       delta=EPS, msg='Sentence {0}, token {1}'.format(sentence_ind, token_ind))
    
    def test_transform_negative01(self):
        X = ['Ваш банк полный ацтой!', 'Ваш магаз — нормас', 'Мне пофиг на ваш ресторан']
        with self.assertRaises(NotFittedError):
            self.fe.transform(X)
        self.fe.fit(X)
        with self.assertRaises(TypeError):
            _ = self.fe.transform(1)
            
    def test_pickling_unpickling_positive01(self):
        X = ['Ваш банк полный ацтой!', 'Ваш магаз — нормас', 'Мне пофиг на ваш ресторан']
        output1 = self.fe.fit_transform(X)
        with open(self.tmp_extractor_name, 'wb') as f:
            pickle.dump(self.fe, f)
        with open(self.tmp_extractor_name, 'rb') as f:
            fe2 = pickle.load(f)
        output2 = fe2.transform(X)
        del fe2
        self.assertTrue(np.array_equal(output1, output2))
    
    
if __name__ == '__main__':
    unittest.main(verbosity=2)