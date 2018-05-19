import unittest
import os
import gensim
import pickle
import numpy as np
from feature_extractors import EmbeddingExtractor
from sklearn.exceptions import NotFittedError

class TestEmbeddingExtractor(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join('data', 'word2vec.w2v')
    
    def tearDown(self):
        pass
    
    def test_fit_positive01(self):
        emb = EmbeddingExtractor(self.path)
        self.assertIsInstance(emb.word2vec, gensim.models.KeyedVectors)
        del emb
    
    def test_fit_negative01(self):
        with self.assertRaises(FileNotFoundError):
            emb = EmbeddingExtractor('this_path_is_not_expected_to_exist')
    
    def test_transform_positive01(self):
        X = ['Ваш банк полный ацтой!', 'Ваш магаз — нормас', 'Мне пофиг на ваш ресторан']
        emb = EmbeddingExtractor(self.path).fit(X)
        X_transformed = emb.transform(X)
        self.assertEqual(len(X_transformed.shape), 4)
        self.assertEqual(X_transformed.shape[0], len(X))
        self.assertEqual(X_transformed.shape[1], 1)
        self.assertEqual(X_transformed.shape[2], emb.max_sentence_length)
        self.assertEqual(X_transformed.shape[3], emb.size + 1)
        del emb
    
    def test_transform_negative01(self):
        X = ['Ваш банк полный ацтой!', 'Ваш магаз — нормас', 'Мне пофиг на ваш ресторан']
        with self.assertRaises(TypeError):
            emb = EmbeddingExtractor(self.path).fit()
            emb.transform(1)
            del emb
        
        with self.assertRaises(NotFittedError):
            EmbeddingExtractor(self.path).transform(X)
            
    def test_pickling_unpickling_positive01(self):
        X = ['Ваш банк полный ацтой!', 'Ваш магаз — нормас', 'Мне пофиг на ваш ресторан']
        emb1 = EmbeddingExtractor(self.path).fit(X)
        output1 = emb1.transform(X)
        with open('emb.pkl', 'wb') as f:
            pickle.dump(emb1, f)
        
        del emb1
        
        with open('emb.pkl', 'rb') as f:
            emb2 = pickle.load(f)
        
        os.remove('emb.pkl')
        output2 = emb2.transform(X)
        self.assertTrue(np.array_equal(output1, output2))
    
    
if __name__ == '__main__':
    unittest.main(verbosity = 2)