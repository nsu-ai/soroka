import unittest
import pickle
import os
import numpy as np
from collections import OrderedDict
from sentiment_analyzer import SentimentAnalyzer
from classifiers.cnn import CNNClassifier
from feature_extractors import EmbeddingExtractor

class TestSentimentAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X = ['Ваш банк полный ацтой!', 'Ваш магаз — нормас', 'Мне пофиг на ваш ресторан']
        y = [0, 2, 1]
        cls.classifier = CNNClassifier(epochs_before_stopping = 1, verbose = True, max_epochs_number = 1, layers = {'conv': [(2, (2, 2))], 'pool': [(2, 2)], 'dense': [2]})
        cls.feature_extractor = EmbeddingExtractor().fit(X)
        X_transformed = cls.feature_extractor.transform(X)
        cls.classifier.fit(X_transformed, y)
    
    @classmethod
    def tearDownClass(cls):
        del cls.classifier
        del cls.feature_extractor
        if os.path.isfile('sent.pkl'):
            os.remove('sent.pkl')
    
    def test_init_positive01(self):
        sent = SentimentAnalyzer(feature_extractor = self.feature_extractor, classifier = self.classifier)
        del sent
    
    def test_init_negative01(self):
        with self.assertRaises(TypeError):
            sent = SentimentAnalyzer(feature_extractor = 1, classifier = self.classifier)
        
        with self.assertRaises(TypeError):
            sent = SentimentAnalyzer(feature_extractor = self.feature_extractor, classifier = 1)
    
    def test_analyze_positive01(self):
        X = OrderedDict([
            ('Первый сайт', ['Ваш банк полный ацтой!', 'Ваш магаз — нормас']),
            ('Второй сайт', ['Мне пофиг на ваш ресторан'])
            ])
        len_X = sum([len(X[key]) for key in X])
        sent = SentimentAnalyzer(feature_extractor = self.feature_extractor, classifier = self.classifier)
        output = sent.analyze(X)
        self.assertIsInstance(output, tuple)
        self.assertEqual(len(output), 3)
        self.assertIsInstance(output[0], int)
        self.assertIsInstance(output[1], int)
        self.assertIsInstance(output[2], int)
        self.assertEqual(output[0] + output[1] + output[2], len_X)
        del sent
    
    def test_analyze_negative01(self):
        with self.assertRaises(TypeError):
            sent = SentimentAnalyzer(feature_extractor = self.feature_extractor, classifier = self.classifier)
            sent.analyze(1)            
    
    def test_pickle_unpickle_positive01(self):
        X = OrderedDict({
            'Первый сайт': ['Ваш банк полный ацтой!', 'Ваш магаз — нормас'],
            'Второй сайт': ['Мне пофиг на ваш ресторан']
            })
        sent = SentimentAnalyzer(feature_extractor = self.feature_extractor, classifier = self.classifier)
        output1 = sent.analyze(X)
        with open('sent.pkl', 'wb') as f:
            pickle.dump(sent, f)
        del sent
        with open('sent.pkl', 'rb') as f:
            sent = pickle.load(f)
        output2 = sent.analyze(X)
        self.assertEqual(output1, output2)
        del sent
    
        
if __name__ == '__main__':
    unittest.main(verbosity = 2)