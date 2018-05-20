import unittest
import os
import pickle
from sentiment_analyzer.train import train
from classifiers.cnn import CNNClassifier

class TestTrain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open('test_corpus.csv', 'w', encoding = 'utf-8') as f:
            f.write("ваш банк фигня :-(,-1\n"
                        "ну бар как бар,0\n"
                        "какой классный клуб,1")
        
        with open('wrong_corpus1.csv', 'w', encoding = 'utf-8') as f:
            f.write('казнить,нельзя помиловать,1')
        
        with open('wrong_corpus2.csv', 'w', encoding = 'utf-8') as f:
            f.write('казнить нельзя помиловать,2')
    
    def test_train_negative01(self):
        with self.assertRaises(ValueError):
            train(filename = 'wrong_corpus1.csv', verbose = True)
        
        with self.assertRaises(ValueError):
            train(filename = 'wrong_corpus2.csv', verbose = True)
    
    def test_train_positive01(self):
        train(filename = 'test_corpus.csv', max_epochs_number = 1, epochs_before_stopping = 1,
                layers = {'conv': [(2, (2, 2))], 'pool': [(2, 2)], 'dense': [2]},
                output = 'test_output.pkl', verbose = True)
        
        self.assertTrue(os.path.isfile('test_output.pkl'))
        with open('test_output.pkl', 'rb') as f:
            clf = pickle.load(f)
        
        self.assertIsInstance(clf, CNNClassifier)
        del clf
    
    @classmethod
    def tearDownClass(cls):
        os.remove('test_corpus.csv')
        os.remove('wrong_corpus1.csv')
        os.remove('wrong_corpus2.csv')
        try:
            os.remove('test_output.pkl')
        except FileNotFoundError:
            pass

if __name__ == '__main__':
    unittest.main(verbosity = 2)