import os
import pickle
import unittest

import numpy as np
import wget

from feature_extractors import ShapeExtractor, EmbeddingExtractor, SeqFeatureUnion


class TestSeqFeatureUnion(unittest.TestCase):
    def setUp(self):
        self.model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'word2vec_small.w2v')
        if not os.path.isfile(self.model_path):
            model_url = 'http://panchenko.me/data/dsl-backup/w2v-ru/all.norm-sz100-w10-cb0-it1-min100.w2v'
            wget.download(url=model_url, out=self.model_path)
        self.fe1 = EmbeddingExtractor(word2vec_name=self.model_path, tokenizer=str.split, lowercase=True)
        self.config_path = os.path.join(os.path.dirname(__file__), 'testdata', 'shape_config.json')
        self.fe2 = ShapeExtractor(config_name=self.config_path, tokenizer=str.split, max_number_of_shapes=100)
        self.tmp_extractor_name = os.path.join(os.path.dirname(__file__), 'testdata', 'tmp_union_extractor.pkl')

    def tearDown(self):
        if os.path.isfile(self.tmp_extractor_name):
            os.remove(self.tmp_extractor_name)
        if hasattr(self, 'fe1'):
            del self.fe1
        if hasattr(self, 'fe2'):
            del self.fe2

    def test_fit_transform_positive01(self):
        EPS = 1e-6
        feature_union = SeqFeatureUnion(transformer_list=[
            ('w2v', self.fe1),
            ('shape', self.fe2)
        ])
        X = ['АбвUp 123, 34+2 , рол hj №23 цена 34$', 'Hello , world !', 'Мама мыла раму % папа#2 ( кто это ) ,']
        X_trans_united = feature_union.fit_transform(X)
        X_trans_1 = feature_union.transformer_list[0][1].transform(X)
        X_trans_2 = feature_union.transformer_list[1][1].transform(X)
        self.assertIsInstance(X_trans_united, np.ndarray)
        self.assertEqual(len(X_trans_united.shape), 4)
        self.assertEqual(X_trans_united.shape[:3], X_trans_1.shape[:3])
        self.assertEqual(X_trans_united.shape[:3], X_trans_2.shape[:3])
        self.assertEqual(X_trans_united.shape[3], X_trans_1.shape[3] + X_trans_2.shape[3] - 1)
        for text_ind in range(X_trans_united.shape[0]):
            for token_ind in range(X_trans_united.shape[2]):
                self.assertAlmostEqual(X_trans_united[text_ind][0][token_ind][0], X_trans_1[text_ind][0][token_ind][-1],
                                       delta=EPS,
                                       msg='Feature extractor 0, text {0}, token {1}'.format(text_ind, token_ind))
                self.assertAlmostEqual(X_trans_united[text_ind][0][token_ind][0], X_trans_2[text_ind][0][token_ind][-1],
                                       delta=EPS,
                                       msg='Feature extractor 1, text {0}, token {1}'.format(text_ind, token_ind))
                for feature_ind in range(X_trans_1.shape[3] - 1):
                    self.assertAlmostEqual(X_trans_united[text_ind][0][token_ind][feature_ind + 1],
                                           X_trans_1[text_ind][0][token_ind][feature_ind], delta=EPS,
                                           msg='Feature extractor 0, text {0}, token {1}, feature {2}'.format(
                                               text_ind, token_ind, feature_ind))
                for feature_ind in range(X_trans_2.shape[3] - 1):
                    self.assertAlmostEqual(X_trans_united[text_ind][0][token_ind][feature_ind + X_trans_1.shape[3]],
                                           X_trans_2[text_ind][0][token_ind][feature_ind], delta=EPS,
                                           msg='Feature extractor 1, text {0}, token {1}, feature {2}'.format(
                                               text_ind, token_ind, feature_ind))

    def test_pickling_unpickling_positive01(self):
        X = ['АбвUp 123, 34+2 , рол hj №23 цена 34$', 'Hello , world !', 'Мама мыла раму % папа#2 ( кто это ) ,']
        feature_union1 = SeqFeatureUnion(transformer_list=[
            ('w2v', self.fe1),
            ('shape', self.fe2)
        ])
        output1 = feature_union1.fit_transform(X)
        with open(self.tmp_extractor_name, 'wb') as f:
            pickle.dump(feature_union1, f)
        del feature_union1
        with open(self.tmp_extractor_name, 'rb') as f:
            feature_union2 = pickle.load(f)
        output2 = feature_union2.transform(X)
        del feature_union2
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