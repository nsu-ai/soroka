import codecs
import copy
import csv
import math
import os
import pickle
import tempfile
import typing
import unittest

from sklearn.metrics import f1_score

from classifiers.cnn import CNNClassifier
from classifiers.text_cnn import TextCNNClassifier
from feature_extractors import EmbeddingExtractor


class TestCNNClassifier(unittest.TestCase):
    def setUp(self):
        cnn = CNNClassifier(layers={'conv': ((32, (3, 0)), (64, (2, 0))), 'pool': ((2, 1), (2, 1)), 'dense': (100,)},
                            batch_size=100)
        emb = EmbeddingExtractor(word2vec_name=os.path.join(os.path.dirname(__file__), '..', '..', 'data',
                                                            'word2vec_small.w2v'))
        self.text_cnn = TextCNNClassifier(feature_extractor=emb, base_estimator=cnn, batch_size=2000, warm_start=False)
        self.texts_for_training, self.labels_for_training = self.load_labeled_texts(
            os.path.join(os.path.dirname(__file__), 'testdata', 'train_data.csv'))
        self.texts_for_testing, self.labels_for_testing = self.load_labeled_texts(
            os.path.join(os.path.dirname(__file__), 'testdata', 'test_data.csv'))
        fp = tempfile.NamedTemporaryFile()
        self.classifier_name = fp.name
        fp.close()
        del fp

    def tearDown(self):
        if hasattr(self, 'text_cnn'):
            del self.text_cnn
        if os.path.isfile(self.classifier_name):
            os.remove(self.classifier_name)

    def test_fit_predict_positive01(self):
        max_epochs_number = 50
        self.text_cnn.base_estimator.max_epochs_number = max_epochs_number
        self.text_cnn.base_estimator.epochs_before_stopping = 3
        self.text_cnn.base_estimator.verbose = False
        res = self.text_cnn.fit(self.texts_for_training, self.labels_for_training,
                                validation=(self.texts_for_testing, self.labels_for_testing))
        self.assertIsInstance(res, TextCNNClassifier)
        self.assertGreater(self.text_cnn.base_estimator.n_iter_, 0)
        self.assertLess(self.text_cnn.base_estimator.n_iter_, max_epochs_number)
        f1 = f1_score(self.labels_for_testing, self.text_cnn.predict(self.texts_for_testing), average='macro')
        self.assertGreater(f1, 0.5)

    def test_fit_predict_positive02(self):
        max_epochs_number = 5
        true_n_iter_ = int(math.ceil(len(self.labels_for_training) / self.text_cnn.batch_size)) * max_epochs_number
        self.text_cnn.base_estimator.max_epochs_number = max_epochs_number
        self.text_cnn.base_estimator.epochs_before_stopping = 3
        self.text_cnn.base_estimator.verbose = True
        res = self.text_cnn.fit(self.texts_for_training, self.labels_for_training)
        self.assertIsInstance(res, TextCNNClassifier)
        self.assertEqual(self.text_cnn.base_estimator.n_iter_, true_n_iter_)
        f1 = f1_score(self.labels_for_testing, self.text_cnn.predict(self.texts_for_testing), average='macro')
        self.assertGreater(f1, 0.3)

    def test_pickling_unpickling(self):
        max_epochs_number = 5
        self.text_cnn.base_estimator.max_epochs_number = max_epochs_number
        self.text_cnn.base_estimator.epochs_before_stopping = 3
        self.text_cnn.base_estimator.verbose = False
        self.text_cnn.fit(self.texts_for_training, self.labels_for_training)
        y_pred_1 = self.text_cnn.predict(self.texts_for_testing)
        with open(self.classifier_name, 'wb') as fp:
            pickle.dump(self.text_cnn, fp)
        del self.text_cnn
        with open(self.classifier_name, 'rb') as fp:
            new_text_cnn = pickle.load(fp)
        y_pred_2 = new_text_cnn.predict(self.texts_for_testing)
        self.assertEqual(y_pred_1.shape, y_pred_2.shape)
        for sample_ind in range(y_pred_1.shape[0]):
            self.assertEqual(y_pred_1[sample_ind], y_pred_2[sample_ind])

    def test_deepcopy(self):
        max_epochs_number = 5
        self.text_cnn.base_estimator.max_epochs_number = max_epochs_number
        self.text_cnn.base_estimator.epochs_before_stopping = 3
        self.text_cnn.base_estimator.verbose = False
        self.text_cnn.fit(self.texts_for_training, self.labels_for_training)
        y_pred_1 = self.text_cnn.predict_proba(self.texts_for_testing)
        new_text_cnn = copy.deepcopy(self.text_cnn)
        y_pred_2 = new_text_cnn.predict_proba(self.texts_for_testing)
        self.assertEqual(y_pred_1.shape, y_pred_2.shape)
        for sample_ind in range(y_pred_1.shape[0]):
            for class_ind in range(y_pred_1.shape[1]):
                self.assertAlmostEqual(y_pred_1[sample_ind][class_ind], y_pred_2[sample_ind][class_ind])

    @staticmethod
    def load_labeled_texts(file_name: str) -> typing.Tuple[list, list]:
        texts = list()
        labels = list()
        min_label = None
        line_index = 1
        with codecs.open(file_name, mode='r', encoding='utf-8') as fp:
            csv_reader = csv.reader(fp, quotechar='"', delimiter=',')
            for cur in csv_reader:
                if len(cur) > 0:
                    assert len(cur) == 2, 'File "{0}": line {1} is wrong!'.format(file_name, line_index)
                    new_text = cur[0].strip()
                    new_label = int(cur[1])
                    if min_label is None:
                        min_label = new_label
                    else:
                        if new_label < min_label:
                            min_label = new_label
                    texts.append(new_text)
                    labels.append(new_label)
                line_index += 1
        for sample_ind in range(len(labels)):
            labels[sample_ind] -= min_label
        return texts, labels


if __name__ == '__main__':
    unittest.main(verbosity=2)
