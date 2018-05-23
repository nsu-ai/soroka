from argparse import ArgumentParser
import csv
import os
import pickle
import sys

from classifiers.cnn import CNNClassifier
from classifiers.text_cnn import TextCNNClassifier
from feature_extractors import EmbeddingExtractor


def parse_cnn_description(convolutional_layers_description: str, dense_layers_description: str) -> dict:
    conv_layers = list()
    pooling_layers = list()
    for cur in convolutional_layers_description.split(';'):
        err_msg = '"{0}" is a wrong description of convolutional layers!'.format(convolutional_layers_description)
        assert len(cur) > 0, err_msg
        parts_of_layer = cur.split('-')
        assert len(parts_of_layer) == 2, err_msg
        assert parts_of_layer[0].isdigit(), err_msg
        assert parts_of_layer[1].isdigit(), err_msg
        feature_maps_number = int(parts_of_layer[0])
        filter_size = int(parts_of_layer[1])
        assert (feature_maps_number > 0) and (filter_size > 1), err_msg
        conv_layers.append((feature_maps_number, (filter_size,0)))
        pooling_layers.append((3, 1))
    dense_layers = list()
    for cur in dense_layers_description.split(';'):
        err_msg = '"{0}" is a wrong description of dense layers!'.format(dense_layers_description)
        assert len(cur) > 0, err_msg
        assert cur.isdigit(), err_msg
        layer_size = int(cur)
        assert layer_size > 1, err_msg
        dense_layers.append(layer_size)
    return {'conv': tuple(conv_layers), 'pool': tuple(pooling_layers), 'dense': tuple(dense_layers)}


def main():
    parser = ArgumentParser()
    parser.add_argument('-t', '--train', dest='trainining_set_name', type=str, required=True,
                        help='Name of CSV file with data for training.')
    parser.add_argument('-e', '--est', dest='estimation_set_name', type=str, required=True,
                        help='Name of CSV file with data for estimation.')
    parser.add_argument('-c', '--conv', dest='convolutional_layers', type=str, required=False,
                        default='32-4;64-3;128-2', help='Description of CNN\'s convolutional layers.')
    parser.add_argument('-d', '--dense', dest='dense_layers', type=str, required=False, default='256;128')
    args = parser.parse_args()

    filename_for_training = os.path.normpath(args.trainining_set_name)
    assert os.path.isfile(filename_for_training), 'File "{0}" does not exist!'.format(filename_for_training)
    filename_for_testing = os.path.normpath(args.estimation_set_name)
    assert os.path.isfile(filename_for_testing), 'File "{0}" does not exist!'.format(filename_for_testing)

    cnn_structure = parse_cnn_description(args.convolutional_layers, args.dense_layers)
    epochs_before_stopping = 5
    max_epochs_number = 200
    word2vec_name = os.path.join('data', 'word2vec_small.w2v')
    assert os.path.isfile(word2vec_name), 'File "{0}" does not exists!'.format(word2vec_name)
    cls_name = os.path.join('data', 'senti_cnn_classifier.pkl')
    cls_dir = os.path.dirname(cls_name)
    assert os.path.isdir(cls_dir), 'Directory "{0}" does not exists!'.format(cls_dir)

    maxInt = sys.maxsize
    decrement = True

    while decrement:
        decrement = False
        try:
            csv.field_size_limit(maxInt)
        except OverflowError:
            maxInt = int(maxInt / 10)
            decrement = True

    X_train = list()
    y_train = list()
    n_positives = 0
    n_negatives = 0
    n_neutrals = 0
    with open(filename_for_training, 'r', encoding='utf-8') as f:
        reader=csv.reader(f, delimiter=',', quotechar='"')
        for line_counter, row in enumerate(reader):
            if len(row) != 2:
                raise ValueError("Row must contain 2 values, but "
                        "row #{number} {row} has {values} values".format(number=line_counter, 
                        row=row, values=len(row)))
            if row[1] not in ['-1', '0', '1']:
                raise ValueError("Class label must be -1, 0, or 1, but class label"
                        " at row #{number} is {label}".format(number=line_counter,
                        label=row[1]))
            X_train.append(row[0])
            y_train.append(int(row[1]) + 1)
            if row[1] == '1':
                n_positives += 1
            elif row[1] == '-1':
                n_negatives += 1
            else:
                n_neutrals += 1
    print('Data for training have been successfully loaded...')
    print('Number of positive texts is {0}.'.format(n_positives))
    print('Number of negative texts is {0}.'.format(n_negatives))
    print('Number of neutrals texts is {0}.'.format(n_neutrals))
    print('')
    n_positives = 0
    n_negatives = 0
    n_neutrals = 0
    X_test = list()
    y_test = list()
    with open(filename_for_testing, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        for line_counter, row in enumerate(reader):
            if len(row) != 2:
                raise ValueError("Row must contain 2 values, but "
                                 "row #{number} {row} has {values} values".format(number=line_counter,
                                                                                  row=row, values=len(row)))
            if row[1] not in ['-1', '0', '1']:
                raise ValueError("Class label must be -1, 0, or 1, but class label"
                                 " at row #{number} is {label}".format(number=line_counter,
                                                                       label=row[1]))
            X_test.append(row[0])
            y_test.append(int(row[1]) + 1)
            if row[1] == '1':
                n_positives += 1
            elif row[1] == '-1':
                n_negatives += 1
            else:
                n_neutrals += 1
    print('Data for estimation have been successfully loaded...')
    print('Number of positive texts is {0}.'.format(n_positives))
    print('Number of negative texts is {0}.'.format(n_negatives))
    print('Number of neutrals texts is {0}.'.format(n_neutrals))
    print('')

    emb = EmbeddingExtractor(word2vec_name=word2vec_name)
    cls = CNNClassifier(layers=cnn_structure, max_epochs_number=max_epochs_number,
                        epochs_before_stopping=epochs_before_stopping, verbose=True, eval_metric='F1', batch_size=50)
    text_cnn = TextCNNClassifier(feature_extractor=emb, base_estimator=cls, batch_size=2000)
    text_cnn.fit(X_train, y_train, validation=(X_test, y_test))
    with open(cls_name, 'wb') as fp:
        pickle.dump(text_cnn, fp)


if __name__ == '__main__':
    main()