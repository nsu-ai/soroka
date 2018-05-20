import csv
import os
import pickle
from classifiers.cnn import CNNClassifier
from feature_extractors import EmbeddingExtractor


def main(filename_for_training=os.path.join('data', 'train_data_for_se.csv'),
         filename_for_testing=os.path.join('data', 'test_data_for_se.csv'),
         delimiter=',', verbose=True, epochs_before_stopping=10, max_epochs_number=1000,
         layers={'conv': ((32, (3, 0)), (64, (2, 0))), 'pool': ((2, 1), (2, 1)), 'dense': (100,)},
         word2vec=os.path.join('data', 'word2vec_small.w2v'),
         fe_output=os.path.join('data', 'senti_feature_extractor.pkl'),
         cls_output=os.path.join('data', 'senti_cnn_classifier.pkl')):

    X_train = list()
    y_train = list()
    with open(filename_for_training, 'r', encoding='utf-8') as f:
        reader=csv.reader(f, delimiter=delimiter, quotechar='"')
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
    X_test = list()
    y_test = list()
    with open(filename_for_testing, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=delimiter, quotechar='"')
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

    emb = EmbeddingExtractor(word2vec_name=word2vec)
    emb.fit(X_train + X_test)
    print('Feature extractor has been fitted...')
    with open(fe_output, 'wb') as f:
        pickle.dump(emb, f)
    X_train_transformed = emb.transform(X_train)
    print('Data for training have been prepared...')
    X_test_transformed = emb.transform(X_test)
    print('Data for testing have been prepared...')
    print('')

    clf = CNNClassifier(verbose=verbose, epochs_before_stopping=epochs_before_stopping,
                        max_epochs_number=max_epochs_number, layers=layers, eval_metric='F1').fit(
        X_train_transformed, y_train,
        validation=(X_test_transformed, y_test)
    )
    print('')
    print('Convolutional neural network has been trained...')
    with open(cls_output, 'wb') as f:
        pickle.dump(clf, f)


if __name__ == '__main__':
    main()