import csv
import os
import pickle
from classifiers.cnn import CNNClassifier
from feature_extractors import EmbeddingExtractor

def main(filename = os.path.join('..', '..', 'data', 'corpus.csv'), delimiter = ',', verbose = False, 
            epochs_before_stopping = 10, max_epochs_number = 1000,
            layers = {'conv': ((32, (3, 3)), (64, (2, 2))), 'pool': ((2, 2), (2, 2)), 'dense': (100, 80)},
            word2vec = os.path.join('data', 'word2vec_main.w2v'),
            output = os.path.join('data', 'senti_cnn_classifier.pkl')):
    
    with open(filename, 'r', encoding = 'utf-8') as f:
        reader = csv.reader(f, delimiter = delimiter)
        X = list()
        y = list()
        for line_counter, row in enumerate(reader):
            if len(row) != 2:
                raise ValueError("Row must contain 2 values, but "
                        "row #{number} {row} has {values} values".format(number = line_counter, 
                        row = row, values = len(row)))
            
            if row[1] not in ['-1', '0', '1']:
                raise ValueError("Class label must be -1, 0, or 1, but class label"
                        " at row #{number} is {label}".format(number = line_counter,
                        label = row[1]))
            
            X.append(row[0])
            y.append(int(row[1]) + 1)
        
        emb = EmbeddingExtractor(word2vec_name = word2vec)
        X_transformed = emb.fit_transform(X)
        
        clf = CNNClassifier(verbose = verbose, epochs_before_stopping = epochs_before_stopping,
            max_epochs_number = max_epochs_number, layers = layers).fit(X_transformed, y)
        
        with open(output, 'wb') as f:
            pickle.dump(clf, f)
    

if __name__ == '__main__':
    main()