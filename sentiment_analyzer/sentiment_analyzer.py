from sklearn.base import BaseEstimator, ClassifierMixin

import gensim
import numpy as np

from collections import OrderedDict
from typing import Tuple


class SentimentAnalyzer(BaseEstimator, ClassifierMixin):
    def __init__(self, feature_extractor, classifier):
        if not hasattr(classifier, 'fit') or not hasattr(classifier, 'predict'):
            raise TypeError("classifier {classifier} must have fit and predict" 
                "methods".format(classifier = classifier))
        
        if not hasattr(feature_extractor, 'transform'):
            raise TypeError("feature_extractor {feature_extractor} must" 
                    "have a transform method".format(feature_extractor = feature_extractor))
        
        self.classifier = classifier
        self.feature_extractor = feature_extractor
        
    
    def analyze(self, web_content: OrderedDict) -> Tuple[int, int, int]:
        if not isinstance(web_content, OrderedDict):
            raise TypeError("web_content must be an OrderedDict,"
                            " but it is a {type}".format(type = type(web_content)))
        X_preprocessed = self.feature_extractor.fit_transform(sum([web_content[key] for key in web_content], []))
        output = self.classifier.predict(X_preprocessed)
        positives = int(sum(output == 2))
        neutrals = int(sum(output == 1))
        negatives = int(sum(output == 0))
        
        return (negatives, neutrals, positives)
    
    def __getstate__(self):
        return {'classifier': self.classifier, 'feature_extractor': self.feature_extractor}
    
    def __setstate__(self, state):
        self.classifier = state['classifier']
        self.feature_extractor = state['feature_extractor']
        return self
        