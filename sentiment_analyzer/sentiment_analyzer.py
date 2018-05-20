import numpy as np

from collections import OrderedDict
from typing import Tuple


class SentimentAnalyzer(object):
    """ Анализатор тональности текстового контента.
    """
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
        """ Проанализировать тональность абзацев заданного веб-контента. Сам веб-контент представляет собой словарь,
        ключами которого являются строковые описания ранее пропарсенных URL-ов, а значениями - списки строк, т.е.
        текстовый контент каждого URL-а, разбитый на последовательность абзацев. Пример:
        {
            "www.abc.com": [
                "мама мыла раму",
                "корова молоко даёт"
            ],
            "https://hello.org": [
                "Здравствуй, мир!",
                "И тебе исполать, добрый молодец!",
                "Доброго здоровьица, девица краса.",
                "Здесь что, все здороваются?"
            ]
        }
        Данный метод анализирует тональность каждого абзаца в каждом URL-е и возвращает три числа: число позитивных
        высказываний (т.е. число абзацев с положительной тональностью), число негативных высказываний (т.е. число
        абзцацев с негативной тональность) и, наконец, общее число всех абзацев.
        :param web_content: словарь текстового контента, разбитого на абзацы, для всех обойдённых URL-ов
        :return Число позитивных высказываний, число негативных высказываний и общее число высказываний.
        """

        if not isinstance(web_content, OrderedDict):
            raise TypeError("web_content must be an OrderedDict,"
                            " but it is a {type}".format(type = type(web_content)))
        X_preprocessed = self.feature_extractor.transform(sum([web_content[key] for key in web_content], []))
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
        