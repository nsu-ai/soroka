import operator
import numpy as np

from collections import OrderedDict
from typing import Tuple

import spacy


class SentimentAnalyzer(object):
    """ Анализатор тональности текстового контента.
    """
    def __init__(self, feature_extractor, classifier):
        self.nlp = spacy.load("../data")

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
                            " but it is a {type}".format(type=type(web_content)))
        results = {
            'POSITIVE': 0,
            'NEUTRAL': 0,
            'NEGATIVE': 0
        }
        for k, v in web_content.items():
            results[max(self.nlp(str(v)).cats.items(), key=operator.itemgetter(1))[0]] += 1

        return results['NEGATIVE'], results['NEUTRAL'], results['POSITIVE']

    def __getstate__(self):
        return {'classifier': self.classifier, 'feature_extractor': self.feature_extractor}

    def __setstate__(self, state):
        self.classifier = state['classifier']
        self.feature_extractor = state['feature_extractor']
        return self