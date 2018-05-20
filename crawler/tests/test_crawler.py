from crawler.crawler import Crawler
import unittest
from collections import OrderedDict
from typing import List


class TestExtractor(unittest.TestCase):
    def setUp(self):
        pass

    def test_init_positive01(self):
        my_crawler = Crawler()
        site_to_check = 'http://zabaykin.ru'
        res = my_crawler.load_and_tokenize([site_to_check], depth=1)
        # print(res)
        # check number of paragraphs
        self.assertEqual(len(res[site_to_check]), 244)

    def test_init_positive02(self):
        my_crawler = Crawler()
        site_to_check = 'http://zabaykin.ru/?p=505'
        res = my_crawler.load_and_tokenize([site_to_check], depth=2)
        # print(res)
        # check number of parsed urls
        self.assertEqual(len(list(set(res.keys()))), 36)

    def test_structure_positive_01(self):
        my_crawler = Crawler()
        site_to_check = 'http://zabaykin.ru/?p=505'
        res = my_crawler.load_and_tokenize([site_to_check], depth=2)
        self.assertIsInstance(res, OrderedDict)
        for k, el in res.items():
            self.assertIsInstance(k, str)
            self.assertTrue(k.startswith('http'))
            self.assertIsInstance(el, List[str])


if __name__ == '__main__':
    unittest.main(verbosity=2)
