from crawler.crawler_01 import Crawler01
import unittest


class TestExtractor(unittest.TestCase):
    def setUp(self):
        pass

    def test_init_positive01(self):
        my_crawler = Crawler01()
        site_to_check = 'http://zabaykin.ru'
        res = my_crawler.load_and_tokenize([site_to_check], depth=1)
        # print(res)
        # check number of paragraphs
        self.assertEqual(len(res[site_to_check]), 244)

    def test_init_positive02(self):
        my_crawler = Crawler01()
        site_to_check = 'http://zabaykin.ru/?p=505'
        res = my_crawler.load_and_tokenize([site_to_check], depth=2)
        # print(res)
        # check number of parsed urls
        self.assertEqual(len(list(set(res.keys()))), 36)


if __name__ == '__main__':
    unittest.main(verbosity=2)
