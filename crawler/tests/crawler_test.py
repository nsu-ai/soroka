from crawler.crawler_01 import Crawler01
import unittest


class TestExtractor(unittest.TestCase):
    def setUp(self):
        pass

    def test_init_positive01(self):
        my_crawler = Crawler01()
        site_to_check = 'http://zabaykin.ru'
        res = my_crawler.load_and_tokenize([site_to_check])
        # print(res)
        self.assertEqual(len(res[site_to_check]), 63)


if __name__ == '__main__':
    unittest.main(verbosity=2)
