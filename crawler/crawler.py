import copy
from bs4 import BeautifulSoup
from bs4.element import Comment
from collections import OrderedDict
from requests import get
from typing import List
from urllib.parse import urlsplit
from multiprocessing import Pool

from nltk import sent_tokenize


class Crawler:
    """ Класс анализирует текстовый контент веб-страниц, начиная с заданного списка URL-ов, и бьёт его на абзацы.

    """
    def __init__(self, divide_by_sentences: bool=False):
        self.divide_by_sentences = divide_by_sentences

    def load_and_tokenize(self, urls: List[str], depth: int=3) -> OrderedDict:
        """ Загрузить все веб-страницы по заданным URL-ам и распарсить текстовый контент в них.

        Производится парсинг как веб-страниц, заданных исходным списком URL-ов, так рекурсивный обход всех других
        веб-страниц, на которые можно перейти из исходных. Текстовый контент каждой страницы структурируется путём
        разбивки на отдельные абзацы. В результате работы функции возвращается словарь, ключами которого являются
        строковые описания URL-ов, обойдённых в процессе парсинга, а значениями - списки строк, т.е. текстовый контент
        каждого URL-а, разбитый на последовательность абзацев. Пример:

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

        :param urls: список стартовых URL-ов
        :param depth: глубина рекурсивного обхода, начиная со стартовых URL-ов

        :return словарь текстового контента, разбитого на абзацы, для всех обойдённых URL-ов

        """
        out = OrderedDict()

        for url in urls:
            # detecting base url
            base_url = "{0.scheme}://{0.netloc}/".format(urlsplit(url))

            # depth == 1
            html_soup = self.get_bs4_from_url(url)
            paragraphs = self.get_paragraphs(html_soup)
            out[url] = paragraphs
            urls_depth_2 = self.get_links_on_page(html_soup, base_url=base_url, old_links=out.keys())

            if depth > 1:
                # This should speed up x4 times for depth > 1
                p = Pool(4)
                html_texts = p.map(Crawler.get_html_str_from_url, urls_depth_2)
                for kk, html_text in enumerate(html_texts):
                    html_soup = BeautifulSoup(html_text, 'html.parser')
                    paragraphs = self.get_paragraphs(html_soup)
                    out[urls_depth_2[kk]] = paragraphs
                    urls_depth_3 = self.get_links_on_page(html_soup, base_url=base_url, old_links=out.keys())
            if depth > 2:
                html_texts = p.map(Crawler.get_html_str_from_url, urls_depth_3)
                for kk, html_text in enumerate(html_texts):
                    html_soup = BeautifulSoup(html_text, 'html.parser')
                    paragraphs = self.get_paragraphs(html_soup)
                    out[urls_depth_3[kk]] = paragraphs
                    urls_depth_4 = self.get_links_on_page(html_soup, base_url=base_url, old_links=out.keys())
            if depth > 3:
                html_texts = p.map(Crawler.get_html_str_from_url, urls_depth_3)
                for kk, html_text in enumerate(html_texts):
                    html_soup = BeautifulSoup(html_text, 'html.parser')
                    paragraphs = self.get_paragraphs(html_soup)
                    out[urls_depth_4[kk]] = paragraphs
                    urls_depth_5 = self.get_links_on_page(html_soup, base_url=base_url, old_links=out.keys())
            if self.divide_by_sentences:
                sentences = []
                for cur_paragraph in out[url]:
                    sentences += sent_tokenize(cur_paragraph)
                out[url] = copy.copy(sentences)

        return out

    @staticmethod
    def get_paragraphs(html_soup):
        # https://stackoverflow.com/questions/1936466/beautifulsoup-grab-visible-webpage-text
        def tag_visible(element):
            if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
                return False
            if isinstance(element, Comment):
                return False
            return True

        texts = html_soup.findAll(text=True)
        visible_texts = filter(tag_visible, texts)
        # return u" ".join(t.strip() for t in visible_texts)
        visible_texts = [v.strip() for v in visible_texts if len(v.strip()) > 10]

        return list(visible_texts)

    @staticmethod
    def get_bs4_from_url(url):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'
        }
        try:
            response = get(url, headers=headers)
            html_text = response.text
        except:
            html_text = ''
        html_soup = BeautifulSoup(html_text, 'html.parser')
        return html_soup

    @staticmethod
    def get_html_str_from_url(url):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'
        }
        try:
            response = get(url, headers=headers)
            html_text = response.text
        except:
            html_text = ''
        return html_text

    @staticmethod
    def get_links_on_page(html_soup, base_url, old_links=[]):
        links = [a.get('href') for a in html_soup.find_all('a', href=True)]

        def add_http(url):
            return url if url.startswith(base_url) else base_url + url

        links = [add_http(l) for l in links]
        links = [l for l in links if l not in old_links]
        return links

    def get_all_pages(self, url):
        pass


if __name__ == '__main__':
    my_crawler = Crawler()
    res = my_crawler.load_and_tokenize(['https://academ.info'])
    # res = my_crawler.load_and_tokenize(['http://zabaykin.ru'], depth=3)
    print(res)
