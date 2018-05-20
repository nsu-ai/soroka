from collections import OrderedDict
from typing import List

# from googlesearch import search
from requests import get
from bs4 import BeautifulSoup

from crawler.crawler import Crawler


class Crawler01(Crawler):
    def __init__(self):
        pass

    def load_and_tokenize(self, urls: List[str], depth: int = 3) -> OrderedDict:
        out = OrderedDict()

        for url in urls:
            html_soup = self.get_bs4_from_url(url)

            paragraphs = []
            for p in html_soup.find_all('p'):
                txt = p.getText().strip()
                if txt:
                    paragraphs.append(txt)
            out[url] = paragraphs
        return out

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

    def get_links_on_page(self, html_soup, base_url):
        links = [a.get('href') for a in html_soup.find_all('a', href=True)]

        def add_http(url):
            return url if url.startswith('http') else base_url + '/' + url

        links = [add_http(l) for l in links]
        return links

    def get_all_pages(self, url):
        pass


if __name__ == '__main__':
    my_crawler = Crawler01()
    res = my_crawler.load_and_tokenize(['https://academ.info'])
    print(res)
