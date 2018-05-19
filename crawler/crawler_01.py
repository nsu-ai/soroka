from collections import OrderedDict
from typing import List

# from googlesearch import search
from requests import get
from bs4 import BeautifulSoup

from crawler.crawler import Crawler


class Crawler01(Crawler):
    def load_and_tokenize(self, urls: List[str], depth: int = 3) -> OrderedDict:
        out = OrderedDict()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'
        }

        for url in urls:
            response = get(url, headers=headers)
            html_text = response.text
            html_soup = BeautifulSoup(html_text, 'html.parser')

            paragraphs = []
            for p in html_soup.find_all('p'):
                txt = p.getText().strip()
                if txt:
                    paragraphs.append(txt)
            out[url] = paragraphs
        return out


if __name__ == '__main__':
    my_crawler = Crawler01()
    res = my_crawler.load_and_tokenize(['https://academ.info'])
    print(res)
