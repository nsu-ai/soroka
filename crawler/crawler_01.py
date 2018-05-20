from collections import OrderedDict
from typing import List
from urllib.parse import urlsplit

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
            # detecting base url
            base_url = "{0.scheme}://{0.netloc}/".format(urlsplit(url))

            # depth == 1
            html_soup = self.get_bs4_from_url(url)
            paragraphs = self.get_paragraphs(html_soup)
            out[url] = paragraphs
            urls_depth_2 = self.get_links_on_page(html_soup, base_url=base_url, old_links=out.keys())

            if depth > 1:
                for url_2 in urls_depth_2:
                    html_soup = self.get_bs4_from_url(url_2)
                    paragraphs = self.get_paragraphs(html_soup)
                    out[url_2] = paragraphs
                    urls_depth_3 = self.get_links_on_page(html_soup, base_url=base_url, old_links=out.keys())
            if depth > 2:
                for url_3 in urls_depth_3:
                    html_soup = self.get_bs4_from_url(url_3)
                    paragraphs = self.get_paragraphs(html_soup)
                    out[url_3] = paragraphs
                    urls_depth_4 = self.get_links_on_page(html_soup, base_url=base_url, old_links=out.keys())
            if depth > 3:
                for url_4 in urls_depth_4:
                    html_soup = self.get_bs4_from_url(url_4)
                    paragraphs = self.get_paragraphs(html_soup)
                    out[url_4] = paragraphs
                    urls_depth_5 = self.get_links_on_page(html_soup, base_url=base_url, old_links=out.keys())
        # for k, el in out.items():
        #     print(k, '!!!')
        #     for e in el:
        #         print(e)
        return out

    @staticmethod
    def get_paragraphs(html_soup):
        paragraphs = []
        for p in html_soup.find_all('p'):
            txt = p.getText().strip()
            if txt:
                paragraphs.append(txt)
        return paragraphs

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
    my_crawler = Crawler01()
    res = my_crawler.load_and_tokenize(['https://academ.info'])
    print(res)
