from argparse import ArgumentParser

from crawler.crawler import Crawler
from ner.spacy_ner import SpacyNamedEntityRecognizer


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', '--name', dest='name', type=str, required=True, help='Name of person or organization.')
    parser.add_argument('-w', '--who', dest='who', type=str, choices=['person', 'organization'],
                        default='person', help='Who has to be found: person or organization?')
    parser.add_argument('-u', '--url', dest='URL', type=str, required=True, help='List of URLs divided by a semicolon.')
    args = parser.parse_args()

    name = args.name.strip()
    assert len(name) > 0, "Name of person or organization is empty!"
    is_person = (args.who == 'person')
    urls = list(filter(lambda it: len(it) > 0, [cur_url.strip() for cur_url in args.URL.split(';')]))
    assert len(name) > 0, "List of URLs is empty!"

    crawler = Crawler()
    full_content = crawler.load_and_tokenize(urls)
    if len(full_content) == 0:
        print('По заданным веб-ссылкам ничего не написано :-(')
    else:
        ner = SpacyNamedEntityRecognizer()
        content_about_name = ner.filter_content(who=name, is_person=is_person, web_content=full_content)
        if len(content_about_name) == 0:
            print('Мы перелопатили весь текст по ссылкам, но {0} никому не известен :('.format(name))
        else:
            pass
