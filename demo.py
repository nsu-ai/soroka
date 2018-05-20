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
    use_spacy_for_sentiment_analysys = False

    name = args.name.strip()
    assert len(name) > 0, "Name of person or organization is empty!"
    is_person = (args.who == 'person')
    urls = list(filter(lambda it: len(it) > 0, [cur_url.strip() for cur_url in args.URL.split(';')]))
    assert len(name) > 0, "List of URLs is empty!"

    print('')
    print('Мы парсим следующие сайты:')
    for it in sorted(urls):
        print('  {0}'.format(it))
    print('')
    crawler = Crawler()
    full_content = crawler.load_and_tokenize(urls, depth=2)
    if len(full_content) == 0:
        print('По заданным веб-ссылкам ничего не написано :-(')
    else:
        ner = SpacyNamedEntityRecognizer()
        print('Мы находим, упоминается ли {0} в текстах на этих сайтах...'.format(name))
        print('')
        content_about_name = ner.filter_content(who=name, is_person=is_person, web_content=full_content)
        if len(content_about_name) == 0:
            if is_person:
                print('Мы перелопатили весь текст по ссылкам, но никто не знает, что это за '
                      'человек - {0} :('.format(name))
            else:
                print('Мы перелопатили весь текст по ссылкам, но никто не знает, что это за '
                      'организация - {0} :('.format(name))
        else:
            if use_spacy_for_sentiment_analysys:
                from sentiment_analyzer.spacy_sentiment_analyzer import SentimentAnalyzer
                se = SentimentAnalyzer()
            else:
                import pickle
                from sentiment_analyzer.sentiment_analyzer import SentimentAnalyzer
                fe_name = ''
                cls_name = ''
                with open(fe_name, 'rb') as fe_fp:
                    fe = pickle.load(fe_fp)
                with open(cls_name, 'rb') as cls_fp:
                    cls = pickle.load(cls_fp)
                se = SentimentAnalyzer(feature_extractor=fe, classifier=cls)
            print('Мы оцениваем эмоциональность этих упоминаний...')
            print('')
            negatives_number, neutral_numbers, positives_number = se.analyze(content_about_name)
            n = negatives_number + neutral_numbers + positives_number
            print('{0} упоминается в тексте {1} раз:'.format(name, n))
            print('')
            print('{0.2%} отрицательных упоминаний;'.format(negatives_number / float(n)))
            print('{0.2%} положительных упоминаний;'.format(positives_number / float(n)))
            print('{0.2%} нейтральных упоминаний.'.format(neutral_numbers / float(n)))
            print('')
            if (negatives_number - positives_number) >= 10:
                print('{0} вызывает много отрицательных эмоций. Нужно поработать над репутацией.'.format(name))
            elif (positives_number - negatives_number) >= 10:
                print('{0} приносит много радости людям. Так держать!'.format(name))
            elif (positives_number > 0) or (negatives_number > 0):
                print('{0} вызывает неоднозначные эмоции. Вы всегда можете перевесить общественное мнение '
                      'на свою сторону.'.format(name))
            else:
                print('{0} не вызывает эмоций. Заявите о себе!'.format(name))
