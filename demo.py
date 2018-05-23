from argparse import ArgumentParser

from crawler.crawler import Crawler
from ner.spacy_ner import SpacyNamedEntityRecognizer


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', '--name', dest='name', type=str, required=True, help='Name of person or organization.')
    parser.add_argument('-w', '--who', dest='who', type=str, choices=['person', 'organization'],
                        default='person', help='Who has to be found: person or organization?')
    parser.add_argument('-u', '--url', dest='URL', type=str, required=True, help='List of URLs divided by a semicolon.')
    parser.add_argument('--sentiment', dest='sentiment_analyzer', type=str, choices=['spacy', 'own'], required=False,
                        default='own',
                        help='What library has to be base for a sentiment analyzer: the spaCy or own CNN?')
    args = parser.parse_args()
    use_spacy_for_sentiment_analysis = (args.sentiment_analyzer == 'spacy')

    name = args.name.strip()
    assert len(name) > 0, "Name of person or organization is empty!"
    is_person = (args.who == 'person')
    urls = list(filter(lambda it: len(it) > 0, [cur_url.strip() for cur_url in args.URL.split(';')]))
    assert len(name) > 0, "List of URLs is empty!"

    print('')
    print('Сорока летает по следующим сайтам:')
    for it in sorted(urls):
        print('  {0}'.format(it))
    print('')
    crawler = Crawler(divide_by_sentences=use_spacy_for_sentiment_analysis)
    full_content = crawler.load_and_tokenize(urls, depth=3)
    if len(full_content) == 0:
        print('По заданным веб-ссылкам ничего не написано :-(')
    else:
        ner = SpacyNamedEntityRecognizer()
        print('Сорока ищет, упоминается ли {0} в текстах на этих сайтах...'.format(name))
        print('')
        content_about_name = ner.filter_content(who=name, is_person=is_person, web_content=full_content)
        if len(content_about_name) == 0:
            if is_person:
                print('Сорока обыскалась везде, но никто не знает, что это за человек - {0} :('.format(name))
            else:
                print('Сорока обыскалась везде, но никто не знает, что это за организация - {0} :('.format(name))
        else:
            if use_spacy_for_sentiment_analysis:
                from sentiment_analyzer.spacy_sentiment_analyzer import SentimentAnalyzer
                se = SentimentAnalyzer()
            else:
                import os
                import pickle
                from sentiment_analyzer.sentiment_analyzer import SentimentAnalyzer
                cls_name = os.path.join('data', 'senti_cnn_classifier.pkl')
                with open(cls_name, 'rb') as cls_fp:
                    cls = pickle.load(cls_fp)
                se = SentimentAnalyzer(cls=cls)
                print('')
            print('Сорока оценивает эмоциональность этих упоминаний...')
            print('')
            negatives_number, neutral_numbers, positives_number = se.analyze(content_about_name)
            n = negatives_number + neutral_numbers + positives_number
            n_as_str = str(n)
            if n_as_str[-1] in {'2', '3', '4'}:
                if len(n_as_str) > 1:
                    if n_as_str[-2] == '1':
                        print('{0} упоминается в тексте {1} раз.'.format(name, n))
                    else:
                        print('{0} упоминается в тексте {1} раза.'.format(name, n))
                else:
                    print('{0} упоминается в тексте {1} раза.'.format(name, n))
            else:
                print('{0} упоминается в тексте {1} раз.'.format(name, n))
            print('')
            print('Коэффициент сороки:')
            print('  {0:.2%} отрицательных упоминаний;'.format(negatives_number / float(n)))
            print('  {0:.2%} положительных упоминаний;'.format(positives_number / float(n)))
            print('  {0:.2%} нейтральных упоминаний.'.format(neutral_numbers / float(n)))
            print('')
            if (positives_number > 0) and (negatives_number > 0):
                if (negatives_number - positives_number) / float(n) >= 0.1:
                    print('Сорока узнала, что {0} вызывает много отрицательных эмоций. Нужно поработать над '
                          'репутацией.'.format(name))
                elif (positives_number - negatives_number) / float(n) >= 0.1:
                    print('Сорока узнала, что {0} приносит много радости людям. Так держать!'.format(name))
                else:
                    print('Сорока узнала, что {0} вызывает неоднозначные эмоции. Вы всегда можете перевесить '
                          'общественное мнение на свою сторону.'.format(name))
            else:
                if positives_number > 0:
                    print('Сорока узнала, что {0} приносит много радости людям. Так держать!'.format(name))
                elif negatives_number > 0:
                    print('Сорока узнала, что {0} вызывает много отрицательных эмоций. Нужно поработать над '
                          'репутацией.'.format(name))
                else:
                    print('Сорока узнала, что {0} не вызывает эмоций. Заявите о себе!'.format(name))
