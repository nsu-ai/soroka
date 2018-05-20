import unittest
import copy
from ner.spacy_ner import SpacyNamedEntityRecognizer


class TestSpacyNer(unittest.TestCase):


    def test_delete_by_person(self):

        test_content = {'url1': ['Наташа помогла Андрею с домашней работой. Наташа хороший человек.', 'Она большая молодец'], 'url2': ['Сегодня день рождения Алексеевой, поздравим ее']}
        nr = SpacyNamedEntityRecognizer()
        content_to_transform = copy.deepcopy(test_content)
        res = nr.filter_content('Наташа Надеина', True, content_to_transform)
        print('RES', res)
        del test_content['url1'][1]
        del test_content['url2']
        print('Content', test_content)
        self.assertEqual(res, test_content)

    def test_delete_by_org(self):
        test_content = {'url1': ['В последнее время не очень люблю ходить в Пятерочку, они отвратительно относятся к своим сотрудникам. В магазине Пятерочка продаются самые разные вещи'], 'url2': ['Вася продал акции'], 'url3': ['Магазин Пятерочка набирает сотрудников на работу']}
        nr = SpacyNamedEntityRecognizer()
        content_to_transform = copy.deepcopy(test_content)
        res = nr.filter_content('Пятерочка', False, content_to_transform)
        print('Res', res)
        del test_content['url2']
        print('Content', test_content)
        self.assertEqual(res, test_content)


if __name__ == '__main__':
    unittest.main()