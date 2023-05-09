import json
import numpy as np
from tqdm.notebook import tqdm
from scipy.special import softmax
from collections import defaultdict

import nltk

nltk.download('punkt')
nltk.download('stopwords')


class BaselineSimplifier:
    def __init__(self, fasttext_model = None):
        self.synonym_dict = {}
        self.complex_words = []
        self.fasttext_model = fasttext_model
        self.tokenize = nltk.word_tokenize
        self.stopwords = nltk.corpus.stopwords.words('russian')

    def areequal(self, a, b):
        a = a.lower()
        b = b.lower()
        return (a in b) or (b in a)

    def process_pair(self, source, target, alignment_distance):
        """
        Извлекаем из пары текстов список сложных слов и список синонимов
        :param source: Исходный, сложный текст
        :param target: Таргет, упрощенный текст
        :param alignment_distance:
        :return: synonyms: список синонимов
                 complex_tokens: список сложных слов
        """
        complex_words = set()
        synonyms = set()
        source_tokenized = self.tokenize(source)
        target_tokenized = self.tokenize(source)
        for i, source_token in enumerate(source_tokenized):
            for j, target_token in enumerate(target_tokenized):
                # если слова нет в таргете, считаем его усложняющим текст
                if source_token not in target:
                    complex_words.add(source_token)
                    # ищем синонимы между простыми и сложными текстами.
                    # проверяем, что они не являются стоп-словами, в предложении занимают близкие позиции (отличающиеся не более, чем на alignement_distance) и при этом не совпадают
                    if target_token not in self.stopwords and abs(i - j) <= alignment_distance and not self.areequal(
                            source_token, target_token):
                        distance = self.fasttext_model.distance(source_token, target_token)
                        synonyms.add((source_token, target_token, distance))
        # из найденных синонимичных пар выбираем топ-10 по близости векторов
        synonyms = set(sorted(synonyms, key=lambda pair: pair[2])[:10])
        return synonyms, complex_words

    def fit(self, source_texts, target_texts, alignment_distance=1):
        """
        Составление словаря синонимов и списка сложных слов на основе параллельного корпуса
        :param source_texts: Исходные, сложные тексты
        :param target_texts: Таргеты, упрощеннные тексты
        :param alignment_distance: Разница между позициями слов в текстах, при которой слова считаются синонимами.
        :return:
        """
        if self.synonym_dict and self.complex_words:
            print('Complex words list and dict of synonyms already provided! They are going to be replaced.')
        if not self.fasttext_model:
            print('Please provide a fasttext model.')
        synonyms = set()
        complex_tokens = set()
        for source, target in tqdm(zip(source_texts, target_texts)):
            synonyms_from_text, complex_tokens_from_text = self.process_pair(source, target, alignment_distance)
            synonyms = synonyms | synonyms_from_text
            complex_tokens = complex_tokens | complex_tokens_from_text
        self.complex_words = complex_tokens
        synonym_dict = defaultdict(dict)
        for source, target, distance in synonyms:
            # для синонимов храним их близость к исходному слову
            synonym_dict[source].update({target: (1 - distance)})
            self.synonym_dict = synonym_dict

    def simplify(self, text):
        """
        Упрощение текста. Сложные слова в тескте меняются на более простые синонимы
        :param text:
        :return:
        """
        if (not self.synonym_dict or not self.complex_words) and not (self.fasttext_model):
            print('Please provide a model or complex words list and dict of synonyms')
        simple_text = []
        text_tokenized = self.tokenize(text)
        for token in text_tokenized:
            # если слово сложное, ищем для него более простую синонимичную замену. если ее нет - удаляем слово
            if token in self.complex_words:
                if token in self.synonym_dict:
                    # синоним выбираем с вероятностью, пропорциональной его семантической близости к исходному слову
                    candidates = list(self.synonym_dict[token].keys())
                    probs = softmax(list(self.synonym_dict[token].values()))
                    simple_sub = np.random.choice(candidates, p=probs)
                    simple_text.append(simple_sub)
            else:
                simple_text.append(token)
        return ' '.join(simple_text)

    def save_data(self, synonyms_path='synonyms.json', complex_list_path='complex_words.json'):
        with open(synonyms_path, 'w', encoding='utf-8') as file:
            json.dump(self.synonym_dict, file, ensure_ascii=False)
        with open(complex_list_path, 'w', encoding='utf-8') as file:
            json.dump(list(self.complex_words), file, ensure_ascii=False)

    def load_data(self, synonyms_path='synonyms.json', complex_list_path='complex_words.json'):
        with open(synonyms_path, 'r', encoding='utf-8') as file:
            self.synonym_dict = json.load(file)
        with open(complex_list_path, 'r', encoding='utf-8') as file:
            self.complex_words = json.load(file)