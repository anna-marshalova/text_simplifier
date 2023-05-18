import re
import json
import numpy as np
from tqdm.notebook import tqdm
from scipy.special import softmax
from collections import defaultdict
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import spacy
from pymorphy2 import MorphAnalyzer
from Levenshtein import distance as lev

from baseline.paths import SYNONYMS_PATH, COMPLEX_WORDS_PATH, WORDLIST_PATH


class BaselineSimplifier:
    def __init__(self, fasttext_model = None):
        self.synonym_dict = {}
        self.complex_words = []
        self.fasttext_model = fasttext_model
        self.tokenize = nltk.word_tokenize
        self.stopwords = nltk.corpus.stopwords.words('russian')
        self.morph = MorphAnalyzer()
        self.nlp = spacy.load("ru_core_news_lg")

    def areequal(self, a, b):
        a = a.lower()
        b = b.lower()
        return (a in b) or (b in a)

    def process_pair(self, source, target, alignment_distance=3, min_lev = 2, num_synonyms = 5, pos=True):
        """
        Извлекаем из пары текстов список сложных слов и список синонимов
        :param source: Исходный, сложный текст
        :param target: Таргет, упрощенный текст
        :param alignment_distance:
        :return: synonyms: список синонимов
                 complex_words: список сложных слов
        """
        complex_words = set()
        synonyms = set()
        source_tokenized = self.tokenize(source.lower())
        target_tokenized = self.tokenize(target.lower())
        for i, source_token in enumerate(source_tokenized):
            for j, target_token in enumerate(target_tokenized):
                # если слова нет в таргете, считаем его усложняющим текст
                if source_token not in target and source_token not in self.stopwords:
                    source_parse = self.morph.parse(source_token)[0]
                    source_lemma = source_parse.normal_form
                    complex_words.add(source_lemma)
                    # ищем синонимы между простыми и сложными текстами.
                    # проверяем, что они не являются стоп-словами, в предложении занимают близкие позиции (отличающиеся не более, чем на alignement_distance) и при этом не совпадают
                    if target_token not in self.stopwords and abs(i - j) <= alignment_distance and lev(source_token, target_token) >= 2:
                        distance = self.fasttext_model.distance(source_token, target_token)
                        target_parse = self.morph.parse(target_token)[0]
                        target_lemma = target_parse.normal_form
                        if (pos and source_parse.tag.POS == target_parse.tag.POS) or not pos:
                            synonyms.add((source_lemma, target_lemma, distance))
        # из найденных синонимичных пар выбираем топ-N по близости векторов
        synonyms = set(sorted(synonyms, key=lambda pair: pair[2])[:num_synonyms])
        return synonyms, complex_words

    def fit(self, source_texts, target_texts, **kwargs):
        """
        Составление словаря синонимов и списка сложных слов на основе параллельного корпуса
        :param source_texts: Исходные, сложные тексты
        :param target_texts: Таргеты, упрощеннные тексты
        :param alignment_distance: Разница между позициями слов в текстах, при которой слова считаются синонимами.
        :return:
        """
        if self.synonym_dict and self.complex_words:
            print('Complex words list and dict of synonyms already provided! They are going to be replaced.')
        assert self.fasttext_model, 'Please provide a fasttext model.'
        synonyms = set()
        complex_words = set()
        for source, target in tqdm(zip(source_texts, target_texts)):
            synonyms_from_text, complex_words_from_text = self.process_pair(source, target, **kwargs)
            synonyms = synonyms | synonyms_from_text
            complex_words = complex_words | complex_words_from_text
        self.complex_words = complex_words
        synonym_dict = defaultdict(dict)
        for source, target, distance in synonyms:
            # для синонимов храним их близость к исходному слову
            synonym_dict[source].update({target: (1 - distance)})
            self.synonym_dict = synonym_dict
        self.gold_standard_synonyms = self.load_gold_standard_synonyms(**kwargs)
    
    def filter_synonyms_by_pos(self, token, synonyms):
        candidates = []
        distances = []
        token_parse = self.morph.parse(token)[0]
        for candidate, distance in synonyms.items():
            cand_parse = self.morph.parse(candidate)[0]
            if token_parse.tag.POS == cand_parse.tag.POS:
                candidates.append(candidate)
                distances.append(distance)
        return  candidates, distances
    
    def filter_synonyms_by_dist(self, synonyms, min_dist = 0.5):
        candidates = []
        distances = []
        for cand, dist in synonyms.items():
            if dist >= min_dist:
                candidates.append(cand)
                distances.append(dist)
        return  candidates, distances

    def simplify(self, text, grammar=True, gc_synonyms = True, fasttext_synonyms = True, leave_complex = True, pos=True, inflect = True, **kwargs):
        """
        Упрощение текста. Сложные слова в тескте меняются на более простые синонимы
        :param text:
        :return:
        """
        if (not self.synonym_dict or not self.complex_words) and not (self.fasttext_model):
            print('Please provide a model or complex words list and dict of synonyms')
        simple_text = []
        if grammar:
            text_tokenized = self.grammatical_simplification(text, **kwargs)
        else:
            text_tokenized = self.tokenize(text)
        if fasttext_synonyms or gc_synonyms:
            for token in text_tokenized:
                token = token.lower()
                lemma = self.morph.parse(token)[0].normal_form
                # если слово сложное, ищем для него более простую синонимичную замену. если ее нет - удаляем слово
                if gc_synonyms and lemma in self.gold_standard_synonyms:
                    simple_sub = self.gold_standard_synonyms[lemma]
                    if inflect:
                        simple_sub = self.inflect_like(simple_sub, token)
                    simple_text.append(simple_sub)
                elif fasttext_synonyms and lemma in self.complex_words and lemma not in self.stopwords:
                    if lemma in self.synonym_dict:
                        # синоним выбираем с вероятностью, пропорциональной его семантической близости к исходному слову
                        #candidates = list(self.synonym_dict[lemma].keys())
                        #distances = list(self.synonym_dict[lemma].values())
                        candidates, distances = self.filter_synonyms_by_dist(self.synonym_dict[lemma], **kwargs)
                        if candidates:
                            probs = softmax(distances)
                            simple_sub = np.random.choice(candidates, p=probs)
                            if inflect:
                                simple_sub = self.inflect_like(simple_sub, token)
                            simple_text.append(simple_sub)
                        else:
                            simple_text.append(token)
                    elif leave_complex:
                        simple_text.append(token)
                else:
                    simple_text.append(token)
        else:
            simple_text = text_tokenized
        return ' '.join(simple_text)
    
    def grammatical_simplification(self, text, levels=2):
        prev_level_text = []
        for level in range(levels):
            cur_level_text = []
            doc = self.nlp(text)
            root = doc[::].root
            for token in doc:
                if token == root or token.head == root or token.head.text in prev_level_text:
                    cur_level_text.append(token.text)
            prev_level_text = cur_level_text
        return cur_level_text
    
    def inflect_like(self, word, ref_word):
        word_parse = self.morph.parse(word)[0]
        ref_parse = self.morph.parse(ref_word)[0]
        result = None
        if ref_parse.tag.POS and ref_parse.tag.POS == word_parse.tag.POS:
            if ref_parse.tag.POS == 'NOUN':
                grammemes = {ref_parse.tag.case, ref_parse.tag.number}
            elif ref_parse.tag.POS.startswith('ADJ') and ref_parse.tag.number == 'plur':
                grammemes = {'plur', ref_parse.tag.case}
            elif ref_parse.tag.POS.startswith('ADJ') and ref_parse.tag.number == 'sing':
                grammemes = {'sing', ref_parse.tag.case, ref_parse.tag.gender}
            else:
                grammemes = set(re.split(',| ', str(ref_parse.tag)))
            result = word_parse.inflect(grammemes)
        if result:
            return result.word
        return word

    def save_data(self, synonyms_path=SYNONYMS_PATH, complex_list_path=COMPLEX_WORDS_PATH):
        with open(synonyms_path, 'w', encoding='utf-8') as file:
            json.dump(self.synonym_dict, file, ensure_ascii=False)
        with open(complex_list_path, 'w', encoding='utf-8') as file:
            json.dump(list(self.complex_words), file, ensure_ascii=False)

    def load_data(self, synonyms_path=SYNONYMS_PATH, complex_list_path=COMPLEX_WORDS_PATH, **kwargs):
        with open(synonyms_path, 'r', encoding='utf-8') as file:
            self.synonym_dict = json.load(file)
        with open(complex_list_path, 'r', encoding='utf-8') as file:
            self.complex_words = json.load(file)
        self.gold_standard_synonyms = self.load_gold_standard_synonyms(**kwargs)
    
    def load_gold_standard_synonyms(self, gs_synonyms_path = WORDLIST_PATH):
        gold_standard_synonyms = {}
        with open(gs_synonyms_path, 'r', encoding='utf-8') as file:
            pairs = file.readlines()
            for pair in pairs:
                complex_word, simple_word = pair.split(' ')
                gold_standard_synonyms.update({complex_word: simple_word})
        return gold_standard_synonyms