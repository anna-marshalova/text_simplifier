import os

ROOT = '/content/'
FASTTEXT_URL = 'http://files.deeppavlov.ai/embeddings/ft_native_300_ru_wiki_lenta_nltk_word_tokenize/ft_native_300_ru_wiki_lenta_nltk_word_tokenize.bin'
FASTTEXT_PATH = os.path.join('root', 'input', 'ft_native_300_ru_wiki_lenta_nltk_word_tokenize.bin')
SYNONYMS_PATH = os.path.join(os.getcwd(), 'baseline', 'synonyms.json')
COMPLEX_WORDS_PATH = os.path.join(os.getcwd(), 'baseline', 'complex_words.json')
WORDLIST_PATH = os.path.join(ROOT,'RuAdapt_Word_Lists', 'dictionary_synonyms.txt')