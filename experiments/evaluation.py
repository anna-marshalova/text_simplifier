import re

import rusyllab
import nltk

nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from easse.sari import corpus_sari
from easse.bleu import corpus_bleu
from easse.bertscore import corpus_bertscore
from easse.quality_estimation import corpus_quality_estimation


def corpus_fkgl(sys_sents, sent_tokenize=sent_tokenize):
    """
    Считаем FKGL (https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests)
    Используется формула Обороневой (адаптированная для русского языка)
    :param sys_sents: Упрощеннные предложения, сгенерированные моделью
    :param sent_tokenize: Токенизатор для разделения текста на предложения
    :return:
    """
    word_pattern = re.compile('[А-яA-z]+')
    w = 0
    c = 0
    s = 0
    for sent in sys_sents:
        # считаем количество предложений (модель может дробить одно предложение на несколько)
        s += len(sent_tokenize(sent))
        # считаем количество слов (без учета знаков препинания)
        words = word_pattern.findall(sent)
        w += len(words)
        # считаем количество слогов
        syllables = rusyllab.split_words(''.join(words).split())
        c += sum(1 for syl in syllables if syl != ' ')
    return 206.836 - (65.14 * c / w) - (1.52 * w / s)


METRIC_FUNCS = {
    'bleu': {'func': corpus_bleu, 'requires_refs': True, 'requires_orig': False},
    'sari': {'func': corpus_sari, 'requires_refs': True, 'requires_orig': True},
    'bertscore': {'func': corpus_bertscore, 'requires_refs': True, 'requires_orig': False},
    'fkgl': {'func': corpus_fkgl, 'requires_refs': False, 'requires_orig': False},
}


def compute_corpus_metrics(orig, refs, simplification_func, compute_quality_estimation=True,
                           metrics=('bleu', 'sari', 'fkgl'), **kwargs):
    """
    Подсчет корпусных метрик
    :param orig: Сложные предложения
    :param refs: Упрощенные предложения из датасета
    :param simplification_func: Функция, упрощающая предложения
    :param compute_quality_estimation: Вычислять ли quality_estimation из библиотеки easse
    :param metrics: Список метрик, которые будут вычислены {'bleu', 'sari', 'fkgl', 'bertscore'}
    :return: computed_metrics: Словарь с метриками
             quality: Метрики качества из библиотеки easse
    """
    computed_metrics = {}
    quality = {}
    assert (len(refs) == len(orig))
    preds = simplification_func(orig, **kwargs)
    assert (len(preds) == len(orig))
    for metric in metrics:
        compute_metric = METRIC_FUNCS[metric]['func']
        kwargs = {'sys_sents': preds}
        if METRIC_FUNCS[metric]['requires_refs']:
            kwargs.update({'refs_sents': refs})
        if METRIC_FUNCS[metric]['requires_orig']:
            kwargs.update({'orig_sents': orig})
        if metric == 'bertscore':
            computed_metric = compute_metric(**kwargs)[2]
        else:
            computed_metric = compute_metric(**kwargs)
        computed_metrics.update({metric: round(computed_metric, 3)})
    if compute_quality_estimation:
        quality = corpus_quality_estimation(sys_sentences=preds, orig_sentences=orig)
    return computed_metrics, quality
