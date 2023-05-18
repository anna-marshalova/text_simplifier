import re

import rusyllab
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from easse.sari import corpus_sari
from easse.bleu import corpus_bleu
from easse.bertscore import corpus_bertscore
from easse.quality_estimation import corpus_quality_estimation

def corpus_fkgl(sys_sents, sent_tokenize = sent_tokenize):
    word_pattern = re.compile('[А-яA-z]+')
    w = 0
    c = 0
    s = 0
    for sent in sys_sents:
        s += len(sent_tokenize(sent))
        words = word_pattern.findall(sent)
        w += len(words)
        syllables = rusyllab.split_words(''.join(words).split())
        c += sum(1 for syl in syllables if syl!=' ')
    return 206.836 - (65.14 * c/w) - (1.52 * w/s)

METRIC_FUNCS = {
    'bleu': {'func': corpus_bleu, 'requires_refs': True, 'requires_orig': False},
    'sari': {'func': corpus_sari, 'requires_refs': True, 'requires_orig': True},
    'bertscore': {'func': corpus_bertscore, 'requires_refs': True, 'requires_orig': False},
    'fkgl': {'func': corpus_fkgl, 'requires_refs': False, 'requires_orig': False},
}

def compute_corpus_metrics(orig, refs, simplification_func, compute_quality_estimation=True,
                           metrics=('bleu', 'sari', 'fkgl'), **kwargs):
    computed_metrics = {}
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
        computed_metrics.update({metric: compute_metric(**kwargs)})
    if compute_quality_estimation:
        quality = corpus_quality_estimation(sys_sentences=preds, orig_sentences=orig)
    return computed_metrics, quality