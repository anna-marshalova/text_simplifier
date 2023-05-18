import pandas as pd
from tqdm.notebook import tqdm
from Levenshtein import distance

SOURCE_COLUMN_NAME = 'INPUT:source'
TARGET_COLUMN_NAME = 'OUTPUT:output'

def filter_ru_adapt(path, min_sim = 0.5, min_lev = 20, max_elong_rate = 0.5):
    df = pd.read_csv(path)
    df = df[(df.cos_sim >= min_sim) & (df.cos_sim < 1)]
    df['lev'] = df.apply(lambda x: distance(x.source, x.target), axis=1)
    df = df[df.lev >= min_lev]
    elong = df.apply(lambda x: (len(x.target) - len(x.source))/ len(x.target), axis=1)
    df = df[elong <= max_elong_rate]
    return pd.DataFrame({SOURCE_COLUMN_NAME:df.source.str.strip('–—').str.strip(), TARGET_COLUMN_NAME:df.target.str.strip('–—').str.strip()})

def prepare_data_for_eval(data):
    refs = []
    orig = data[SOURCE_COLUMN_NAME].unique()
    for sent in tqdm(orig):
        refs.append(data.where(data[SOURCE_COLUMN_NAME] == sent)[TARGET_COLUMN_NAME].dropna())
    return list(orig), refs

def info(paths, datasets):
    for path, dataset in zip(paths, datasets):
        dataset_name = path.split('/')[-1]
        print(f'Name: {dataset_name}\nSize: {dataset.shape[0]}')