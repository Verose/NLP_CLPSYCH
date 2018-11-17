import json
import os
import pandas as pd
import string

from gensim.models.wrappers import FastText

from word_embeddings.cosine_similarity.CosineSimilarity import CosineSimilarity


def get_medical_data():
    medical_data = pd.read_csv(os.path.join('..', 'data', 'all_data.csv'))
    medical_data = medical_data.iloc[1:]  # remove first row

    # remove punctuation
    for column in medical_data:
        medical_data[column] = medical_data[column].str.replace('[{}]'.format(string.punctuation), '')

    return medical_data


def read_conf():
    cfg = {}
    json_file = open('medical.json').read()
    json_data = json.loads(json_file, encoding='utf-8')
    cfg['size'] = json_data['window']['size']
    return cfg


if __name__ == '__main__':
    data = get_medical_data()
    conf = read_conf()
    model = FastText.load_fasttext_format(os.path.join('..', 'data', 'FastText-pretrained-hebrew', 'wiki.he.bin'))

    cosine_calcs = CosineSimilarity(model, data, conf['size'])
    cosine_calcs.calculate_all_avg_scores()
