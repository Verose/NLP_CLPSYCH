import json
import os
import string
import warnings

import pandas as pd

from word_embeddings.cosine_similarity.cosine_similarity import CosineSimilarity

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models.wrappers import FastText


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
    cfg['mode'] = json_data['mode']
    return cfg


def average_cosine_similarity_several_window_sizes():
    for win_size in range(1, conf['size'] + 1):
        cosine_calcs = CosineSimilarity(model, data, data_dir, conf['mode'], window_size=win_size)
        cosine_calcs.calculate_all_avg_scores()

        control_score = cosine_calcs.calculate_avg_score_for_group('control')
        patients_score = cosine_calcs.calculate_avg_score_for_group('patients')
        print('Scores for window size {}: \nControl: {}, Patients: {}'.format(win_size, control_score, patients_score))


if __name__ == '__main__':
    data = get_medical_data()
    conf = read_conf()
    data_dir = os.path.join('..', 'data')
    model = FastText.load_fasttext_format(os.path.join(data_dir, 'FastText-pretrained-hebrew', 'wiki.he.bin'))

    average_cosine_similarity_several_window_sizes()
