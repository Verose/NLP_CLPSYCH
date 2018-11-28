import json
import logging
import os
import string
import warnings

import matplotlib.pyplot as plt
import pandas as pd

from word_embeddings.cosine_similarity.cosine_similarity import CosineSimilarity

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models.wrappers import FastText

output_dir = os.path.join('..', 'outputs')
logger = logging.getLogger('Main')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(filename=os.path.join(output_dir, 'main_outputs.txt'))
formatter = logging.Formatter("%(message)s", '%H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def get_medical_data():
    medical_data = pd.read_csv(os.path.join(data_dir, 'all_data.csv'))
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
    cfg['mode'] = json_data['mode']['type']
    cfg['extras'] = json_data['mode']['extras']
    cfg['plot'] = json_data['output']['plot']
    return cfg


def average_cosine_similarity_several_window_sizes():
    for win_size in range(1, conf['size'] + 1):
        cosine_calcs = CosineSimilarity(model, data, conf['mode'], conf['extras'], win_size, data_dir)
        cosine_calcs.calculate_all_avg_scores()

        if conf['mode'] == 'pos' and win_size == 1:
            control_items = cosine_calcs.calculate_items_for_group('control')
            patients_items = cosine_calcs.calculate_items_for_group('patients')
            logger.info('Showing results using POS tags: {}'.format(conf['extras']['pos']))
            logger.info('Average valid words per answer: Control: {0:.4f}, Patients: {1:.4f}'
                        .format(control_items, patients_items))
            # logger.info('\nCosine Similarity is calculated as an average over the answers.')
            # logger.info('An answer is calculated by the average cosine similarity over all possible windows')
            # logger.info('Word repetitions are calculated as an average over the answers.')
            # logger.info('An answer is calculated by summation over all possible windows.')

        control_score = cosine_calcs.calculate_avg_score_for_group('control')
        patients_score = cosine_calcs.calculate_avg_score_for_group('patients')
        logger.info('\nScores for window size {}: '.format(win_size))
        logger.info('Average Cos-Sim per answer: Control: {0:.4f}, Patients: {1:.4f}'
                    .format(control_score, patients_score))

        if conf['plot']:
            plot_control_patients_score_by_question(cosine_calcs, win_size)

        if conf['mode'] == 'pos':
            control_repetitions = cosine_calcs.calculate_repetitions_for_group('control')
            patients_repetitions = cosine_calcs.calculate_repetitions_for_group('patients')
            logger.info('Average word repetition occurrences per answer: Control: {0:.4f}, Patients: {1:.4f}'
                        .format(control_repetitions, patients_repetitions))


def plot_control_patients_score_by_question(cosine_calcs, win_size):
    control_user_score_by_question, patient_user_score_by_question = cosine_calcs.get_user_to_question_scores()
    calc_Xy_by_question_and_plot(control_user_score_by_question, marker='*', c='red', label='control')
    calc_Xy_by_question_and_plot(patient_user_score_by_question, marker='.', c='blue', label='patients')

    plt.xlabel('questions')
    plt.ylabel('cos sim scores')
    plt.xticks(range(1, 19))
    plt.legend()
    plt.title('Cosine Similarity Scores Per-User Per-Question For Window Size: {}'.format(win_size))
    plt.savefig(os.path.join(output_dir, "cos_sim_per_question_win{}.png".format(win_size)))
    plt.show()


def calc_Xy_by_question_and_plot(user_score_by_question, marker, c, label):
    X = []
    y = []

    for user_to_score in user_score_by_question.values():
        questions = []
        scores = []
        for question, score in user_to_score.items():
            questions += [question]
            scores += [score]
        X += [questions]
        y += [scores]

    plt.scatter(X, y, marker=marker, c=c, label=label)


if __name__ == '__main__':
    data_dir = os.path.join('..', 'data')
    data = get_medical_data()
    conf = read_conf()
    model = FastText.load_fasttext_format(os.path.join(data_dir, 'FastText-pretrained-hebrew', 'wiki.he.bin'))

    average_cosine_similarity_several_window_sizes()
