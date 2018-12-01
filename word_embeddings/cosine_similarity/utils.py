import json
import os
import string

import pandas as pd
from matplotlib import pyplot as plt


def get_vector_repr_of_word(model, word):
    try:
        return model[word]
    except KeyError:
        if str.isdecimal(word):
            replacement_word = '<מספר>'
        elif str.isalpha(word):
            replacement_word = '<אנגלית>'
        elif any(i.isdigit() for i in word) and any("\u0590" <= c <= "\u05EA" for c in word):
            replacement_word = '<אות ומספר>'
        else:
            replacement_word = '<לא ידוע>'
        return model[replacement_word]


def get_medical_data(data_dir):
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


def plot_groups_histograms(control_scores_by_question, patient_scores_by_question, win_size, output_dir):
    plt.clf()
    plt.hist(control_scores_by_question, label='control')
    plt.hist(patient_scores_by_question, label='patients')
    plt.legend()
    plt.title('Cosine Similarity Scores Histogram Per-Question For Window Size: {}'.format(win_size))
    plt.savefig(os.path.join(output_dir, "cos_sim_per_question_histogram_win{}.png".format(win_size)))
    # plt.show()


def plot_groups_scores_by_question(control_scores_by_question, patient_scores_by_question, win_size, output_dir):
    plt.clf()
    calc_Xy_by_question_and_plot(control_scores_by_question, marker='*', c='red', label='control')
    calc_Xy_by_question_and_plot(patient_scores_by_question, marker='.', c='blue', label='patients')

    plt.xlabel('questions')
    plt.ylabel('cos sim scores')
    plt.xticks(range(1, 19))
    plt.legend()
    plt.title('Cosine Similarity Scores Per-Question For Window Size: {}'.format(win_size))
    plt.savefig(os.path.join(output_dir, "cos_sim_per_question_win{}.png".format(win_size)))
    # plt.show()


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
