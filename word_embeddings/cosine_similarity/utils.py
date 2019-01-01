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

    # remove punctuation
    for column in medical_data:
        medical_data[column] = medical_data[column].str.replace('[{}]'.format(string.punctuation), '')

    return medical_data


def read_conf(data_dir):
    json_file = open(os.path.join(data_dir, 'medical.json')).read()
    json_data = json.loads(json_file, encoding='utf-8')
    return json_data


def plot_groups_histograms(control_scores_by_question,
                           patient_scores_by_question,
                           win_size,
                           header,
                           output_dir):
    plt.clf()
    plt.hist(control_scores_by_question, label='control')
    plt.hist(patient_scores_by_question, label='patients')
    plt.legend()
    plt.title('Cos-Sim Histogram Per-Question For Win: {} with {}'.format(win_size, header))
    plt.savefig(os.path.join(output_dir, "cos_sim_per_question_histogram_win{}_{}.png".format(win_size, header)))


def plot_groups_scores_by_question(control_scores_by_question,
                                   patient_scores_by_question,
                                   win_size,
                                   header,
                                   output_dir):
    plt.clf()
    calc_Xy_by_question_and_plot(control_scores_by_question, marker='*', c='red', label='control')
    calc_Xy_by_question_and_plot(patient_scores_by_question, marker='.', c='blue', label='patients')

    plt.xlabel('questions')
    plt.ylabel('cos sim scores')
    plt.xticks(range(1, 19))
    plt.legend()
    plt.title('Cos-Sim Per-Question For Win: {} with {}'.format(win_size, header))
    plt.savefig(os.path.join(output_dir, "cos_sim_per_question_win{}_{}.png".format(win_size, header)))


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


def ttest_results_to_csv(tests, output_dir, logger):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('max_colwidth', -1)
    dfs = []
    headers = ['t-statistic', 'p-value']
    for i in tests:
        df = pd.DataFrame([(item.tstat, item.pval) for item in i.questions_list], columns=headers)
        dfs += [df]

    keys = ['header: {}, window: {}'.format(test.header, test.window_size) for test in tests]
    dfs = pd.concat(dfs, axis=1, keys=keys)
    dfs.index += 1
    dfs.insert(0, 'question', range(1, len(tests[0].questions_list) + 1))
    logger.debug(dfs)
    dfs.to_csv(os.path.join(output_dir, "t-test_results.csv"), index=False)


def cossim_scores_to_csv(tests, output_dir, logger):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('max_colwidth', -1)
    dfs = []
    headers = ['user', 'question', 'cos-sim score']
    for i in tests:
        df = pd.DataFrame([(item.userid, item.question_num, item.score) for item in i.questions_list], columns=headers)
        dfs += [df]

    keys = ['header: {}, window: {}'.format(test.header, test.window_size) for test in tests]
    dfs = pd.concat(dfs, axis=1, keys=keys)
    logger.debug(dfs)
    dfs.to_csv(os.path.join(output_dir, "cossim_results.csv"), index=False)


def plot_grid_search(grid_search, output_dir):
    plt.clf()

    for pos_tags, diff_scores in grid_search:
        X = [score[1] for score in diff_scores]
        y = [score[0] for score in diff_scores]
        plt.scatter(X, y, label=pos_tags)

    plt.xlabel('cos-sim diff')
    plt.ylabel('window size')
    plt.yticks(range(1, 5))
    plt.legend()
    plt.title('Grid Search - window size and cos-sim diff of groups')

    plt.savefig(os.path.join(output_dir, "grid_search.png"))
