import glob
import json
import os
import string

import pandas as pd
from matplotlib import pyplot as plt

from word_embeddings.common.utils import remove_females, remove_depressed, DATA_DIR, OUTPUTS_DIR


def get_medical_data(clean_data=False):
    medical_data = pd.read_csv(os.path.join(DATA_DIR, 'all_data.csv'))

    if clean_data:
        # check features csv for gender and diagnosis group
        df_res = pd.read_csv(os.path.join(DATA_DIR, 'features_all.csv'))
        df_res = remove_females(df_res, [])
        df_res = remove_depressed(df_res, [])
        users = df_res['id'].values
        medical_data = medical_data[medical_data['id'].isin(users)]

    # remove punctuation
    for column in medical_data:
        medical_data[column] = medical_data[column].str.replace('[{}]'.format(string.punctuation), '')

    return medical_data


def read_conf():
    json_file = open(os.path.join(DATA_DIR, 'medical.json')).read()
    json_data = json.loads(json_file, encoding='utf-8')
    return json_data


def pos_tags_jsons_generator():
    json_pattern = os.path.join(DATA_DIR, 'answers_pos_tags', '*.json')
    json_files = [pos_json for pos_json in glob.glob(json_pattern) if pos_json.endswith('.json')]

    for file in json_files:
        with open(file, encoding='utf-8') as f:
            ans_pos_tags = json.load(f)
            yield int(os.path.basename(file).split('.')[0]), ans_pos_tags


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


def ttest_results_to_csv(tests, logger, unique_name=''):
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
    dfs.to_csv(os.path.join(OUTPUTS_DIR, "t-test_results{}.csv".format(unique_name)), index=False)


def cossim_scores_to_csv(tests, logger, unique_name=''):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('max_colwidth', -1)
    dfs = []
    header_prefix = ['group', 'user', 'question']
    header = ['cos-sim score', 'valid words', '#valid words']
    for i, t in enumerate(tests):
        if i == 0:
            df = pd.DataFrame(
                [(item.group, item.userid, item.question_num, item.score, item.valid_words, item.n_valid) for item in
                 t.questions_list],
                columns=header_prefix + header)
        else:
            df = pd.DataFrame(
                [(item.score, item.valid_words, item.n_valid) for item in t.questions_list],
                columns=header)
        dfs += [df]

    keys = ['header: {}, window: {}'.format(test.header, test.window_size) for test in tests]
    dfs = pd.concat(dfs, axis=1, keys=keys)
    logger.debug(dfs)
    dfs.to_csv(os.path.join(OUTPUTS_DIR, "cossim_results{}.csv".format(unique_name)), index=False)


def plot_window_size_vs_scores_per_group(groups_scores):
    """
    One figure: for every word category set (e.g. all, content words, noun-verb, verbs):
    win_sizes axis: window sizes
    avg_scores axis: average cosine sim
    plot two curves: one for patients and one for control.
    :param groups_scores: dictionary from pos tags tests to control/patients tuples of (win sizes, scores_list)
    :return: nothing
    """
    plt.clf()
    fig, ax = plt.subplots()
    win_sizes = range(1, 5)

    for i, (pos_tags, groups) in enumerate(groups_scores.items()):
        control_scores = groups["control"]
        patients_scores = groups["patients"]

        # plot control group
        win_sizes = [score[0] for score in control_scores]
        avg_scores = [score[1] for score in control_scores]
        if i == 0:
            ax.plot(win_sizes, avg_scores, marker='*', c='red', label='control')
        else:
            ax.plot(win_sizes, avg_scores, marker='*', c='red')
        ax.annotate(pos_tags, xy=(win_sizes[i], avg_scores[i]), xycoords='data', xytext=(-20, 20),
                    textcoords='offset points', arrowprops=dict(arrowstyle="->"), size=8)

        # plot patients group
        win_sizes = [score[0] for score in patients_scores]
        avg_scores = [score[1] for score in patients_scores]
        if i == 0:
            ax.plot(win_sizes, avg_scores, marker='.', c='blue', label='patients')
        else:
            ax.plot(win_sizes, avg_scores, marker='.', c='blue')
        ax.annotate(pos_tags,
                    xy=(win_sizes[i], avg_scores[i]), xycoords='data', xytext=(-20, -20),
                    textcoords='offset points', arrowprops=dict(arrowstyle="->"), size=8)

    ax.set_xlabel('window sizes')
    ax.set_xticks(win_sizes)
    ax.set_ylabel('avg cos-sim score')
    plt.legend()
    ax.set_title('Group Scores Per Window Size')

    plt.savefig(os.path.join(OUTPUTS_DIR, "group_scores_per_win_size.png"))


def plot_grid_search(grid_search, output_dir):
    plt.clf()
    y = range(1, 5)

    for pos_tags, diff_scores in grid_search:
        X = [score[1] for score in diff_scores]
        y = [score[0] for score in diff_scores]
        plt.scatter(X, y, label=pos_tags)

    plt.xlabel('cos-sim diff')
    plt.ylabel('window size')
    plt.yticks(y)
    plt.legend(fontsize='x-small')
    plt.title('Grid Search - window size and cos-sim diff of groups')

    plt.savefig(os.path.join(output_dir, "grid_search.png"))
