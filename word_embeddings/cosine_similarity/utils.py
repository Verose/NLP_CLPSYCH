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
    header_prefix = ['group', 'user', 'question', 'valid words', '#valid words']
    header = ['cos-sim score']
    for i, t in enumerate(tests):
        if i == 0:
            df = pd.DataFrame(
                [(item.group, item.userid, item.question_num, item.valid_words, item.n_valid, item.score) for item in
                 t.questions_list],
                columns=header_prefix + header)
        else:
            df = pd.DataFrame([(item.score,) for item in t.questions_list], columns=header)
        dfs += [df]

    keys = ['header: {}, window: {}'.format(test.header, test.window_size) for test in tests]
    dfs = pd.concat(dfs, axis=1, keys=keys)
    logger.debug(dfs)
    dfs.to_csv(os.path.join(OUTPUTS_DIR, "cossim_results{}.csv".format(unique_name)), index=False)


def plot_window_size_vs_scores_per_group(groups_scores):
    """
    Plots a figure for every word category set (e.g. all, content words, noun-verb, verbs) where:
    win_sizes axis: window sizes
    mean_scores axis: average cosine sim
    plot two curves: one for patients and one for control.
    :param groups_scores: dictionary from pos tags tests to control/patients tuples of (win sizes, scores_list)
    :return: nothing
    """

    for i, (pos_tags, groups) in enumerate(groups_scores.items()):
        plt.clf()
        fig, ax = plt.subplots()
        control_scores = groups["control"]
        patients_scores = groups["patients"]

        win_sizes = grid_plot_group(ax, i, control_scores, marker='*', c='red', label='control', xytext=(-20, 20))
        win_sizes = grid_plot_group(ax, i, patients_scores, marker='.', c='blue', label='patients', xytext=(-20, -20))

        ax.set_xlabel('window sizes')
        ax.set_xticks(win_sizes)
        ax.set_ylabel('avg cos-sim score')
        plt.legend()
        ax.set_title('Group Scores Per Window Size {}'.format(pos_tags))

        plt.savefig(os.path.join(OUTPUTS_DIR, "group_scores_per_win_size_{}.png".format(pos_tags)))


def grid_plot_group(ax, i, scores, marker, c, label, xytext):
    # plot patients group
    win_sizes = [score[0] for score in scores]
    scores = [score[1] for score in scores]

    if i == 0:
        ax.plot(win_sizes, scores, marker=marker, c=c, label=label)
    else:
        ax.plot(win_sizes, scores, marker=marker, c=c)
    ax.annotate('score',
                xy=(win_sizes[i], scores[i]), xycoords='data', xytext=xytext,
                textcoords='offset points', arrowprops=dict(arrowstyle="->"), size=8)
    return win_sizes


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
