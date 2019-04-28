import os
import string

import pandas as pd
from matplotlib import pyplot as plt

from word_embeddings.common.utils import remove_females, get_depressed, DATA_DIR, OUTPUTS_DIR


def get_medical_data(clean_data=False):
    medical_data = pd.read_csv(os.path.join(DATA_DIR, 'all_data.csv'))

    if clean_data:
        # check features csv for gender and diagnosis group
        df_res = pd.read_csv(os.path.join(DATA_DIR, 'features_all.csv'))
        df_res = remove_females(df_res, [])
        # df_res = remove_depressed(df_res, [])
        df_res = get_depressed(df_res)
        users = df_res['id'].values
        medical_data = medical_data[medical_data['id'].isin(users)]

    # remove punctuation
    for column in medical_data:
        medical_data[column] = medical_data[column].str.replace('[{}]'.format(string.punctuation), '')

    return medical_data


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
    questions = [a.question_num for a in tests[0].questions_list]
    dfs.insert(0, 'question', questions)
    logger.debug(dfs)
    dfs.to_csv(os.path.join(OUTPUTS_DIR, "t-test_results{}.csv".format(unique_name)), index=False)


def cossim_scores_per_user_to_csv(users_to_scores, pos_tags):
    win_sizes = []
    dfs = None

    for win_size, control_scores, patients_scores in users_to_scores:
        control_items = [(user, score, 'control') for user, score in control_scores.items()]
        patients_items = [(user, score, 'patients') for user, score in patients_scores.items()]
        df = pd.DataFrame(control_items + patients_items, columns=['user', 'Window_{}'.format(win_size), 'Group'])
        dfs = dfs.merge(df, on=['user', 'Group']) if dfs is not None else df
        win_sizes.append(win_size)

    rearrange = ['user'] + ['Window_{}'.format(win_size) for win_size in win_sizes] + ['Group']
    dfs = dfs[rearrange]
    dfs.to_csv(os.path.join(OUTPUTS_DIR, "scores_{}.csv".format(pos_tags)), index=False)
    pass


def cossim_scores_per_question_to_csv(tests, logger, unique_name=''):
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


def plot_word_categories_to_coherence(groups_scores, ttest_scores, pos_tags):
    """
    For given word category set (e.g. all words, content words):
    x axis: window sizes, y axis: average cosine sim
    plots two curves: one for patients and one for control.
    :param groups_scores: dictionary from control/patients to tuples of (win sizes, scores_list)
    :param ttest_scores:
    :param pos_tags: the pos tags word category currently checked
    :return: nothing
    """
    plt.clf()
    fig, ax = plt.subplots()
    control_scores = groups_scores["control"]
    patients_scores = groups_scores["patients"]

    # plot control group
    win_sizes = [size for size, scores in control_scores]
    cont_avg_scores = [scores for size, scores in control_scores]
    ax.plot(win_sizes, cont_avg_scores, marker='*', c='blue', label='control')

    # plot patients group
    win_sizes = [size for size, scores in patients_scores]
    pat_avg_scores = [scores for size, scores in patients_scores]
    ax.plot(win_sizes, pat_avg_scores, marker='.', c='red', label='patients')

    ax.set_xlabel('k')
    ax.set_xticks(win_sizes)
    ax.set_ylabel('Coherence Score')
    plt.legend()
    ax.set_title('{} Words'.format(pos_tags))

    plt.savefig(os.path.join(OUTPUTS_DIR, "{}_words_to_coherence.png".format(pos_tags)))

    tstatistic = [tstatistic for _, tstatistic, pvalue in ttest_scores]
    pvalue = [pvalue for _, tstatistic, pvalue in ttest_scores]
    data = {'k': win_sizes, 'Control': cont_avg_scores, 'Patients': pat_avg_scores, 't': tstatistic, 'p': pvalue}
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(OUTPUTS_DIR, "{}_words_to_coherence.csv".format(pos_tags)), index=False)


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
