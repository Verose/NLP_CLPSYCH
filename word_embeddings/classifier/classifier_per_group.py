import os

import pandas as pd
from scipy import stats

from word_embeddings.common.utils import DATA_DIR, OUTPUTS_DIR


def avg_for_column(scores, column, test):
    if test == 'age':
        return age_avg_for_column(scores, column)
    else:
        return education_avg_for_column(scores, column)


def age_avg_for_column(scores, column):
    young = scores[scores['age'] <= 35][column].mean()
    old = scores[scores['age'] > 35][column].mean()
    return young, old


def education_avg_for_column(scores, column):
    academia = scores[scores['education'] == 'hs'][column].mean()
    high_school = scores[scores['education'] == 'acd'][column].mean()
    return high_school, academia


def avg_dep(scores_dependency, test, columns):
    # calculate averages for dependency scores
    dependency_nouns = avg_for_column(scores_dependency, 'nouns', test)
    dependency_verbs = avg_for_column(scores_dependency, 'verbs', test)

    df = pd.DataFrame.from_dict({'nouns': dependency_nouns, 'verbs': dependency_verbs})
    df.insert(0, 'group', columns)
    df.to_csv(os.path.join(OUTPUTS_DIR, "{}_dep.csv".format(test)), index=False)

    # return 2 lists: young results and old results
    young = [dependency_nouns[0], dependency_verbs[0]]
    old = [dependency_nouns[1], dependency_verbs[1]]

    tstatistic, pvalue = stats.ttest_ind(young, old)
    df = pd.DataFrame([[tstatistic, pvalue]], columns=['t-statistic', 'p-value'])
    df.to_csv(os.path.join(OUTPUTS_DIR, "{}_dep_t-test.csv".format(test)), index=False)

    return young, old


def avg_all(scores_all, test, columns):
    # calculate averages for all scores per window
    all_win1 = avg_for_column(scores_all, 'Window_1', test)
    all_win2 = avg_for_column(scores_all, 'Window_2', test)
    all_win3 = avg_for_column(scores_all, 'Window_3', test)
    all_win4 = avg_for_column(scores_all, 'Window_4', test)
    all_win5 = avg_for_column(scores_all, 'Window_5', test)

    df = pd.DataFrame.from_dict(
        {'Window_1': all_win1, 'Window_2': all_win2, 'Window_3': all_win3,
         'Window_4': all_win4, 'Window_5': all_win5})
    df.insert(0, 'group', columns)
    df.to_csv(os.path.join(OUTPUTS_DIR, "{}_all.csv".format(test)), index=False)

    # return 2 lists: young results and old results
    young = [res[0] for res in [all_win1, all_win2, all_win3, all_win4, all_win5]]
    old = [res[1] for res in [all_win1, all_win2, all_win3, all_win4, all_win5]]

    tstatistic, pvalue = stats.ttest_ind(young, old)
    df = pd.DataFrame([[tstatistic, pvalue]], columns=['t-statistic', 'p-value'])
    df.to_csv(os.path.join(OUTPUTS_DIR, "{}_all_t-test.csv".format(test)), index=False)

    return young, old


def avg_content(scores_content, test, columns):
    # calculate averages for content scores per window
    content_win1 = avg_for_column(scores_content, 'Window_1', test)
    content_win2 = avg_for_column(scores_content, 'Window_2', test)
    content_win3 = avg_for_column(scores_content, 'Window_3', test)
    content_win4 = avg_for_column(scores_content, 'Window_4', test)
    content_win5 = avg_for_column(scores_content, 'Window_5', test)

    df = pd.DataFrame.from_dict(
        {'Window_1': content_win1, 'Window_2': content_win2, 'Window_3': content_win3,
         'Window_4': content_win4, 'Window_5': content_win5})
    df.insert(0, 'group', columns)
    df.to_csv(os.path.join(OUTPUTS_DIR, "{}_content.csv".format(test)), index=False)

    # return 2 lists: young results and old results
    young = [res[0] for res in [content_win1, content_win2, content_win3, content_win4, content_win5]]
    old = [res[1] for res in [content_win1, content_win2, content_win3, content_win4, content_win5]]

    tstatistic, pvalue = stats.ttest_ind(young, old)
    df = pd.DataFrame([[tstatistic, pvalue]], columns=['t-statistic', 'p-value'])
    df.to_csv(os.path.join(OUTPUTS_DIR, "{}_content_t-test.csv".format(test)), index=False)

    return young, old


def classify_per_age_groups(scores_all, scores_content, scores_dependency):
    """
    Classify patients/controls group by age difference: age <= 35, and age > 35.
    :return:
    """
    young_dep, old_dep = avg_dep(scores_dependency, 'age', ['young', 'old'])
    young_all, old_all = avg_all(scores_all, 'age', ['young', 'old'])
    young_content, old_content = avg_content(scores_content, 'age', ['young', 'old'])

    tstatistic, pvalue = stats.ttest_ind(young_dep + young_all + young_content, old_dep + old_all + old_content)
    df = pd.DataFrame([[tstatistic, pvalue]], columns=['t-statistic', 'p-value'])
    df.to_csv(os.path.join(OUTPUTS_DIR, "age_t-test.csv"), index=False)


def classify_per_education_groups(scores_all, scores_content, scores_dependency):
    """
    Classify patients/controls group by age difference: age <= 35, and age > 35.
    :return:
    """
    young_dep, old_dep = avg_dep(scores_dependency, 'edu', ['hs', 'acd'])
    young_all, old_all = avg_all(scores_all, 'edu', ['hs', 'acd'])
    young_content, old_content = avg_content(scores_content, 'edu', ['hs', 'acd'])

    tstatistic, pvalue = stats.ttest_ind(young_dep + young_all + young_content, old_dep + old_all + old_content)
    df = pd.DataFrame([[tstatistic, pvalue]], columns=['t-statistic', 'p-value'])
    df.to_csv(os.path.join(OUTPUTS_DIR, "edu_t-test.csv"), index=False)


if __name__ == '__main__':
    scores_all = pd.read_csv(os.path.join(DATA_DIR, 'scores_all.csv'))
    scores_content = pd.read_csv(os.path.join(DATA_DIR, 'scores_content.csv'))
    scores_dependency = pd.read_csv(os.path.join(DATA_DIR, 'scores_dependency.csv'))
    all_questions = pd.read_csv(os.path.join(DATA_DIR, 'all_questions.csv'))

    all_questions.rename(index=str, columns={"id": "user"}, inplace=True)

    # add age and education column to scores_all
    scores_all_new = scores_all.merge(all_questions, on=['user'])
    scores_all = scores_all_new[list(scores_all.columns) + ['age', 'education']]

    # add age and education column to scores_content
    scores_content_new = scores_content.merge(all_questions, on=['user'])
    scores_content = scores_content_new[list(scores_content.columns) + ['age', 'education']]

    # add age and education and group columns to scores_dependency
    scores_dep_new = scores_dependency.merge(all_questions, on=['user'])
    scores_dependency = scores_dep_new[list(scores_dependency.columns) + ['age', 'education', 'label']]

    classify_per_age_groups(scores_all, scores_content, scores_dependency)
    classify_per_education_groups(scores_all, scores_content, scores_dependency)
