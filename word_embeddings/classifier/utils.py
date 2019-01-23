import logging
import os

OUTPUT_DIR = os.path.join('..', 'outputs')
DATA_DIR = os.path.join('..', 'data')

LOGGER = logging.getLogger('Main')
LOGGER.setLevel(logging.INFO)
file_handler = logging.FileHandler(filename=os.path.join(OUTPUT_DIR, 'classifier_outputs.txt'), encoding='utf-8')
formatter = logging.Formatter(u"%(message)s")
file_handler.setFormatter(formatter)
LOGGER.addHandler(file_handler)


def remove_females(df, removed):
    # only keep male users
    res = df[df['gender'] == 1]
    removed.extend(list(df[df['gender'] == 2]['id']))
    return res


def remove_depressed(df, removed):
    # only keep control and schizophrenia groups
    res = df[(df['diagnosys_group'] == 'control') | (df['diagnosys_group'] == 'schizophrenia')]
    removed.extend(list(df[df['diagnosys_group'] == 'depression']['id']))
    return res
