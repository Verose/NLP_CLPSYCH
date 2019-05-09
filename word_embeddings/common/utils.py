import glob
import json
import os
import pickle
import socket

import numpy as np
# import requests

OUTPUTS_DIR = os.path.join('..', 'outputs')
DATA_DIR = os.path.join('..', 'data')
HOST = socket.gethostbyname("localhost")


def read_conf():
    json_file = open(os.path.join(DATA_DIR, 'medical.json')).read()
    json_data = json.loads(json_file, encoding='utf-8')
    return json_data


def remove_females(df, removed):
    # only keep male users
    res = df[df['gender'] == 1]
    removed.extend(list(df[df['gender'] == 2]['id']))
    return res


def remove_depressed(df, removed):
    # only keep control and schizophrenia groups
    res = df[df['diagnosys_group'].isin(['control', 'schizophrenia'])]
    removed.extend(list(df[df['diagnosys_group'] == 'depression']['id']))
    return res


def get_vector_for_word(model, words):
    vectors = []

    for word in words:
        vectors.append(model[word].tolist())

    return vectors

# def get_vector_for_word(words):
    # headers = {
    #     'Content-Type': 'application/json',
    # }
    # data = {"words": words}
    # data = json.dumps(data)
    #
    # response = requests.get(
    #     'http://{}:5000/word_embeddings/vectors'.format(HOST), data=data, headers=headers, timeout=8
    # )
    # result = json.loads(response.text)
    # return result


def get_words_in_model(model, words):
    result_dict = {}

    for word in words:
        result_dict[word] = word in model

    return result_dict

# def get_words_in_model(words):
#     headers = {
#         'Content-Type': 'application/json',
#     }
#     data = {"words": words}
#     data = json.dumps(data)
#
#     response = requests.get(
#         'http://{}:5000/word_embeddings/is_in'.format(HOST), data=data, headers=headers, timeout=8
#     )
#     result = json.loads(response.text)
#     return result


def load_model(word_embeddings_file, is_rsdd=False):
    if is_rsdd:
        rsdd_data_path = os.path.join(DATA_DIR, 'ft_pretrained', 'rsdd_word2vec.pickle')
        with open(rsdd_data_path, 'rb') as f:
            model = pickle.load(f)
    else:
        model = {}
        words = get_words()
        with open(os.path.join(DATA_DIR, 'ft_pretrained', word_embeddings_file), encoding="utf-8") as embeddings:
            for line in embeddings:
                values = line.split(" ")
                if values[0] in words:
                    model[values[0]] = np.array([float(val) for val in values[1:]])
    return model


def get_words():
    words = []
    pos_tags_generator = pos_tags_jsons_generator()

    for _, ans_jsons in pos_tags_generator:
        for tags in ans_jsons.values():
            tokens = tags['tokens']
            words += tokens
    return set(words)


def get_sets_words():
    control_nouns = read_relevant_set('nouns', 'control')
    control_verbs = read_relevant_set('verbs', 'control')
    patients_nouns = read_relevant_set('nouns', 'patients')
    patients_verbs = read_relevant_set('verbs', 'patients')
    reference_nouns = read_reference_set('nouns')
    reference_verbs = read_reference_set('verbs')
    words = []

    for words_dict in [control_nouns, control_verbs, patients_nouns, patients_verbs, reference_nouns, reference_verbs]:
        words.extend(list(words_dict.keys()))
        words.extend([item for sublist in list(words_dict.values()) for item in sublist])
    return set(words)


def pos_tags_jsons_generator():
    json_pattern = os.path.join(DATA_DIR, 'answers_pos_tags', '*.json')
    json_files = [pos_json for pos_json in glob.glob(json_pattern) if pos_json.endswith('.json')]

    for file in json_files:
        with open(file, encoding='utf-8') as f:
            ans_pos_tags = json.load(f)
            yield int(os.path.basename(file).split('.')[0]), ans_pos_tags


def read_relevant_set(pos_tag, group):
    with open(os.path.join(DATA_DIR, 'relevant_sets', group, '{}_norm_relevant_set_{}.json'.format(group, pos_tag)),
              encoding='utf-8') as f:
        relevant_set = json.load(f)
    return relevant_set


def read_reference_set(pos_tag):
    with open(os.path.join(DATA_DIR, 'reference_sets', 'norm_reference_set_{}.json'.format(pos_tag)),
              encoding='utf-8') as f:
        relevant_set = json.load(f)
    return relevant_set
