import glob
import json
import os

import numpy as np

OUTPUTS_DIR = os.path.join('..', 'outputs')
DATA_DIR = os.path.join('..', 'data')


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
# def get_vector_repr_of_word(word, is_in=False):
#     headers = {
#         'Content-Type': 'application/json',
#     }
#     data = {"word": word, 'isin': is_in}
#     data = json.dumps(data)
#
#     response = requests.get('http://localhost:5000/word_embeddings', data=data, headers=headers, timeout=8)
#
#     if is_in:
#         result = bool(response.text)
#     else:
#         result = np.fromstring(response.text[1:-1], dtype=float, sep=',')
#     return result


def load_model(words, word_embeddings_file):
    model = {}
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


def pos_tags_jsons_generator():
    json_pattern = os.path.join(DATA_DIR, 'answers_pos_tags', '*.json')
    json_files = [pos_json for pos_json in glob.glob(json_pattern) if pos_json.endswith('.json')]

    for file in json_files:
        with open(file, encoding='utf-8') as f:
            ans_pos_tags = json.load(f)
            yield int(os.path.basename(file).split('.')[0]), ans_pos_tags
