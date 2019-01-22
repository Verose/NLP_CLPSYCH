import glob
import json
import os
import sys
from collections import defaultdict
from time import sleep

import pandas as pd
import requests
from tqdm import tqdm

DATA_DIR = os.path.join('..', 'data')
OUTPUTS_DIR = os.path.join('..', 'outputs')
# sent = "גנן גידל דגן בגן"
# sent = "הבית הלבן הגדול"
# sent = "היא הלכה מהר"


def get_response(headers, data):
    counter = 0
    retry_timer = 5

    while True:
        try:
            response = requests.post('http://onlp.openu.org.il:8000/yap/heb/joint', headers=headers, data=data)
            return response
        except requests.exceptions.ConnectionError:
            counter += 1

            if counter == 100:
                print('More than 500 seconds without getting a response! Exiting...')
                exit(0)

            sleep(retry_timer)
            retry_timer *= 2


def get_dependency_tree_for_sentence(sent):
    headers = {
        'Content-Type': 'application/json',
    }

    sentence = u"{}  ".format(sent)
    data = {"text": sentence}
    data = json.dumps(data)

    response = get_response(headers, data)
    tree = json.loads(response.text)
    dep_tree = tree['dep_tree'].split('\n')
    dep_dict = {}

    for dep in dep_tree:
        if not dep:
            continue

        dep = dep[:-4]  # remove trailing \t\t
        dep = dep.split('\t')
        num = dep[0]
        word = dep[1]
        tag = dep[3]
        second = dep[4]
        next_word = dep[6]
        dep = dep[7]

        dep_dict[num] = {
            "word": word,
            "tag": tag,
            "tag2": second,
            "next": next_word,
            "dep": dep
        }

    return dep_dict


def get_relevant_set():
    data = pd.read_csv(os.path.join(DATA_DIR, 'all_data.csv'))
    data = data.iloc[:, 2:]  # ignore first 2 columns
    relevant_set_nouns = defaultdict(list)
    relevant_set_verbs = defaultdict(list)

    noun_tags = ['NN', 'NNP', 'NNPS', 'NNS']
    adjective_tags = ['JJ', 'JJR', 'JJS']
    verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    adverb_tags = ['RB', 'RBR', 'RBS', 'RP']

    for i, row in tqdm(data.iterrows(), file=sys.stdout, total=len(data), desc='All Users'):
        for ans in tqdm(row, file=sys.stdout, total=len(row), leave=False, desc='Questions'):
            for sentence in ans.split('.'):
                if not sentence:
                    continue

                sleep(1)
                dep_tree = get_dependency_tree_for_sentence(sentence)

                for dependency in dep_tree.values():
                    if dependency['dep'] == 'ROOT':
                        continue

                    curr_tag = dependency['tag']
                    curr_word = dependency['word']
                    next_tag = dep_tree[dependency['next']]['tag']
                    next_word = dep_tree[dependency['next']]['word']

                    if curr_tag in adjective_tags and next_tag in noun_tags:
                        relevant_set_nouns[next_word].append(curr_word)
                    elif curr_tag in adverb_tags and next_tag in verb_tags:
                        relevant_set_verbs[next_word].append(curr_word)

    with open(os.path.join(OUTPUTS_DIR, 'relevant_set_nouns.json'), 'w', encoding='utf-8') as out_file:
        json.dump(relevant_set_nouns, out_file, ensure_ascii=False)
    with open(os.path.join(OUTPUTS_DIR, 'relevant_set_verbs.json'), 'w', encoding='utf-8') as out_file:
        json.dump(relevant_set_verbs, out_file, ensure_ascii=False)


if __name__ == '__main__':
    get_relevant_set()
