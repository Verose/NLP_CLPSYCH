import json
import optparse
import os
import string
import sys
from collections import defaultdict
from time import sleep

import pandas as pd
import requests
from tqdm import tqdm

DATA_DIR = os.path.join('..', 'data')
OUTPUTS_DIR = os.path.join('..', 'outputs')

noun_tags = ['NN', 'NNP', 'NNPS', 'NNS']
adjective_tags = ['JJ', 'JJR', 'JJS']
verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
adverb_tags = ['RB', 'RBR', 'RBS', 'RP']
punctuation = string.punctuation.replace('-', '')


def get_response(headers, data):
    timeout = 300
    retry_counter = 1

    while True:
        try:
            # response = requests.post('http://onlp.openu.org.il:8000/yap/heb/joint', headers=headers, data=data)
            response = requests.post('http://localhost:8000/yap/heb/joint', headers=headers, data=data, timeout=timeout)
            return response
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            if retry_counter > 10:
                print('\nFailed to get a response after 10 retries and timeout {}s. Exiting...'.format(timeout))
                return None
            sleep(5)


def get_dependency_tree_for_sentence(sent):
    headers = {
        'Content-Type': 'application/json',
    }

    sentence = u"{}  ".format(sent)
    data = {"text": sentence}
    data = json.dumps(data)

    response = get_response(headers, data)
    if not response:
        return None
    tree = json.loads(response.text)
    dep_tree = tree['dep_tree'].split('\n')
    dep_dict = {}

    for dep in dep_tree:
        if not dep:
            continue

        dep = dep[:-5]  # remove trailing \r\t\t
        dep = dep.split('\t')
        num = dep[0]
        word = dep[1]
        lemma = dep[2]
        tag = dep[3]
        head = dep[6]
        dependency = dep[7]

        dep_dict[num] = {
            "word": word,
            "tag": tag,
            "lemma": lemma,
            "next": head,
            "dep": dependency
        }

    return dep_dict


def save_sets(nouns_set, verbs_set, file_pattern):
    with open(os.path.join(OUTPUTS_DIR, file_pattern.format('nouns')), 'w', encoding='utf-8') as out_file:
        json.dump(nouns_set, out_file, ensure_ascii=False)
    with open(os.path.join(OUTPUTS_DIR, file_pattern.format('verbs')), 'w', encoding='utf-8') as out_file:
        json.dump(verbs_set, out_file, ensure_ascii=False)


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


def repair_document(sentence):
    # no need for newline (solves problem with newline in the middle of a sentence)
    sentence = sentence.replace('\n', ' ')

    # wrongly placed quotation marks
    sentence = sentence.replace('."', '".')
    sentence = sentence.replace(' "', '"')

    # will otherwise be removed later as punctuation
    sentence = sentence.replace('+', ' פלוס ')
    sentence = sentence.replace('%', ' אחוז ')
    sentence = sentence.replace('ו/או', 'ו או')
    sentence = sentence.replace('/', ' או ')
    sentence = sentence.replace('__', 'איקס')

    # redundant space before punctuation
    while ' ,' in sentence:
        sentence = sentence.replace(' ,', ',')
    while ' .' in sentence:
        sentence = sentence.replace(' .', '.')

    while '( ' in sentence:
        sentence = sentence.replace('( ', '(')
    while ' )' in sentence:
        sentence = sentence.replace(' )', ')')
    while ' ?' in sentence:
        sentence = sentence.replace(' ?', '?')
    while ' :' in sentence:
        sentence = sentence.replace(' :', ':')

    # misc
    sentence = sentence.replace('prefix=', '')

    # redundant spaces
    while '  ' in sentence:
        sentence = sentence.replace('  ', ' ')
    return sentence


def get_relevant_set(data):
    control_nouns, control_verbs = defaultdict(list), defaultdict(list)
    patients_nouns, patients_verbs = defaultdict(list), defaultdict(list)

    for i, row in tqdm(data.iterrows(), file=sys.stdout, total=len(data), desc='All Users'):
        is_control = row['label'] == 'control'
        row = row[2:]
        for ans in tqdm(row, file=sys.stdout, total=len(row), leave=False, desc='Questions'):
            if not ans or ans is pd.np.nan:  # some users didn't answer all of the questions
                continue

            ans = repair_document(ans)

            for sentence in ans.split('.'):
                if not sentence:
                    continue

                dep_tree = get_dependency_tree_for_sentence(sentence)
                if not dep_tree:
                    print('Saving current results and exiting...')
                    save_sets(control_nouns, control_verbs, 'control_relevant_set_{{}}_{}.json'.format(i))
                    save_sets(patients_nouns, patients_verbs, 'patients_relevant_set_{{}}_{}.json'.format(i))
                    exit(0)

                for dependency in dep_tree.values():
                    if dependency['next'] == '0':
                        continue

                    curr_tag = dependency['tag']
                    curr_word = dependency['lemma']
                    next_tag = dep_tree[dependency['next']]['tag']
                    next_word = dep_tree[dependency['next']]['lemma']

                    if curr_tag in adjective_tags and next_tag in noun_tags:
                        if is_control:
                            control_nouns[next_word].append(curr_word)
                        else:
                            patients_nouns[next_word].append(curr_word)
                    elif curr_tag in adverb_tags and next_tag in verb_tags:
                        if is_control:
                            control_verbs[next_word].append(curr_word)
                        else:
                            patients_verbs[next_word].append(curr_word)

    save_sets(control_nouns, control_verbs, 'control_norm_relevant_set_{}.json')
    save_sets(patients_nouns, patients_verbs, 'patients_norm_relevant_set_{}.json')


def get_reference_set(data, dataset):
    dataset = dataset.replace('/', '_') if dataset else None  # support nested folders
    control_nouns = read_relevant_set('nouns', 'control')
    control_verbs = read_relevant_set('verbs', 'control')
    patients_nouns = read_relevant_set('nouns', 'patients')
    patients_verbs = read_relevant_set('verbs', 'patients')

    reference_set_nouns = defaultdict(list)
    reference_set_verbs = defaultdict(list)

    for i, article in tqdm(enumerate(data), file=sys.stdout, total=len(data), desc='Articles'):
        with open(article, encoding='utf-8') as f:
            if 'haaretz' in dataset:
                article = f.read()
                article = article.replace('\n', '').split('.')[:-1]  # last is blank
            else:
                article = f.readlines()

        for sentence in tqdm(article, file=sys.stdout, total=len(article), leave=False, desc='Sentences'):
            if not sentence or '•' in sentence:  # skip empty sentences and lists
                continue

            sentence = repair_document(sentence)
            dep_tree = get_dependency_tree_for_sentence(sentence)
            if not dep_tree:
                print('Saving current results and exiting...')
                save_sets(reference_set_nouns, reference_set_verbs, 'reference_set_{{}}_{}_{}.json'.format(dataset, i))
                exit(0)

            for dependency in dep_tree.values():
                if dependency['next'] == '0':
                    continue

                curr_tag = dependency['tag']
                curr_word = dependency['lemma']
                next_tag = dep_tree[dependency['next']]['tag']
                next_word = dep_tree[dependency['next']]['lemma']

                relevant_noun = next_word in control_nouns or next_word in patients_nouns
                relevant_verb = next_word in control_verbs or next_word in patients_verbs

                if curr_tag in adjective_tags and next_tag in noun_tags and relevant_noun:
                    reference_set_nouns[next_word].append(curr_word)
                elif curr_tag in adverb_tags and next_tag in verb_tags and relevant_verb:
                    reference_set_verbs[next_word].append(curr_word)

    save_sets(reference_set_nouns, reference_set_verbs, 'norm_reference_set_{{}}_{}.json'.format(dataset))


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--set_type', action="store", choices=['relevant', 'reference'])
    parser.add_option('--folder', action="store", default="doctors_articles")
    options, remainder = parser.parse_args()

    if not os.path.exists(os.path.join(OUTPUTS_DIR, 'tmp')):
        os.makedirs(os.path.join(OUTPUTS_DIR, 'tmp'))

    if options.set_type == 'relevant':
        df = pd.read_csv(os.path.join(DATA_DIR, 'all_data.csv'))
        get_relevant_set(df)
    elif options.set_type == 'reference':
        articles_path = os.path.join(DATA_DIR, options.folder)
        articles = os.listdir(articles_path)
        articles = [os.path.join(articles_path, article) for article in articles]

        get_reference_set(articles, options.folder)
