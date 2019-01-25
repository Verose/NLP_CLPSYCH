import json
import optparse
import os
import string
import subprocess
import sys
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

DATA_DIR = os.path.join('..', 'data')
OUTPUTS_DIR = os.path.join('..', 'outputs')

noun_tags = ['NN', 'NNP', 'NNPS', 'NNS']
adjective_tags = ['JJ', 'JJR', 'JJS']
verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
adverb_tags = ['RB', 'RBR', 'RBS', 'RP']


def read_relevant_set(pos_tag, group):
    with open(os.path.join(DATA_DIR, 'relevant_sets', group, '{}_norm_relevant_set_{}.json'.format(group, pos_tag)),
              encoding='utf-8') as f:
        relevant_set = json.load(f)
    return relevant_set


def get_dependency_tree_for_sentence(sent, i):
    sent = sent.translate(str.maketrans("", "", string.punctuation))
    with open(os.path.join(OUTPUTS_DIR, 'tmp', 'sentence_{}.txt'.format(i)), 'w', encoding='utf-8', newline='\n') as f:
        [f.write(u'{}\n'.format(word)) for word in sent.split(' ')]
        f.write('\n')
    go_workspace = os.environ['GOPATH']
    yap_path = os.path.join(go_workspace, 'src', 'yap')
    yap = os.path.join(yap_path, 'yap.exe' if os.name == 'nt' else 'yap')

    hebma_cmd = [
        yap,
        "hebma",
        "-raw",
        os.path.join(OUTPUTS_DIR, 'tmp', 'sentence_{}.txt'.format(i)),
        "-out",
        os.path.join(OUTPUTS_DIR, 'tmp', "output_{}.txt".format(i))
    ]
    p = subprocess.Popen(hebma_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    p.communicate()

    joint_cmd = [
        yap,
        "joint",
        "-f",
        os.path.join(yap_path, 'conf', "jointzeager.yaml"),
        "-in",
        os.path.join(OUTPUTS_DIR, 'tmp', "output_{}.txt".format(i)),
        "-l",
        os.path.join(yap_path, 'conf', "hebtb.labels.conf"),
        "-m",
        os.path.join(yap_path, 'data', "joint_arc_zeager_model_temp_i33.b64"),
        "-nolemma",
        "-oc",
        os.path.join(OUTPUTS_DIR, 'tmp', "output_dep_{}.txt".format(i)),
        "-om",
        os.path.join(OUTPUTS_DIR, 'tmp', "output_om_{}.txt".format(i)),
        "-os",
        os.path.join(OUTPUTS_DIR, 'tmp', "output_os_{}.txt".format(i)),
        "-jointstr",
        "ArcGreedy",
        "-oraclestr",
        "ArcGreedy"
    ]
    p = subprocess.Popen(joint_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    p.communicate()

    with open(os.path.join(OUTPUTS_DIR, 'tmp', 'output_dep_{}.txt'.format(i)), encoding='utf-8') as f:
        dep_tree = f.read()

    dep_tree = dep_tree.split('\t_\t_\n')  # every line ends with \t_\t_\n
    dep_dict = {}
    # assert len([line for line in dep_tree if line.startswith('1\t')]) == 1

    for dep in dep_tree:
        if not dep:
            continue

        dep = dep.replace('\n', '')

        if not dep:
            continue

        dep = dep.split('\t')
        num = dep[0]
        word = dep[1]
        tag = dep[3]
        head = dep[6]
        dependency = dep[7]

        dep_dict[num] = {
            "word": word,
            "tag": tag,
            "next": head,
            "dep": dependency
        }

    return dep_dict


def get_relevant_set(data, i):
    relevant_set_nouns = defaultdict(list)
    relevant_set_verbs = defaultdict(list)

    for _, row in tqdm(data.iterrows(), file=sys.stdout, total=len(data), desc='All Users'):
        for ans in tqdm(row, file=sys.stdout, total=len(row), leave=False, desc='Questions'):
            if not ans or ans is pd.np.nan:  # some users didn't answer all of the questions
                continue

            ans = ans.replace('."', '".')
            ans = ans.replace(' ,', ',')
            ans = ans.replace(' .', '.')

            for sentence in ans.split('.'):
                if not sentence:
                    continue

                dep_tree = get_dependency_tree_for_sentence(sentence, i)

                for dependency in dep_tree.values():
                    if dependency['next'] == '0':
                        continue

                    curr_tag = dependency['tag']
                    curr_word = dependency['word']
                    next_tag = dep_tree[dependency['next']]['tag']
                    next_word = dep_tree[dependency['next']]['word']

                    if curr_tag in adjective_tags and next_tag in noun_tags:
                        relevant_set_nouns[next_word].append(curr_word)
                    elif curr_tag in adverb_tags and next_tag in verb_tags:
                        relevant_set_verbs[next_word].append(curr_word)

    with open(os.path.join(OUTPUTS_DIR, 'relevant_set_nouns_{}.json'.format(i)), 'w', encoding='utf-8') as out_file:
        json.dump(relevant_set_nouns, out_file, ensure_ascii=False)
    with open(os.path.join(OUTPUTS_DIR, 'relevant_set_verbs_{}.json'.format(i)), 'w', encoding='utf-8') as out_file:
        json.dump(relevant_set_verbs, out_file, ensure_ascii=False)


def get_reference_set(data, i):
    control_nouns = read_relevant_set('nouns', 'control')
    control_verbs = read_relevant_set('verbs', 'control')
    patients_nouns = read_relevant_set('nouns', 'patients')
    patients_verbs = read_relevant_set('verbs', 'patients')

    relevant_set_nouns = defaultdict(list)
    relevant_set_verbs = defaultdict(list)

    for article in tqdm(data, file=sys.stdout, total=len(data), desc='Articles'):
        with open(article, encoding='utf-8') as f:
            article = f.readlines()[:-1]  # last line is article metadata

        for sentence in tqdm(article, file=sys.stdout, total=len(article), leave=False, desc='Sentences'):
            if not sentence or sentence is pd.np.nan:  # some users didn't answer all of the questions
                continue

            sentence = sentence.replace('."', '".')
            sentence = sentence.replace(' ,', ',')
            sentence = sentence.replace(' .', '.')
            sentence = sentence.replace('prefix=', '')
            dep_tree = get_dependency_tree_for_sentence(sentence, i)

            for dependency in dep_tree.values():
                if dependency['next'] == '0':
                    continue

                curr_tag = dependency['tag']
                curr_word = dependency['word']
                next_tag = dep_tree[dependency['next']]['tag']
                next_word = dep_tree[dependency['next']]['word']

                relevant_noun = next_word in control_nouns or next_word in patients_nouns
                relevant_verb = next_word in control_verbs or next_word in patients_verbs

                if curr_tag in adjective_tags and next_tag in noun_tags and relevant_noun:
                    relevant_set_nouns[next_word].append(curr_word)
                elif curr_tag in adverb_tags and next_tag in verb_tags and relevant_verb:
                    relevant_set_verbs[next_word].append(curr_word)

    with open(os.path.join(OUTPUTS_DIR, 'reference_set_nouns_{}.json'.format(i)), 'w', encoding='utf-8') as out_file:
        json.dump(relevant_set_nouns, out_file, ensure_ascii=False)
    with open(os.path.join(OUTPUTS_DIR, 'reference_set_verbs_{}.json'.format(i)), 'w', encoding='utf-8') as out_file:
        json.dump(relevant_set_verbs, out_file, ensure_ascii=False)


if __name__ == '__main__':
    # slice data[i*(len//k):(i+1)*(len//k)]
    parser = optparse.OptionParser()
    parser.add_option('-i', action="store", type="int", help="job (block) index")
    parser.add_option('-k', action="store", type="int", help="jobs amount")
    parser.add_option('--set_type', action="store", choices=['relevant', 'reference'])
    options, remainder = parser.parse_args()

    block = options.i
    jobs = options.k

    if not os.path.exists(os.path.join(OUTPUTS_DIR, 'tmp')):
        os.makedirs(os.path.join(OUTPUTS_DIR, 'tmp'))

    start = lambda data: block * (len(data) // jobs)
    end = lambda data: (block + 1) * (len(data) // jobs)

    if options.set_type == 'relevant':
        df = pd.read_csv(os.path.join(DATA_DIR, 'all_data.csv'))
        df = df.iloc[:, 2:]  # ignore first 2 columns
        df = df.iloc[start(df):end(df)]

        get_relevant_set(df, block)
    elif options.set_type == 'reference':
        articles_path = os.path.join(DATA_DIR, 'doctors_articles')
        articles = os.listdir(articles_path)
        articles = [os.path.join(articles_path, article) for article in articles]
        articles = articles[start(articles):end(articles)]

        get_reference_set(articles, block)
