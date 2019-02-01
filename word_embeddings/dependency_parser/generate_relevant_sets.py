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
punctuation = string.punctuation.replace('-', '')


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


def get_dependency_tree_for_sentence(sent, i, dataset=None):
    sent = sent.translate(str.maketrans("", "", punctuation))
    dataset = dataset.replace('/', '_') if dataset else None  # support nested folders
    unique_name = '{}'.format(i) if not dataset else '{}_{}'.format(i, dataset)

    while '  ' in sent:  # fix sentences
        sent = sent.replace('  ', ' ')

    with open(
            os.path.join(OUTPUTS_DIR, 'tmp', 'sentence_{}.txt'.format(unique_name)),
            'w',
            encoding='utf-8',
            newline='\n'
    ) as f:
        [f.write(u'{}\n'.format(word)) for word in sent.split(' ')]
        f.write('\n')
    go_workspace = os.environ['GOPATH']
    yap_path = os.path.join(go_workspace, 'src', 'yap')
    yap = os.path.join(yap_path, 'yap.exe' if os.name == 'nt' else 'yap')

    hebma_cmd = [
        yap,
        "hebma",
        "-raw",
        os.path.join(OUTPUTS_DIR, 'tmp', 'sentence_{}.txt'.format(unique_name)),
        "-out",
        os.path.join(OUTPUTS_DIR, 'tmp', "output_{}.txt".format(unique_name))
    ]
    p = subprocess.Popen(hebma_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    p.communicate()

    joint_cmd = [
        yap,
        "joint",
        "-f",
        os.path.join(yap_path, 'conf', "jointzeager.yaml"),
        "-in",
        os.path.join(OUTPUTS_DIR, 'tmp', "output_{}.txt".format(unique_name)),
        "-l",
        os.path.join(yap_path, 'conf', "hebtb.labels.conf"),
        "-m",
        os.path.join(yap_path, 'data', "joint_arc_zeager_model_temp_i33.b64"),
        "-nolemma",
        "-oc",
        os.path.join(OUTPUTS_DIR, 'tmp', "output_dep_{}.txt".format(unique_name)),
        "-om",
        os.path.join(OUTPUTS_DIR, 'tmp', "output_om_{}.txt".format(unique_name)),
        "-os",
        os.path.join(OUTPUTS_DIR, 'tmp', "output_os_{}.txt".format(unique_name)),
        "-jointstr",
        "ArcGreedy",
        "-oraclestr",
        "ArcGreedy"
    ]
    p = subprocess.Popen(joint_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    p.communicate()

    with open(
            os.path.join(OUTPUTS_DIR, 'tmp', 'output_dep_{}.txt'.format('{}'.format(unique_name))),
            encoding='utf-8'
    ) as f:
        dep_tree = f.read()

    dep_tree = dep_tree.split('\t_\t_\n')  # every line ends with \t_\t_\n
    dep_dict = {}

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


def get_relevant_set(data, i):
    relevant_set_nouns = defaultdict(list)
    relevant_set_verbs = defaultdict(list)

    for _, row in tqdm(data.iterrows(), file=sys.stdout, total=len(data), desc='All Users'):
        for ans in tqdm(row, file=sys.stdout, total=len(row), leave=False, desc='Questions'):
            if not ans or ans is pd.np.nan:  # some users didn't answer all of the questions
                continue

            ans = repair_document(ans)

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


def get_reference_set(data, i, dataset):
    control_nouns = read_relevant_set('nouns', 'control')
    control_verbs = read_relevant_set('verbs', 'control')
    patients_nouns = read_relevant_set('nouns', 'patients')
    patients_verbs = read_relevant_set('verbs', 'patients')

    reference_set_nouns = defaultdict(list)
    reference_set_verbs = defaultdict(list)

    for article in tqdm(data, file=sys.stdout, total=len(data), desc='Articles'):
        with open(article, encoding='utf-8') as f:
            article = f.readlines()

        for sentence in tqdm(article, file=sys.stdout, total=len(article), leave=False, desc='Sentences'):
            if not sentence or '•' in sentence:  # skip empty sentences and lists
                continue

            sentence = repair_document(sentence)
            dep_tree = get_dependency_tree_for_sentence(sentence, i, dataset)

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
                    reference_set_nouns[next_word].append(curr_word)
                elif curr_tag in adverb_tags and next_tag in verb_tags and relevant_verb:
                    reference_set_verbs[next_word].append(curr_word)

    with open(
            os.path.join(OUTPUTS_DIR, 'reference_set_nouns_{}_{}.json'.format(i, dataset)),
            'w',
            encoding='utf-8'
    ) as out_file:
        json.dump(reference_set_nouns, out_file, ensure_ascii=False)
    with open(
            os.path.join(OUTPUTS_DIR, 'reference_set_verbs_{}_{}.json'.format(i, dataset)),
            'w',
            encoding='utf-8'
    ) as out_file:
        json.dump(reference_set_verbs, out_file, ensure_ascii=False)


if __name__ == '__main__':
    # slice data[i*(len//k):(i+1)*(len//k)]
    parser = optparse.OptionParser()
    parser.add_option('-i', action="store", type="int", help="job (block) index")
    parser.add_option('-k', action="store", type="int", help="jobs amount")
    parser.add_option('--set_type', action="store", choices=['relevant', 'reference'])
    parser.add_option('--folder', action="store", default="doctors_articles")
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
        articles_path = os.path.join(DATA_DIR, options.folder)
        articles = os.listdir(articles_path)
        articles = [os.path.join(articles_path, article) for article in articles]
        articles = articles[start(articles):end(articles)]

        get_reference_set(articles, block, options.folder)
