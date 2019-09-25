import json
import optparse
import os
import re
import string
from multiprocessing.pool import Pool

import nltk

from common.utils import DATA_DIR

dataset_labels = {
    'rsdd': ['control', 'depression'],
    'smhd': ['control', 'schizophrenia']
}


def write_user_json(user_id):
    save_path = os.path.join('..', DATA_DIR, 'pos_tags_{}'.format(dataset), '{}.json'.format(user_id))

    if os.path.isfile(save_path):
        print("Skipping user {} file already exists".format(user_id))
        return

    user_data = None

    with open(reddit_file, 'r') as file:
        for i, row in enumerate(file, 1):
            if i == user_id:
                user_data = row
                break

    if not user_data:
        print("No data for user {} how is this possible?".format(user_id))
        return

    if dataset == 'rsdd':
        user_info = json.loads(user_data)[0]
        labels = [user_info['label']]
    else:
        user_info = json.loads(user_data)
        labels = user_info['label']
    posts = user_info['posts']

    true_label = None
    for label in labels:
        if label in dataset_labels[dataset]:
            true_label = label
            break

    if not true_label:
        return

    user_json = {
        "id": "{}".format(user_id),
        "label": true_label,
    }
    user_tokens = []
    user_pos_tags = []

    for post_stuff in posts:
        if dataset == 'rsdd':
            post = post_stuff[1]
        else:
            post = post_stuff['text']
        post = re.sub(r"http\S+", "", post)
        post = post.translate(str.maketrans('', '', string.punctuation))  # #$%&@
        tokens = nltk.word_tokenize(post)
        pos_tags = nltk.pos_tag(tokens)

        user_tokens.append(tokens)
        user_pos_tags.append([item[1] for item in pos_tags])
    user_json["tokens"] = user_tokens
    user_json["posTags"] = user_pos_tags

    with open(save_path, 'w') as file:
        json.dump(user_json, file)

    print("Finished with user {}".format(user_id))


def init(*args):
    """ store the counters for later use """
    global dataset
    global reddit_file
    dataset = args[0]
    reddit_file = args[1]


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--dataset', choices=['rsdd', 'smhd'], default='rsdd', action="store")
    parser.add_option('--n_processes', default=1, action="store")
    options, _ = parser.parse_args()

    dataset = options.dataset
    reddit_file = os.path.join('..', DATA_DIR, '{}_posts'.format(dataset), 'training')

    i = 0
    with open(reddit_file) as f:
        for i, l in enumerate(f):
            pass
    file_length = i + 1

    pool = Pool(
        processes=8,
        initializer=init,
        initargs=(dataset, reddit_file,)
    )
    results = pool.map(write_user_json, range(1, file_length))
    pool.close()
