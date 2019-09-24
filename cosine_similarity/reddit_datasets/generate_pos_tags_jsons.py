import json
import optparse
import os
import string
import sys

import nltk
from tqdm import tqdm

from common.utils import DATA_DIR


dataset_labels = {
    'rsdd': ['control', 'depression'],
    'smhd': ['control', 'schizophrenia']
}


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--dataset', choices=['rsdd', 'smhd'], default='rsdd', action="store")
    parser.add_option('--skip', default=0, action="store")
    options, = parser.parse_args()

    skip = options.skip
    dataset = options.dataset

    with open(os.path.join('..', DATA_DIR, '{}_posts'.format(dataset), 'training'), 'r') as file:
        reddit = file.readlines()[skip:]

    for i, user_data in tqdm(enumerate(reddit, 1), file=sys.stdout, total=len(reddit),
                             desc='{} training users'.format(dataset.upper())):
        user_info = json.loads(user_data)[0]
        label = user_info['label']
        posts = user_info['posts']
        user_id = i + skip

        if label not in dataset_labels[dataset]:
            continue

        user_json = {
            "id": "{}".format(user_id),
            "label": label,
        }
        user_tokens = []
        user_pos_tags = []

        for post_stuff in posts:
            post = post_stuff[1]
            post = post.translate(str.maketrans('', '', string.punctuation))  # #$%&@
            tokens = nltk.word_tokenize(post)
            pos_tags = nltk.pos_tag(tokens)

            user_tokens.append(tokens)
            user_pos_tags.append([item[1] for item in pos_tags])
        user_json["tokens"] = user_tokens
        user_json["posTags"] = user_pos_tags

        with open(
                os.path.join('..', DATA_DIR, 'pos_tags_{}'.format(dataset), '{}.json'.format(user_id)), 'w') as file:
            json.dump(user_json, file)
