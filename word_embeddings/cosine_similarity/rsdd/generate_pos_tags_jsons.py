import json
import os
import string
import sys

import nltk
from tqdm import tqdm

from word_embeddings.common.utils import DATA_DIR


skip = 3608
# skip = 0


with open(os.path.join('..', DATA_DIR, 'rsdd_posts', 'training'), 'r') as out_file:
    rsdd = out_file.readlines()[skip:]


for i, user_data in tqdm(enumerate(rsdd, 1), file=sys.stdout, total=len(rsdd), desc='RSDD training users'):
    user_info = json.loads(user_data)[0]
    label = user_info['label']
    posts = user_info['posts']
    user_id = i + skip

    if label not in ['control', 'depression']:
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

    with open(os.path.join('..', DATA_DIR, 'pos_tags_rsdd', '{}.json'.format(user_id)), 'w') as out_file:
        json.dump(user_json, out_file)
