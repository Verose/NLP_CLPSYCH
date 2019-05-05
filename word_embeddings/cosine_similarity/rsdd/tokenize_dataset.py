import json
import os
import string

from nltk.tokenize import word_tokenize

from word_embeddings.common.utils import DATA_DIR

vocab = []
with open(os.path.join('..', DATA_DIR, 'rsdd_posts', 'training'), 'r') as f:
    tmp_vocab = []
    lines_read = 0

    for line in f:
        item = json.loads(line)
        user_info = item[0]
        posts = user_info['posts']
        text1 = "It's true that the chicken was the best bamboozler in the known multiverse."

        for post_stuff in posts:
            post = post_stuff[1]
            post = post.translate(str.maketrans('', '', string.punctuation))  # #$%&@
            tokens = word_tokenize(post)
            tmp_vocab.extend(tokens)
        vocab.extend(set(tmp_vocab))
        lines_read += 1

        if lines_read % 5 == 0:
            vocab = list(set(vocab))

with open(os.path.join('..', DATA_DIR, 'rsdd_vocab.txt'), 'w') as f:
    for word in vocab:
        f.write("%s\n" % word)
