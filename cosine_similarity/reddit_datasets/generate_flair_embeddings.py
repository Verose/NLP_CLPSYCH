import glob
import json
import logging
import os
from multiprocessing import Value
from multiprocessing.pool import Pool

from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings, Sentence, FlairEmbeddings

from common.utils import DATA_DIR

# avoid "WARNING this function is deprecated, use smart_open.open instead"
logging.getLogger('smart_open').setLevel(logging.ERROR)


THRESHOLD = 500
controls_counter = None
depressions_counter = None


def init(*args):
    """ store the counters for later use """
    global controls_counter
    global depressions_counter
    controls_counter = args[0]
    depressions_counter = args[1]


def save_json(jfile):
    global controls_counter
    global depressions_counter

    user_id = os.path.basename(jfile).split('.')[0]
    document_embeddings = get_doc_embeddings()

    with open(jfile, encoding='utf-8') as f:
        user_data = json.load(f)
        label = user_data['label']

        if label == 'control' and controls_counter.value >= THRESHOLD:
            return

        if len(user_data['tokens']) > 500:
            print('Skipping file {}.json'.format(user_id))
            return

        if label == 'control':
            with controls_counter.get_lock():
                controls_counter.value += 1
        else:
            with depressions_counter.get_lock():
                depressions_counter.value += 1
        posts_list = user_data['tokens']  # each post is a list of tokens
        pos_tags_list = user_data['posTags']
        posts_lowercase_list = []
        posts_embeddings_list = []
        pos_tags_list_lowercase = []

        for i, (post, pos_tags) in enumerate(zip(posts_list, pos_tags_list)):
            post_lowercase = [token.lower() for token in post]
            if 5 <= len(post_lowercase) <= 40:
                posts_lowercase_list.append(post_lowercase)
                pos_tags_list_lowercase.append(pos_tags)
                post_sentence = Sentence(' '.join([post for post in post_lowercase]))
                document_embeddings.embed(post_sentence)
                posts_embeddings_list.append(post_sentence.get_embedding().tolist())
            else:
                continue

        user_data["tokens"] = posts_lowercase_list
        user_data["posTags"] = pos_tags_list_lowercase
        user_data["embeddings"] = posts_embeddings_list

        with open(os.path.join('..', DATA_DIR, 'pos_tags_rsdd_embeds', '{}.json'.format(user_id)), 'w') as out_file:
            json.dump(user_data, out_file)
        print('Finished with file {}.json'.format(user_id))


def get_doc_embeddings():
    # initialize the word embeddings
    glove_embedding = WordEmbeddings('glove')
    flair_embedding_forward = FlairEmbeddings('news-forward-fast')
    flair_embedding_backward = FlairEmbeddings('news-backward-fast')

    # initialize the document embeddings, mode = mean
    return DocumentPoolEmbeddings(
        [glove_embedding, flair_embedding_backward, flair_embedding_forward],
        fine_tune_mode='none'
    )


if __name__ == "__main__":
    json_pattern = os.path.join('..', DATA_DIR, 'pos_tags_rsdd', '*.json')
    json_files = [pos_json for pos_json in glob.glob(json_pattern) if pos_json.endswith('.json')]

    controls_counter = Value('i', 0)
    depressions_counter = Value('i', 0)

    pool = Pool(
        processes=1,
        initializer=init,
        initargs=(controls_counter, depressions_counter, )
    )
    results = pool.map(save_json, json_files)
    pool.close()
