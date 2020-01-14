import glob
import json
import logging
import optparse
import os
from multiprocessing.pool import Pool

from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings, Sentence, FlairEmbeddings

from common.utils import DATA_DIR

# avoid "WARNING this function is deprecated, use smart_open.open instead"
logging.getLogger('smart_open').setLevel(logging.ERROR)


def init(*args):
    """ store the dataset for later use """
    global dataset
    dataset = args[0]


def save_json(jfile):
    global dataset

    user_id = os.path.basename(jfile).split('.')[0]
    save_path = os.path.join('..', DATA_DIR, 'pos_tags_{}_embeds_nolimit'.format(dataset), '{}.json'.format(user_id))

    if os.path.isfile(save_path):
        print("Skipping user {} file already exists".format(user_id))
        return

    document_embeddings = get_doc_embeddings()

    with open(jfile, encoding='utf-8') as f:
        user_data = json.load(f)

        if len(user_data['tokens']) > 500:
            print('User {}.json has {} posts NOT skipping'.format(user_id, len(user_data['tokens'])))
            # return

        posts_list = user_data['tokens']  # each post is a list of tokens
        pos_tags_list = user_data['posTags']
        posts_lowercase_list = []
        posts_embeddings_list = []
        pos_tags_list_lowercase = []

        for i, (post, pos_tags) in enumerate(zip(posts_list, pos_tags_list)):
            post_lowercase = [token.lower() for token in post]
            if any("http" in word for word in post_lowercase):
                continue
            posts_lowercase_list.append(post_lowercase)
            if 0 < len(post_lowercase):
                pos_tags_list_lowercase.append(pos_tags)
                post_sentence = Sentence(' '.join([post for post in post_lowercase]))
                document_embeddings.embed(post_sentence)
                posts_embeddings_list.append(post_sentence.get_embedding().tolist())
            elif len(post_lowercase) > 100:
                print('long post')
            else:
                continue

        user_data["tokens"] = posts_lowercase_list
        user_data["posTags"] = pos_tags_list_lowercase
        user_data["embeddings"] = posts_embeddings_list

        with open(save_path, 'w') as out_file:
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
    parser = optparse.OptionParser()
    parser.add_option('--dataset', choices=['rsdd', 'smhd'], default='rsdd', action="store")
    parser.add_option('--n_processes', default=2, type=int, action="store")
    options, _ = parser.parse_args()

    dataset = options.dataset
    json_pattern = os.path.join('..', DATA_DIR, 'pos_tags_{}'.format(dataset), '*.json')
    json_files = [pos_json for pos_json in glob.glob(json_pattern) if pos_json.endswith('.json')]

    pool = Pool(
        processes=options.n_processes,
        initializer=init,
        initargs=(dataset,)
    )
    results = pool.map(save_json, json_files)
    pool.close()
