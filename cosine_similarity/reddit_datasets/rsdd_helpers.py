"""
1. convert rsdd posts using bert
   use pos tags or original sentences?
   save vectors to new file?
2. use dbscan on bert vectors
   know how to know which post came from which user
3. visualize?
"""
import glob
import json
import os

import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from common.utils import DATA_DIR


def count_all_posts():
    json_pattern = os.path.join('..', DATA_DIR, 'pos_tags_rsdd_embeds', '*.json')
    json_files = [pos_json for pos_json in glob.glob(json_pattern) if pos_json.endswith('.json')]
    sum_all_posts = 0

    for jfile in json_files:
        with open(jfile, encoding='utf-8') as f:
            pos_tags = json.load(f)
            posts_list = pos_tags['tokens']  # each post is a list of tokens
            sum_all_posts += len(posts_list)

    print('Overall posts num: {}.json\n'.format(sum_all_posts))


def count_users_with_embeddings():
    json_pattern = os.path.join('..', DATA_DIR, 'pos_tags_rsdd_embeds', '*.json')
    json_files = [pos_json for pos_json in glob.glob(json_pattern) if pos_json.endswith('.json')]
    sum_controls = 0
    sum_depression = 0

    for jfile in json_files:
        with open(jfile, encoding='utf-8') as f:
            pos_tags = json.load(f)
            label = pos_tags['label']  # each post is a list of tokens

            if label == 'control':
                sum_controls += 1
            elif label == 'depression':
                sum_depression += 1
            else:
                print("Unidentified label: {}".format(label))

    print('Overall controls num: {} and depression num: {}'.format(sum_controls, sum_depression))


def print_svd():
    print("*******Starting to run!*******")
    json_pattern = os.path.join('..', DATA_DIR, 'pos_tags_rsdd_embeds', '*.json')
    json_files = [pos_json for pos_json in glob.glob(json_pattern) if pos_json.endswith('.json')]
    all_embeddings = []

    for i, file in enumerate(json_files):
        with open(file, encoding='utf-8') as f:
            pos_tags = json.load(f)
            posts_list = pos_tags['embeddings']
            all_embeddings.extend(posts_list)
        if i % 150 == 0:
            print("Finished loading {} users".format(i))
        if i == 6:
            break

    print("*******Finished loading all of the vectors*******")

    # standardize the data
    X = StandardScaler().fit_transform(all_embeddings)
    svd = TruncatedSVD(n_components=2).fit_transform(X)
    # colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, X.shape[0])]
    plt.scatter(svd[:, 0], svd[:, 1], s=0.5, linewidths=0.5, alpha=0.7)
    plt.savefig('dbscan_svd.png')


if __name__ == "__main__":
    # count_all_posts()
    # count_users_with_embeddings()
    print_svd()
