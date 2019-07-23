import glob
import json
import optparse
import os
from collections import Counter
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from word_embeddings.common.utils import DATA_DIR

"""
1. convert rsdd posts to flair embeddings
   save embeddings to new file
2. use dbscan on subset of embeddings
   know how to convert back from cluster points to posts
3. visualize?
"""


def load_embeddings():
    index = 0
    embeddings = []

    for i, file in enumerate(json_files):
        user_id = os.path.basename(file).split('.')[0]

        with open(file, encoding='utf-8') as f:
            data = json.load(f)
            posts_embeddings = data['embeddings']

            for post_ind in range(len(posts_embeddings)):
                emb_ind_to_user_n_post_ind[index] = (user_id, post_ind)
                index += 1
            embeddings.extend(posts_embeddings)
        if i % 150 == 0:
            print("Finished loading {} users".format(i))
    print("*******Finished loading all of the vectors*******")

    return embeddings


def create_filtered_jsons(points_it, embeds_dir, out_dir):
    print("*******Creating filtered jsons*******")
    user_post_inds = []
    user = ''
    prev_user = ''
    is_first = True
    print("Finished writing users: ", end='')

    for _, emb_ind in points_it:
        user, point_ind = emb_ind_to_user_n_post_ind[emb_ind]
        if is_first:
            prev_user = user
            is_first = False

        if user == prev_user:
            user_post_inds.append(point_ind)
        else:
            # create jsons for previous user
            write_filtered_jsons(prev_user, user_post_inds, embeds_dir, out_dir)
            prev_user = user
            user_post_inds = [point_ind]
    write_filtered_jsons(user, user_post_inds, embeds_dir, out_dir)


def write_filtered_jsons(user, inds, embeds_dir, out_path):
    with open(os.path.join(embeds_dir, '{}.json').format(user),
              encoding='utf-8') as f:
        user_data = {}
        data = json.load(f)
        user_data['label'] = data['label']
        slicer = itemgetter(*inds)
        user_data['tokens'] = slicer(data['tokens'])
        user_data['posTags'] = slicer(data['posTags'])
        user_data['embeddings'] = slicer(data['embeddings'])

        with open(os.path.join(out_path, '{}.json'.format(user)), 'w') as out_file:
            json.dump(user_data, out_file)
    print("{}, ".format(user), end='')


if __name__ == "__main__":
    print("*******Starting to run!*******")
    parser = optparse.OptionParser()
    parser.add_option('--eps', action="store", type=int)
    parser.add_option('--min_samples', action="store", type=int)
    parser.add_option('--output', action="store", type=str, default="")
    options, _ = parser.parse_args()

    eps = options.eps
    min_samples = options.min_samples
    print("Using eps={}, min_samples={}".format(eps, min_samples))

    embeddings_dir = os.path.join('..', DATA_DIR, 'pos_tags_rsdd_embeds')
    json_pattern = os.path.join(embeddings_dir, '*.json')
    json_files = [pos_json for pos_json in glob.glob(json_pattern) if pos_json.endswith('.json')]
    emb_ind_to_user_n_post_ind = {}

    all_embeddings = load_embeddings()

    # standardize the data
    X = StandardScaler().fit_transform(all_embeddings)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    print("*******Fitting the data*******")
    labels = dbscan.fit_predict(X)
    print("*******Finished fitting the data*******")

    # declare the number of clusters
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors_map = [plt.cm.Spectral(num) for num in np.linspace(0, 1, len(unique_labels))]

    # Show dbscan results as SVD
    svd = TruncatedSVD(n_components=2).fit_transform(X)
    colors = [colors_map[l] for l in labels]
    plt.scatter(svd[:, 0], svd[:, 1], c=colors, s=0.5, linewidths=0.5, alpha=0.7)
    plt.savefig('dbscan_svd_{}_{}.png'.format(eps, min_samples))

    print('Estimated number of clusters: %d' % n_clusters)
    print('Estimated number of noise points: %d' % n_noise)
    silhouette_score = metrics.silhouette_score(X, labels) if n_clusters > 0 else -2
    print("Silhouette Coefficient: %0.3f" % silhouette_score)

    # find largest cluster
    counter = Counter(labels)

    for label, points in counter.items():
        print("Label {} has {} points".format(label, points))

    largest_cluster = counter.most_common()[0]
    print("The largest cluster is {} with {} points".format(largest_cluster[0], largest_cluster[1]))

    # convert back from points of the largest cluster to posts
    largest_cluster_points = np.where(labels == largest_cluster[0])[0]
    points_iter = np.ndenumerate(largest_cluster_points)

    output_dir = os.path.join('..', DATA_DIR, 'pos_tags_rsdd_embeds_filtered', options.output)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    create_filtered_jsons(points_iter, embeddings_dir, output_dir)
