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
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

from sklearn.cluster import DBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from word_embeddings.common.utils import DATA_DIR

if __name__ == "__main__":
    print("*******Starting to run!*******")
    json_pattern = os.path.join('..', DATA_DIR, 'pos_tags_rsdd_embeds', '*.json')
    json_files = [pos_json for pos_json in glob.glob(json_pattern) if pos_json.endswith('.json')]
    # index_to_data = {}
    # index = 0
    all_posts = []
    all_embeddings = []
    eps = 5
    min_samples = 10

    print("Using eps={}, min_samples={}".format(eps, min_samples))

    for i, file in enumerate(json_files):
        user_id = os.path.basename(file).split('.')[0]
        with open(file, encoding='utf-8') as f:
            pos_tags = json.load(f)
            posts_list = pos_tags['embeddings']
            label = pos_tags['label']

            # for _ in range(len(posts_list)):
            #     index += 1
            #     index_to_data[index] = (user_id, label)
            all_embeddings.extend(posts_list)
        if i % 150 == 0:
            print("Finished loading {} users".format(i))
        if i == 6:
            break

    print("*******Finished loading all of the vectors*******")

    # standardize the data
    X = StandardScaler().fit_transform(all_embeddings)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    print("*******Fitting the data*******")
    labels = dbscan.fit_predict(X)
    print("*******Finished fitting the data*******")

    # identifying the core samples
    # core_samples_mask = np.zeros_like(labels, dtype=bool)
    # core_samples_mask[dbscan.core_sample_indices_] = True

    # declare the number of clusters
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    # if n_clusters > 0:
    #     silhouette_score = metrics.silhouette_score(X, labels)
    #
    #     print('Estimated number of clusters: %d' % n_clusters)
    #     print('Estimated number of noise points: %d' % n_noise)
    #
    #     # plot the cluster assignments
    #     plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="plasma")
    #     plt.xlabel("Feature 0")
    #     plt.ylabel("Feature 1")
    #     plt.show()

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors_map = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    #
    # for k, color in zip(unique_labels, colors):
    #     if k == -1:
    #         # Black used for noise.
    #         color = [0, 0, 0, 1]
    #
    #     class_member_mask = (labels == k)
    #
    #     xy = X[class_member_mask & core_samples_mask]
    #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(color), markeredgecolor='k', markersize=14)
    #
    #     xy = X[class_member_mask & ~core_samples_mask]
    #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(color), markeredgecolor='k', markersize=6)
    # plt.xlabel("Feature 0")
    # plt.ylabel("Feature 1")
    #
    # plt.title('Estimated number of clusters: %d' % n_clusters)
    # plt.show()

    # Show dbscan results as SVD
    svd = TruncatedSVD(n_components=2).fit_transform(X)
    # Set the colour of noise pts to black
    # for i in range(0, len(labels)):
    #     if labels[i] == -1:
    #         labels[i] = 7
    # colors = [LABELS[l] for l in labels]
    # colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(set(labels)))]
    colors = [colors_map[l] for l in labels]
    plt.scatter(svd[:, 0], svd[:, 1], c=colors, s=0.5, linewidths=0.5, alpha=0.7)
    plt.savefig('dbscan_svd_{}_{}.png'.format(eps, min_samples))

    print('Estimated number of clusters: %d' % n_clusters)
    print('Estimated number of noise points: %d' % n_noise)
    silhouette_score = metrics.silhouette_score(X, labels)
    print("Silhouette Coefficient: %0.3f" % silhouette_score)
