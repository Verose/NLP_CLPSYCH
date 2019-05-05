import json
import os
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt

from word_embeddings.common.utils import DATA_DIR, OUTPUTS_DIR


def get_rsdd_data():
    data = defaultdict(int)
    control_posts = []
    depression_posts = []
    control_words = []
    depression_words = []
    counter = 0

    with open(os.path.join('..', DATA_DIR, 'rsdd_posts', 'training'), 'r') as f:
        for line in f:
            item = json.loads(line)
            user_info = item[0]
            label = user_info['label']
            posts = user_info['posts']
            data[label] += 1
            if label == 'control':
                control_posts.append(len(posts))
            elif label == 'depression':
                depression_posts.append(len(posts))

            sum_words = 0
            for post in posts:
                sum_words += len(post[1].split())
            # average words per post
            if label == 'control':
                control_words += [sum_words/len(posts)]
            elif label == 'depression':
                depression_words += [sum_words/len(posts)]

            if counter % 5000 == 0:
                print("Processed {} users".format(counter))

            counter += 1

    print("Finished processing {} users".format(counter))
    print("data info: " + str(data))
    print("Control: ")
    print("Posts: avg: {}, max: {}".format(np.mean(control_posts), np.max(control_posts)))
    print("Words: avg: {}, max: {}".format(np.mean(control_words), np.max(control_words)))
    print("Depression: ")
    print("Posts: avg: {}, max: {}".format(np.mean(depression_posts), np.max(depression_posts)))
    print("Words: avg: {}, max: {}".format(np.mean(depression_words), np.max(depression_words)))

    plt.clf()
    posts_bins = [0, 100, 200, 500, 750, 1000, 1500, 1750, 2000, 5000, 8000, 11000]
    posts_labs = ['{}{}'.format(b/1000, 'k') for b in posts_bins[1:]]
    control_p_h = np.histogram(control_posts, bins=posts_bins)
    patients_p_h = np.histogram(depression_posts, bins=posts_bins)
    plt.bar(range(len(control_p_h[0])), control_p_h[0], width=1, label='control')
    plt.bar(range(len(patients_p_h[0])), patients_p_h[0], width=1, label='patients')
    plt.xticks(range(len(control_p_h[0])), labels=posts_labs)
    plt.yticks(range(0, 11001, 1000))
    plt.xlabel('#posts bins')
    plt.ylabel('#Users')
    plt.legend()
    plt.savefig(os.path.join('..', OUTPUTS_DIR, 'rsdd_posts_histogram.png'))

    plt.clf()
    words_bins = [0, 10, 25, 50, 75, 100, 125]
    words_labs = [str(b) for b in words_bins[1:]]
    control_w_h = np.histogram(control_words, bins=words_bins)
    patients_w_h = np.histogram(depression_words, bins=words_bins)
    plt.bar(range(len(control_w_h[0])), control_w_h[0], width=1, label='control')
    plt.bar(range(len(patients_w_h[0])), patients_w_h[0], width=1, label='patients')
    plt.xticks(range(len(control_w_h[0])), labels=words_labs)
    plt.yticks([0, 100, 500, 2500, 11000, 20000])
    plt.xlabel('words avg bins')
    plt.ylabel('#Users')
    plt.legend()
    plt.savefig(os.path.join('..', OUTPUTS_DIR, 'rsdd_words_histogram.png'))


if __name__ == "__main__":
    get_rsdd_data()
