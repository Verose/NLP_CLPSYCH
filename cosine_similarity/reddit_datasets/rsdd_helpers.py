import glob
import json
import optparse
import os

import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from common.utils import DATA_DIR


def count_min_posts(folder, min_posts):
    json_pattern = os.path.join('..', DATA_DIR, 'pos_tags_{}_embeds_filtered'.format(dataset), folder, '*.json')
    json_files = [pos_json for pos_json in glob.glob(json_pattern) if pos_json.endswith('.json')]
    sum_controls = 0
    sum_depression = 0
    for jfile in json_files:
        with open(jfile, encoding='utf-8') as f:
            pos_tags = json.load(f)
            label = pos_tags['label']
            if len(pos_tags['tokens']) < min_posts:
                continue
            if label == 'control':
                sum_controls += 1
            elif label == 'schizophrenia':
                sum_depression += 1
    print('Overall controls num: {} and depression num: {}'.format(sum_controls, sum_depression))


def count_all_posts():
    json_pattern = os.path.join('..', DATA_DIR, 'pos_tags_{}_embeds'.format(dataset), '*.json')
    json_files = [pos_json for pos_json in glob.glob(json_pattern) if pos_json.endswith('.json')]
    sum_all_posts = 0

    for jfile in json_files:
        with open(jfile, encoding='utf-8') as f:
            pos_tags = json.load(f)
            posts_list = pos_tags['tokens']  # each post is a list of tokens
            sum_all_posts += len(posts_list)

    print('Overall posts num: {}\n'.format(sum_all_posts))


def count_users_with_embeddings():
    json_pattern = os.path.join('..', DATA_DIR, 'pos_tags_{}_embeds'.format(dataset), '*.json')
    json_files = [pos_json for pos_json in glob.glob(json_pattern) if pos_json.endswith('.json')]
    sum_controls = 0
    sum_patients = 0

    for jfile in json_files:
        with open(jfile, encoding='utf-8') as f:
            pos_tags = json.load(f)
            label = pos_tags['label']  # each post is a list of tokens

            if label == 'control':
                sum_controls += 1
            elif label == patients_label:
                sum_patients += 1
            else:
                print("Unidentified label: {}".format(label))

    print('Overall controls num: {} and patients num: {}'.format(sum_controls, sum_patients))


def print_svd():
    print("*******Starting to run!*******")
    json_pattern = os.path.join('..', DATA_DIR, 'pos_tags_{}_embeds'.format(dataset), '*.json')
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


def count_participants_datafile():
    from tqdm import tqdm
    import pandas as pd
    import sys

    data = pd.read_csv(os.path.join('..', DATA_DIR, 'all_data_{}.csv'.format(dataset)))
    # data = data.head(10_000)
    controls = 0
    patients = 0

    for _, data in tqdm(data.iterrows(), file=sys.stdout, total=len(data), leave=False, desc='Users'):
        user_id = data['id']
        label = data['label']

        if label == 'control':
            controls += 1
        elif label == patients_label:
            patients += 1

    print('There are {} controls and {} patients in {}'.format(controls, patients, options.dataset))


if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option('--dataset', choices=['rsdd', 'smhd', 'tssd'], default='rsdd', action="store")
    parser.add_option('--svd', default=False, action="store_true")
    parser.add_option('--count_posts', default=False, action="store_true")
    parser.add_option('--count_users', default=False, action="store_true")
    parser.add_option('--count_datafile', default=False, action="store_true")
    parser.add_option('--count_min_posts', default=False, action="store_true")
    parser.add_option('--folder', default='45_100', action="store")
    parser.add_option('--min_posts', default=20, action="store")
    options, _ = parser.parse_args()
    dataset = options.dataset
    patients_label = 'depression' if options.dataset == 'rsdd' else 'schizophrenia'

    if options.svd:
        print_svd()
    if options.count_posts:
        count_all_posts()
    if options.count_users:
        count_users_with_embeddings()
    if options.count_datafile:
        count_participants_datafile()
    if options.count_min_posts:
        count_min_posts(options.folder, options.min_posts)
