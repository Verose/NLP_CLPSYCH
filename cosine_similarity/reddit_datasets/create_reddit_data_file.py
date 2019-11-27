import glob
import json
import optparse
import os
import sys

import pandas as pd
from tqdm import tqdm

from common.utils import DATA_DIR


dataset_labels = {
    'rsdd': ['control', 'depression'],
    'smhd': ['control', 'schizophrenia']
}


def handle_user_info(user_id, labels, num_posts, df):
    if min_posts and num_posts < min_posts:
        # print('Skipping user {} with less than {} posts'.format(user_id, min_posts))
        return df
    true_label = None
    for label in labels:
        if label in dataset_labels[dataset]:
            true_label = label
            break
    if not true_label:
        return df

    df = df.append({'id': user_id, 'label': true_label}, ignore_index=True)
    return df


if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option('--folder', action="store", default='')
    parser.add_option('--dataset', choices=['rsdd', 'smhd'], default='rsdd', action="store")
    parser.add_option('--min_posts', default=None, type=int, action="store")
    options, _ = parser.parse_args()

    dataset = options.dataset
    min_posts = options.min_posts
    df = pd.DataFrame(columns=['id', 'label'])

    if not options.folder:
        with open(os.path.join('..', DATA_DIR, '{}_posts'.format(dataset), 'training'), 'r') as out_file:
            reddit = out_file.readlines()

        for user_id, user_data in tqdm(enumerate(reddit, 1), file=sys.stdout, total=len(reddit),
                                       desc='{} training users'.format(dataset.upper())):
            if dataset == 'rsdd':
                user_info = json.loads(user_data)[0]
                labels = [user_info['label']]
            else:
                user_info = json.loads(user_data)
                labels = user_info['label']
            df = handle_user_info(user_id, labels, len(user_info['posts']), df)
        df.to_csv(os.path.join('..', DATA_DIR, 'all_data_{}.csv'.format(dataset)), index=False)
    else:
        json_pattern = os.path.join('..', DATA_DIR, 'pos_tags_{}_embeds_filtered'.format(dataset), options.folder,
                                    '*.json')
        json_files = [pos_json for pos_json in glob.glob(json_pattern)]

        for jfile in tqdm(json_files, file=sys.stdout, total=len(json_files),
                          desc='{} filtered users'.format(dataset.upper())):
            user_id = os.path.basename(jfile).split('.')[0]
            with open(jfile) as f:
                user_info = json.load(f)
                labels = [user_info['label']]
                df = handle_user_info(user_id, labels, len(user_info['tokens']), df)
        filename = 'all_data_{}_{}{}.csv'.format(dataset, options.folder, '_' + str(min_posts) if min_posts else '')
        df.to_csv(os.path.join('..', DATA_DIR, filename), index=False)
