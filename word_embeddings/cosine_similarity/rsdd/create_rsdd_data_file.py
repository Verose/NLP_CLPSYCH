import glob
import json
import optparse
import os
import sys

import pandas as pd
from tqdm import tqdm

from word_embeddings.common.utils import DATA_DIR


def handle_user_info(user_info, df):
    label = user_info['label']

    if label not in ['control', 'depression']:
        return

    df = df.append({'id': user_id, 'label': label}, ignore_index=True)
    return df


if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option('--folder', action="store", default='')
    options, remainder = parser.parse_args()

    df = pd.DataFrame(columns=['id', 'label'])

    if not options.folder:
        with open(os.path.join('..', DATA_DIR, 'rsdd_posts', 'training'), 'r') as out_file:
            rsdd = out_file.readlines()

        for user_id, user_data in tqdm(
                enumerate(rsdd, 1), file=sys.stdout, total=len(rsdd), desc='RSDD training users'):
            user_info = json.loads(user_data)[0]
            df = handle_user_info(user_info, df)
        df.to_csv(os.path.join('..', DATA_DIR, 'all_data_rsdd.csv'), index=False)
    else:
        json_pattern = os.path.join('..', DATA_DIR, 'pos_tags_rsdd_embeds_filtered', options.folder, '*.json')
        json_files = [pos_json for pos_json in glob.glob(json_pattern)]

        for jfile in tqdm(json_files, file=sys.stdout, total=len(json_files), desc='RSDD filtered users'):
            user_id = os.path.basename(jfile).split('.')[0]
            with open(jfile) as f:
                user_info = json.load(f)
                df = handle_user_info(user_info, df)
        df.to_csv(os.path.join('..', DATA_DIR, 'all_data_rsdd_{}.csv'.format(options.folder)), index=False)
