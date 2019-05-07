import json
import os
import sys

import pandas as pd
from tqdm import tqdm

from word_embeddings.common.utils import DATA_DIR

df = pd.DataFrame(columns=['id', 'label'])


with open(os.path.join('..', DATA_DIR, 'rsdd_posts', 'training'), 'r') as out_file:
    rsdd = out_file.readlines()


for user_id, user_data in tqdm(enumerate(rsdd, 1), file=sys.stdout, total=len(rsdd), desc='RSDD training users'):
    user_info = json.loads(user_data)[0]
    label = user_info['label']

    if label not in ['control', 'depression']:
        continue

    df = df.append({'id': user_id, 'label': label}, ignore_index=True)


df.to_csv(os.path.join('..', DATA_DIR, 'all_data_rsdd.csv'), index=False)
