import os
import sys

import pandas as pd
from tqdm import tqdm

from common.utils import DATA_DIR

data = pd.read_csv(os.path.join('..', DATA_DIR, 'all_data_rsdd.csv'))
data = data.head(10_000)
controls = 0
depressed = 0


for _, data in tqdm(data.iterrows(), file=sys.stdout, total=len(data), leave=False, desc='Users'):
    user_id = data['id']
    label = data['label']

    if label == 'control':
        controls += 1
    elif label == 'depression':
        depressed += 1

print('There are {} controls and {} depressed'.format(controls, depressed))
