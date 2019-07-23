import json
import os

import pandas as pd

from word_embeddings.common.utils import DATA_DIR

if __name__ == "__main__":
    with open(os.path.join(DATA_DIR, "RSSD", "subreddits.json")) as f:
        for line in f:
            data = json.loads(line)
            pass
    # df = pd.read_csv(os.path.join(DATA_DIR, "RSSD", "69M_reddit_accounts.csv"))
    # print(df.head())
