import json
import optparse
import os
import sys

from tqdm import tqdm

"""
Script usage:
input: --reddit_file: reddit comments file to work on
output: --out_dir: for each user read in the file:
for a user <username>, creates or appends to end of <username>.csv:
<create_time_epoch>, "<subreddit>", "<comment>"
the --skip options allows to continue running from specific position
"""

if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--reddit_file', action="store", type=str)
    parser.add_option('--out_dir', action="store", type=str)
    parser.add_option('--skip', action="store", type=int, default=0)
    options, _ = parser.parse_args()

    with open(options.reddit_file, 'r') as out_file:
        reddit_file = out_file.readlines()[options.skip:]

    for i, user_data in tqdm(enumerate(reddit_file, 1),
                             file=sys.stdout,
                             total=len(reddit_file),
                             desc='Creating user files'):
        user_info = json.loads(user_data)
        user = user_info["author"]
        created_epoch = user_info["created_utc"]
        subreddit = user_info["subreddit"]
        post = user_info["body"]
        out_path = os.path.join(options.out_dir, '{user}.csv'.format(user=user))

        try:
            with open(out_path, "a+", encoding='utf-8') as out:
                location = out.tell()
                if location == 0:
                    out.write("epoch,subreddit,post\n")
                out.write("{},\"{}\",\"{}\"\n".format(created_epoch, subreddit, post))
        except UnicodeEncodeError as e:
            print("\nskipping line {line} because: {reason}".format(line=i + options.skip, reason=e.reason))
