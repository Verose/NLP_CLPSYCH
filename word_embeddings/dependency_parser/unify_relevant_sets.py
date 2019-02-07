import glob
import json
import optparse
import os
from collections import defaultdict, Counter

DATA_DIR = os.path.join('..', 'data')


def relevant():
    for group in ['control', 'patients']:
        for pos_tag_to_test in ['nouns', 'verbs']:
            json_pattern = os.path.join(DATA_DIR, 'relevant_sets', group, 'relevant_set_{}*.json'.format(
                pos_tag_to_test))
            json_files = [pos_json for pos_json in glob.glob(json_pattern) if pos_json.endswith('.json')]
            relevant_dict = defaultdict(list)

            for file in json_files:
                with open(file, encoding='utf-8') as f:
                    input_dict = json.load(f)
                    for pos_tag, tag_list in input_dict.items():
                        [relevant_dict[pos_tag].append(tag) for tag in tag_list]

            relevant_set = {}

            for pos, tag_list in relevant_dict.items():
                # take 10 most occurring items
                most_occurrences = Counter(tag_list).most_common()[:10]
                tag_list = [item[0] for item in most_occurrences]
                relevant_set[pos] = tag_list

            save_location = os.path.join(DATA_DIR, 'relevant_sets', group, '{}_norm_relevant_set_{}.json'.format(
                group, pos_tag_to_test))
            with open(save_location, 'w', encoding='utf-8') as out_file:
                json.dump(relevant_set, out_file, ensure_ascii=False)


def reference():
    for pos_tag_to_test in ['nouns', 'verbs']:
        relevant_dict = defaultdict(list)

        root_dir = os.path.join(DATA_DIR, 'reference_sets')
        for dir_name, _, file_list in os.walk(root_dir):
            for file_name in file_list:
                full_path = os.path.join(dir_name, file_name)

                # only take reference files with the requested pos tag
                if not (file_name.startswith('reference_set_') and pos_tag_to_test in file_name):
                    continue

                with open(full_path, encoding='utf-8') as f:
                    input_dict = json.load(f)
                    for pos_tag, tag_list in input_dict.items():
                        [relevant_dict[pos_tag].append(tag) for tag in tag_list]

        reference_set = {}

        for pos, tag_list in relevant_dict.items():
            # take 10 most occurring items
            most_occurrences = Counter(tag_list).most_common()[:10]
            tag_list = [item[0] for item in most_occurrences]
            reference_set[pos] = tag_list

        save_location = os.path.join(DATA_DIR, 'reference_sets', 'norm_reference_set_{}.json'.format(pos_tag_to_test))
        with open(save_location, 'w', encoding='utf-8') as out_file:
            json.dump(reference_set, out_file, ensure_ascii=False)


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--set_type', action="store", choices=['relevant', 'reference'], default='reference')
    options, remainder = parser.parse_args()

    if options.set_type == 'relevant':
        relevant()
    else:
        reference()

