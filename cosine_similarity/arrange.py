import csv
import os


# This script takes the cos-sim scores csv and rearranges it by questions
# input csv needs to be utf-8 encoded first
if __name__ == '__main__':
    with open(os.path.join('..', 'outputs', 'cossim_rearranged.csv'), 'w+', newline='', encoding="utf-8") as fo:
        writer = csv.writer(fo)

        for i in range(1, 19):
            with open(os.path.join('..', 'outputs', 'cossim_results.csv'), 'r', encoding="utf-8") as f:
                question_to_users = []
                for line in csv.reader(f):
                    if line[1] == str(i):
                        question_to_users.append(line)
                    elif not line[1].isdigit():
                        writer.writerow(line)

                for line in question_to_users:
                    writer.writerow(line)
