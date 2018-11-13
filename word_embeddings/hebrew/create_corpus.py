# Download wiki corpus from https://dumps.wikimedia.org/hewiki/latest/
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from gensim.corpora import WikiCorpus
import csv


if __name__ == '__main__':
    space = " "
    print("Starting to create data corpus")

    with open("wiki_hebrew_corpus.txt", 'w', encoding="utf-8") as output:
        wiki_counter = 0
        print("Adding articles from wiki corpus")
        wiki = WikiCorpus("hewiki-latest-pages-articles.xml.bz2", lemmatize=False, dictionary={})

        for text in wiki.get_texts():
            article = space.join([t for t in text])
            output.write(article + '\n')
            wiki_counter += 1
            if wiki_counter % 1000 == 0:
                print("Saved " + str(wiki_counter) + " articles")

        skip_first_2_rows = 0
        csv_counter = 0
        print("Adding records from medical corpus")

        with open("all_data.csv", encoding="utf-8") as file:
            csv_file = csv.reader(file)

            # skip first 2 rows
            for answers in csv_file:
                if skip_first_2_rows < 2:
                    skip_first_2_rows += 1
                    continue

                article = space.join([t for t in answers[2:]])
                output.write(article + '\n')
                csv_counter += 1
                if csv_counter % 10 == 0:
                    print("Saved " + str(csv_counter) + " csv rows")

    print("Finished - Saved " + str(wiki_counter + csv_counter) + " articles")
