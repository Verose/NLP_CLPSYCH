import logging
import warnings

from gensim.models import FastText

from word_embeddings.cosine_similarity import utils
from word_embeddings.cosine_similarity.feature_extractor import FeatureExtractor
from word_embeddings.cosine_similarity.pos_tags_window import POSSlidingWindow
from word_embeddings.cosine_similarity.utils import *

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

OUTPUT_DIR = os.path.join('..', 'outputs')
DATA_DIR = os.path.join('..', 'data')
LOGGER = logging.getLogger('Main')
LOGGER.setLevel(logging.INFO)
file_handler = logging.FileHandler(filename=os.path.join(OUTPUT_DIR, 'main_outputs.txt'))
formatter = logging.Formatter("%(message)s")
file_handler.setFormatter(formatter)
LOGGER.addHandler(file_handler)


def extract_features_to_csv():
    feature_extractor = FeatureExtractor(DATA_DIR)
    feature_extractor.read_answers_pos_tags()
    feature_extractor.update_pos_tags_data()

    data = get_medical_data(DATA_DIR)
    conf = read_conf(DATA_DIR)
    model = FastText.load_fasttext_format(os.path.join(DATA_DIR, 'FastText-pretrained-hebrew', 'wiki.he.bin'))
    # TODO: need several types of cos-sim? maybe just take all?
    cosine_sim = POSSlidingWindow(model, data, conf['size'], conf['extras'], DATA_DIR)
    cosine_sim.calculate_all_avg_scores()
    control_scores_by_question = cosine_sim.get_user_to_question_scores('control')
    patient_scores_by_question = cosine_sim.get_user_to_question_scores('patients')

    feature_extractor.update_cosine_sim_scores(control_scores_by_question, patient_scores_by_question)
    feature_extractor.update_sentiment_scores()
    utils.classifier_features_to_pd(feature_extractor.get_features(), LOGGER, OUTPUT_DIR)


if __name__ == '__main__':
    extract_features_to_csv()
