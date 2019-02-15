import logging
import os

from word_embeddings.common.utils import OUTPUTS_DIR

LOGGER = logging.getLogger('Main')
LOGGER.setLevel(logging.INFO)
file_handler = logging.FileHandler(filename=os.path.join(OUTPUTS_DIR, 'classifier_outputs.txt'), encoding='utf-8')
formatter = logging.Formatter(u"%(message)s")
file_handler.setFormatter(formatter)
LOGGER.addHandler(file_handler)


