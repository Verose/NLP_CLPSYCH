import logging
import os

OUTPUT_DIR = os.path.join('..', 'outputs')
DATA_DIR = os.path.join('..', 'data')

LOGGER = logging.getLogger('Main')
LOGGER.setLevel(logging.INFO)
file_handler = logging.FileHandler(filename=os.path.join(OUTPUT_DIR, 'classifier_outputs.txt'))
formatter = logging.Formatter(u"%(message)s")
file_handler.setFormatter(formatter)
LOGGER.addHandler(file_handler)
