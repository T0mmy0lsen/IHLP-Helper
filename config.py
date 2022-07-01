import os

# A guide for installing tensorflow is always helpful
# https://towardsdatascience.com/the-ultimate-tensorflow-gpu-installation-guide-for-2022-and-beyond-27a88f5e6c6e

# Because direct paths are always better
BASE_PATH = os.path.dirname(__file__)

# Path for Word Embedding
PATH_INPUT_FOR_WORD_EMBEDDING = f'{BASE_PATH}'
PATH_OUTPUT_FOR_WORD_EMBEDDING = f'{BASE_PATH}'

# Paths for raw database data
PATH_INPUT_COMMUNICATION = f'{BASE_PATH}/data/communication.csv'
PATH_INPUT_ITEM = f'{BASE_PATH}/data/item.csv'
PATH_INPUT_RELATION = f'{BASE_PATH}/data/relation.csv'
PATH_INPUT_RELATION_HISTORY = f'{BASE_PATH}/data/relation_history.csv'
PATH_INPUT_REQUEST = f'{BASE_PATH}/data/request.csv'