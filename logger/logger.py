import logging
import logging.config

from pprint import pprint
from pathlib import Path

from config import config_log, config_wdb


# config must be an associative container
def start_log(config=None, mode=logging.INFO):
    if config:
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=mode)

