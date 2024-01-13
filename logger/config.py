# https://docs.python.org/3/library/logging.config.html

import logging
from logging import config
import logging.handlers

config_wdb = {"project_name": "hse-m4xig1", "run_name": "test spex+", "config": {}}

config_log = {
    "formatters": {
        "brief": {"format": "%(message)s"},
        "precise": {
            "format": "%(asctime)s %(levelname)-8s %(name)-15s %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "brief",
            "level": "DEBUG",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "precise",
            "filename": "logconfig.log",
            "backupCount": 10,
            "encoding": "utf8",
        },
    },
    "version": 1,
    "disable_existing_loggers": False,
}
