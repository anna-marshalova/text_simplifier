import logging
import importlib

from experiments.paths import LOG_PATH, LOG_PATH_LOCAL

def set_logging():
    logger = logging.getLogger('results')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(LOG_PATH)
    logger.addHandler(fh)
    fhl = logging.FileHandler(LOG_PATH_LOCAL)
    logger.addHandler(fhl)
    return logger

def reload(module_path):
    module = importlib.import_module(module_path)
    importlib.reload(module)

def stringify_dict(d):
    return ' '.join(f'{k}:{v}' for k, v in d.items())