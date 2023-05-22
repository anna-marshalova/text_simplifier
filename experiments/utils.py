import os
import logging
import importlib

from experiments.paths import LOG_PATH, LOG_PATH_LOCAL

def set_logging():
    """
    Настройка логирования
    :return:
    """
    # чтобы путь к файлам был абсолютным. после выполнения функции нужно вернуться в text_simplifer
    os.chdir('/')
    logger = logging.getLogger('results')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    # чтобы логи сохранялись на диск
    fh = logging.FileHandler(LOG_PATH)
    logger.addHandler(fh)
    # чтобы логи сохранялись локально
    fhl = logging.FileHandler(LOG_PATH_LOCAL)
    logger.addHandler(fhl)
    return logger

def reload(module_path):
    """
    Перезагрузка модуля
    :param module_path: Путь к модулю, разделенный точками (например, 'experiments.utils')
    """
    module = importlib.import_module(module_path)
    importlib.reload(module)
    print(f'{module} reloaded successfully')

def stringify_dict(d):
    return ' '.join(f'{k}:{v}' for k, v in d.items())