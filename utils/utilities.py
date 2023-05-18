def stringify_dict(d):
    return ' '.join(f'{k}:{v}' for k, v in d.items())

import importlib

def reload(module_path):
    module = importlib.import_module(module_path)
    importlib.reload(module)