import os

DRIVE_ROOT = os.path.join('content','drive','MyDrive','simplicity')

CHECKPOINTS_FOLDER = os.path.join(DRIVE_ROOT, 'checkpoints')

def get_checkpoints_path(model_id, dataset):
    return os.path.join(CHECKPOINTS_FOLDER, f'{model_id}-{dataset}.pt')