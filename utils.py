

import os
from pathlib2 import Path

def load_model_path(root=None, version=None, v_num=None, best=False):
    """ When best = True, return the best model's path in a directory 
        by selecting the best model with largest epoch. If not, return
        the last model saved. You must provide at least one of the 
        first three args.
    Args: 
        root: The root directory of checkpoints. It can also be a
            model ckpt file. Then the function will return it.
        version: The name of the version you are going to load.
        v_num: The version's number that you are going to load.
        best: Whether return the best model.
    """
    def sort_by_epoch(path):
        name = path.stem
        epoch=int(name.split('-')[1].split('=')[1])
        return epoch
    
    def generate_root():
        if root is not None:
            return root
        elif version is not None:
            return str(Path('lightning_logs', version+'_{}'.format(v_num), 'checkpoints'))
        else:
            return str(Path('lightning_logs', 'version_{}'.format(v_num), 'checkpoints'))

    if root==version==v_num==None:
        return None

    root = generate_root()

    if Path(root).is_file():
        return root
    if best:
        files=[i for i in list(Path(root).iterdir()) if i.stem.startswith('best')]
        files.sort(key=sort_by_epoch, reverse=True)
        res = str(files[0])
    else:
        res = str(Path(root))
    return res

def load_model_path_by_args(args):

    return load_model_path(root=args.load_dir, version=args.load_ver, v_num=args.load_v_num)

if __name__ == '__main__':
    print( os.environ.get('PATH_CHECKPOINT', 'lightning_logs/bottleneck_test_version_1/checkpoints'))
    CHECKPOINT_PATH = os.environ.get('PATH_CHECKPOINT', 'lightning_logs/bottleneck_test_version_1/checkpoints')
    VERSION = 'bottleneck_test_version/'

    CHECKPOINT_PATH = os.environ.get('PATH_CHECKPOINT', 'lightning_logs/'+ VERSION + 'checkpoints')

    res = load_model_path(root = CHECKPOINT_PATH, version = None, v_num = None)
    print(res)