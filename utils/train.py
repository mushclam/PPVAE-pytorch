import logging
import os
import time

import random
import numpy as np
import torch


def fix_seed(seed):
    # Python
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # CUDA
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except:
        print('The model can not run with deterministic algorithms only.')
    if torch.version.cuda >= str(10.2):
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'
        # or
        # os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:2'
    else:
        os.environ['CUDA_LAUNCH_BLOCKING']='1'

def get_logger(
    args, log_name='train',
    log_level=logging.INFO,
    log_file=True,
    file_level=None,
    log_stream=True,
    stream_level=None,
    path=None
    ):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    if log_file:
        file_level = file_level if file_level is not None else log_level
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        handler = logging.FileHandler(
            os.path.join(args.save_dir if path is None else path, '{}.log'.format(log_name)),
            'w'
        )
        handler.setLevel(file_level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if log_stream:
        stream_level = stream_level if stream_level is not None else log_level
        console = logging.StreamHandler()
        console.setLevel(stream_level)
        console.setFormatter(formatter)
        logger.addHandler(console)
    logger.propagate = False
    logger.info(args)
    return logger


def create_save_path(args):
    model_name = args.model.model_name
    suffix = "/{}".format(model_name) \
        + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    from pathlib import Path
    saved_name = Path(args.save_dir).stem + suffix
    args.save_dir = args.save_dir + suffix

    if os.path.exists(args.save_dir):
        print(f'Warning: the folder {args.save_dir} exists.')
    else:
        print('Creating {}'.format(args.save_dir))
        os.makedirs(args.save_dir)
    # save the config file and model file.
    import shutil
    shutil.copyfile(args.conf, args.save_dir + "/config.yaml")
    os.makedirs(args.save_dir + "/parser")
    shutil.copytree("parser/", args.save_dir + "/parser")
    return  saved_name