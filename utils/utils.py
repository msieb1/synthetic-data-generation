import numpy as np
from sklearn.model_selection import train_test_split
from os.path import join
import os
import matplotlib.pyplot as plt
import datetime
import sys

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader



def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")
def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)

def ensure_folder(folder):
    path_fragments = os.path.split(folder)
    joined = '.'
    for fragment in path_fragments:
        joined = os.path.join(joined, fragment)
        if not os.path.exists(joined):
            os.mkdir(joined)

def time_stamped_w_name(fname, fmt='%Y-%m-%d-%H-%M-%S_{fname}'):
    return datetime.datetime.now().strftime(fmt).format(fname=fname)

def time_stamped(fmt='%Y-%m-%d-%H-%M-%S{}'):
    return datetime.datetime.now().strftime(fmt).format('')



    

"""
GPU wrappers from 
https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/pytorch_util.py
"""

_use_gpu = False
device = None

def set_gpu_mode(mode, gpu_id=0):
    global _use_gpu
    global device
    global _gpu_id
    _gpu_id = gpu_id
    _use_gpu = mode
    device = torch.device("cuda:" + str(gpu_id) if _use_gpu else "cpu")

def gpu_enabled():
    return _use_gpu

def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)

# noinspection PyPep8Naming
def FloatTensor(*args, **kwargs):
    return torch.FloatTensor(*args, **kwargs).to(device)

def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)

def get_numpy(tensor):
    # not sure if I should do detach or not here
    return tensor.to('cpu').detach().numpy()

def zeros(*sizes, **kwargs):
    return torch.zeros(*sizes, **kwargs).to(device)

def ones(*sizes, **kwargs):
    return torch.ones(*sizes, **kwargs).to(device)

def randn(*args, **kwargs):
    return torch.randn(*args, **kwargs).to(device)

def zeros_like(*args, **kwargs):
    return torch.zeros_like(*args, **kwargs).to(device)

def normal(*args, **kwargs):
    return torch.normal(*args, **kwargs).to(device)