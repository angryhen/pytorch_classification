import argparse
import numpy as np
import torch
from tqdm import tqdm
import pathlib
import torchvision

from configs.config import get_cfg_defaults


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--configs', type=str, default='c15',
                    help='the yml which include all parameters!')
args = parser.parse_args()


def get_config():
    config = get_cfg_defaults()
    return config


def main():
    config = get_config()
