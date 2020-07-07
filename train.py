import argparse
import numpy as np
import torch
from tqdm import tqdm
import pathlib
import torchvision


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='c15',
                    help='the yml which include all parameters !')
parser.add_argument()
