import argparse
import os
import time
import sys
import cv2
import numpy as np
import shutil
import pandas as pd
import torch
import torch.nn as nn
from alfred.utils.log import logger
from sklearn.metrics import classification_report

from configs.config import get_cfg_defaults
from data.dataloader import create_dataloader
from data.transform import create_transform
from lib.use_model import choice_model


# load yml_file
def get_config(args):
    config = get_cfg_defaults()
    if args.configs:
        yml_file = args.configs
        config.merge_from_file(yml_file)
    config.freeze()
    return config


# load model
def load_model(config):
    model = choice_model(config.model.name, config.model.num_classes)
    if torch.cuda.is_available():
        model = model.cuda()
    ch = torch.load(config.model.checkpoint)
    model.load_state_dict(ch['state_dict'])
    return model


def single_image(img, model):
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    trans = create_transform(config, 'test')
    img = trans(image=img)['image']
    img = img.unsqueeze(0)
    if torch.cuda.is_available():
        input = img.cuda(non_blocking=config.test.dataloader.non_blocking)
    output = model(input)
    output = torch.clamp(output, 0, 100)
    smax = nn.Softmax()
    output = smax(output)
    # print(output)
    score = max(output.data.cpu().numpy()[0])
    _, preds = torch.max(output.data, 1)
    result = preds.data.cpu().numpy()
    return result, score


def test_csv(val_loader, model, target_names, half=False):
    from tqdm import tqdm
    model.eval()

    test_pred = []
    test_target = []
    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader)):
            if torch.cuda.is_available():
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            if half:
                input = input.half()
            # compute output
            output = model(input)
            output = torch.clamp(output, 0, 100)
            smax = nn.Softmax()
            output = smax(output)
            if torch.cuda.is_available():
                output = output.data.cpu().numpy()
                target = target.data.cpu().numpy()
            else:
                output = output.data.numpy()
            test_target.append(target)
            test_pred.append(output)
    test_pred = np.vstack(test_pred)
    test_target = np.concatenate(test_target)
    test_pred = np.argmax(test_pred,axis=1)
    result = classification_report(test_target, test_pred, target_names=target_names)
    return result



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ResNeSt for image classification')
    parser.add_argument('-c', '--configs', type=str, default=None,
                        help='the yml which include all parameters!')
    parser.add_argument('--images', type=str,
                       default=None,
                       help='your image path')
    parser.add_argument('--save', type=str,
                       default=None,
                       help='result save path')

    args = parser.parse_args()

    config = get_config(args)

    device = torch.device(config.device)
    models = load_model(config)
    models.eval()

    path = args.images
    save_path = args.save

    # predict single image
    logger.info('test mode: single image')
    # files = os.listdir(path)
    # for i in files:
    #     file = os.path.join(path, i)
    #     result, score = single_image(file, models)
    #     save_name = os.path.join(save_path, f'{result}_{score}_{i[:-4]}.jpg')
    #     shutil.copyfile(file, save_name)
    #
    #     logger.info('result:', result)
    #     print('result:', result, 'label:', config.labels_list[int(result)])

    labels_list = ['paper', 'glass', 'cardboard', 'plastic', 'trash', 'metal']
    files = os.listdir(path)
    with open('result.csv', 'w') as f:
        for i in files:
            file = os.path.join(path, i)
            result, score = single_image(file, models)
            print(result)
            f.write(f'{i[:-4]},{labels_list[result[0]]}\n')
