import argparse
import os
import time

import cv2
import numpy as np
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
def get_config():
    config = get_cfg_defaults()
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
    _, preds = torch.max(output.data, 1)
    result = preds.data.cpu().numpy()
    return result


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
    paser = argparse.ArgumentParser(description='ResNeSt for image classification')

    paser.add_argument('--image', type=str,
                       default=None,
                       help='your image path')
    paser.add_argument('--csv', type=str,
                       default=1,
                       help='in oder to test all image')
    args = paser.parse_args()

    config = get_config()
    device = torch.device(config.device)

    models = load_model(config)
    models.eval()

    # predict single image
    if args.image:
        result = single_image(args.image, models)
        print('result:', result, 'label:', config.labels[int(result)])

    else:
        print('single image: None')

    if args.csv:
        # skf = KFold(n_splits=0)
        test_data = pd.read_csv(config.test.dataset)
        logger.info(f"test set: {test_data.shape}")

        test_loader = create_dataloader(config, test_data, 'test')

        result = test_csv(test_loader, models, config.labels_list, half=False)

        if not os.path.exists(config.test.log_file):
            with open(config.test.log_file, 'w') as f:
                pass
        with open(config.test.log_file, 'a') as fc:
            fc.write(
                '\n%s %s\n' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
                               f'{config.model}'))
            fc.write(result)
        print(result)
