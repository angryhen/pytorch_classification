"""
An example that uses TensorRT's Python api to make inferences.
"""
import ctypes
import os
import random
import sys
import threading
import time

import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torch.nn as nn
import torchvision

from onnx_tool.trtWarpper import TrtWarp

INPUT_W = 224
INPUT_H = 224
CONF_THRESH = 0.5
IOU_THRESHOLD = 0.4


class myThread(threading.Thread):
    def __init__(self, func, args):
        threading.Thread.__init__(self)
        self.func = func
        self.args = args

    def run(self):
        self.func(*self.args)


def preprocess_image(input_image_path):
    image = cv2.imread(input_image_path)  # BGR 0-255 hwc
    resized_img = cv2.resize(image, (INPUT_W, INPUT_H))
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)  # RGB
    # normalize
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inp_image = ((resized_img / 255. - mean) / std).astype(np.float32)  # R-0.485  B-
    image = np.transpose(inp_image, [2, 0, 1])
    # CHW to NCHW format
    image = np.expand_dims(image, axis=0)
    # Convert the image to row-major order, also known as "C order":
    image = np.ascontiguousarray(image)
    return image


def post_result(output):
    # result = output[0]
    result = torch.Tensor(output).cuda()
    result = torch.reshape(result, (-1, 254))
    smax = nn.Softmax()
    result = smax(result)
    _, preds = torch.max(result.data, 1)
    index = preds.data.cpu().numpy()

    # get label
    label = []
    with open('label.txt', 'r') as f:
        labels = f.readlines()
    for i in labels:
        label.append(i[:-1])


    # print(result)
    print(result[0, index[0]])
    print(index, label[index[0]])


if __name__ == '__main__':
    # load custom plugins
    input_image_paths = "/home/du/disk2/Desk/dataset/ibox/cls/shandong_data/xianchang/cut/e4b9a1e5b7b4e4bdace9babbe8bea3e885bf5fe4b9a1e5b7b4e4bdac5f313230e5858b/11-10_08-03-35_000080_1.jpg"
    engine_file_path = "/home/du/c++/trt_project/eff-b4.engine"

    # a YoLov5TRT instance
    yolov5_warpper = TrtWarp(engine_file_path)

    test_img = np.random.randn(1,3, 224,224)
    yolov5_warpper.infer(test_img)

    time.sleep(3)
    start = time.time()
    image = preprocess_image(input_image_paths)
    print(f'预处理： {time.time() - start}')

    for i in range(10):
        result = yolov5_warpper.infer(image)
    post_result(result)
    print(f'总时间： {time.time() - start}')

    # second time
    start = time.time()
    image = preprocess_image(input_image_paths)
    print(f'预处理： {time.time() - start}')

    result = yolov5_warpper.infer(image)
    print(f'总时间： {time.time() - start}')
    post_result(result)

    # destory the instance
    yolov5_warpper.destory()
