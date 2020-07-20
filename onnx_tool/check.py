import cv2
import time
import numpy as np
import onnx
import onnxruntime
from albumentations import Compose
from albumentations import Resize, Normalize
from albumentations.pytorch import ToTensorV2

labels_list = ['hzy-hzy-pz-bxgw-500ml', 'ty-hzy-pz-gw-500ml', 'ty-tyxmtx-pz-lplldc-480ml', 'ty-tylythsnr-tz-hsnrw-105g',
               'tdr-tdrstglm-tz-yw-83g', 'ty-tyxmtx-pz-lpqnhc-480ml', 'asm-asmnc-pz-yw-500ml', 'ty-qbsds-pz-nmw-500ml',
               'ty-tync-pz-mxw-310ml', 'yh-yhbkf-pz-yw-450ml', 'ty-tybhc-pz-nmw-500ml', 'ty-tylc-pz-mlw-500ml',
               'ty-tyhsnrm-tz-nr-105g', 'tdr-tdrsslltgm-tz-yw-90g', 'ty-tyhld-gz-xclc-310ml']

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def create_transform():
    transforms = Compose([
        Resize(224, 224),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2()])
    return transforms

def process(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    trans = create_transform()
    img = trans(image=img)['image']
    img = img.unsqueeze(0)
    img = img.numpy()
    print(img.shape)
    return img

# load onnx
onnx_model = onnx.load('resnest50.onnx')
print(onnx.checker.check_model(onnx_model))

onnx.helper.printable_graph(onnx_model.graph)

ort_session = onnxruntime.InferenceSession('resnest50.onnx')

# prepare img
img_path = '/home/du/disk2/Desk/dataset/ibox/cls/dbox/dbox_dedup2/hzy-hzy-pz-bxgw-500ml/2020-05-11-17-51-54_ty_ty-qbsds-pz-nmw-500ml_000006.jpg'
img = process(img_path)

# run
start = time.time()
outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: img})
print(time.time() - start)

# print
ind = np.argmax(outputs[0])
print(labels_list[int(ind)])