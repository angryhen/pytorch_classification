from torch2trt import torch2trt
import torch
import torchvision
import time
import collections
from use_model import choice_model

def resume_custom(checkpoint, model):
    ch = torch.load(checkpoint)
    ans = collections.OrderedDict()
    for k, v in ch['state_dict'].items():
        ind = k.find('model')
        ans[k[ind:]] = v

    # ans.pop('model.fc.weight')
    # ans.pop('model.fc.bias')
    model_dict = model.state_dict()
    model_dict.update(ans)
    model.load_state_dict(model_dict)
    return model

checkpoint = '/home/du/Desktop/my_project/pytorch_classification/logs/resnest50_c15/18.pth'
model_name = 'resnest50'
onnx_model_name = 'resnest50.onnx'
n_class = 15

model = choice_model(model_name, n_class)
model = resume_custom(checkpoint, model)
model = model.cuda().eval()

# model = torchvision.models.resnet18(pretrained=True).cuda().eval()
data = torch.randn((1,3,224,224)).cuda()
model_trt = torch2trt(model, [data], fp16_mode=True)


a = model(data)
b = model_trt(data)
print(torch.max(torch.abs(a - b)))
#
print('start run')
start = time.time()
n = 10000
for i in range(n):
    output = model(data)
print(f'time: {(time.time() - start)/n}')

start = time.time()
for i in range(n):
    output_trt = model_trt(data)
print(f'time_trt: {(time.time() - start)/n}')
#
# # print(output)
# # print(output_trt)
# print(torch.max(torch.abs(output - output_trt)))


