from torch2trt import torch2trt
import torch
import torchvision
import time
import collections
from lib.use_model import choice_model

def resume_custom(checkpoint, model):
    ch = torch.load(checkpoint)
    ans = collections.OrderedDict()
    for k, v in ch['state_dict'].items():
        # ind = k.find('model')
        # ans[k[ind:]] = v
        ans[k] = v

    model_dict = model.state_dict()
    model_dict.update(ans)
    model.load_state_dict(model_dict)
    return model


if __name__ == '__main__':
    # parameters
    checkpoint = '/home/du/Desktop/my_project/pytorch_classification/logs/11-13_17:32_efficientnet-b4_c254/50.pth'
    model_name = 'efficientnet-b4'
    # onnx_model_name = 'resnest50.onnx'
    n_class = 254
    img_size = 224
    data = torch.randn((1,3,img_size,img_size)).cuda()
    n = 100

    # model
    model = choice_model(model_name, n_class)

    model = resume_custom(checkpoint, model)
    model = model.cuda().eval()
    data = data.contiguous()
    model_trt = torch2trt(model, [data], fp16_mode=False)

    # test error
    result = model(data)
    result_trt = model_trt(data)
    print(torch.max(torch.abs(result - result_trt)))

    # test speed
    print('start run')
    start = time.time()
    for i in range(n):
        output = model(data)
    print(f'time: {(time.time() - start)/n}')

    start = time.time()
    for i in range(n):
        output_trt = model_trt(data)
    print(f'time_trt: {(time.time() - start)/n}')

    # save trt_model
    model_trt_pth = f'{model_name}_trt.pth'
    torch.save(model_trt.state_dict(), model_trt_pth)

    # load trt_model
    # from torch2trt import TRTModule
    # model_trt = TRTModule()
    # model_trt.load_state_dict(torch.load(model_trt_pth))