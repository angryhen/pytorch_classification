import onnx
import onnxruntime
import torch
import torchvision
import numpy as np
import collections
from lib.use_model import choice_model

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


if __name__ == "__main__":
    checkpoint = '/home/du/Desktop/my_project/pytorch_classification/logs/resnest50_c15/18.pth'
    model_name = 'resnest50'
    onnx_model_name = 'resnest50.onnx'
    n_class = 15

    model = choice_model(model_name, n_class)
    model = resume_custom(checkpoint, model)

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    x = torch.randn(1, 3, 224, 224, requires_grad=True).cuda()

    # Export the model
    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      onnx_model_name,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      verbose=True,
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      # dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                      #               'output': {0: 'batch_size'}}
                      dynamic_axes=None
                      )

    onnx_model = onnx.load(onnx_model_name)
    onnx.checker.check_model(onnx_model)


