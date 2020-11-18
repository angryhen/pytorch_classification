import collections
import struct

import torch
from torchsummary import summary

from lib.use_model import choice_model


def resume_custom(checkpoint, model):
    ch = torch.load(checkpoint)
    ans = collections.OrderedDict()
    for k, v in ch['state_dict'].items():
        ans[k] = v

    model_dict = model.state_dict()
    model_dict.update(ans)
    model.load_state_dict(model_dict)
    return model


def main():
    checkpoint = '/home/du/Desktop/my_project/pytorch_classification/logs/11-13_17:32_efficientnet-b4_c254/50.pth'
    model_name = 'efficientnet-b4'
    n_class = 254

    # model
    model = choice_model(model_name, n_class)
    net = model = resume_custom(checkpoint, model)
    net.to('cuda:0').eval()
    print('model: ', net)


    tmp = torch.ones(1, 3, 224, 224).to('cuda:0')
    print('input: ', tmp)
    out = net(tmp)

    print('output:', out)

    summary(net, (3, 224, 224))
    # return
    f = open("eff-b4.wts", 'w')
    f.write("{}\n".format(len(net.state_dict().keys())))
    for k, v in net.state_dict().items():
        print('key: ', k)
        print('value: ', v.shape)
        vr = v.reshape(-1).cpu().numpy()
        f.write("{} {}".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")


if __name__ == '__main__':
    main()
