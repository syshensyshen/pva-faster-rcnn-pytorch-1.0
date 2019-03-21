import torch
import os
from collections import OrderedDict
import cv2
import numpy as np
import torchvision.transforms as transforms
import argparse
from models.lite import lite_faster_rcnn
from models.pvanet import pva_net

def parse_args():
  """
  Parse input arguments
  """
  parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
  parser.add_argument('--model', default='lite', type=str)
  parser.add_argument('--save_dir', default='./', type=str)

  args = parser.parse_args()
  return args


"""  Functions  """
def merge(params, name, layer):
    # global variables
    global weights, bias
    global bn_param

    if layer == 'Convolution':
        # save weights and bias when meet conv layer
        if 'weight' in name:
            weights = params.data
            bias = torch.zeros(weights.size()[0])
        elif 'bias' in name:
            bias = params.data
        bn_param = {}

    elif layer == 'BatchNorm':
        # save bn params
        bn_param[name.split('.')[-1]] = params.data

        # running_var is the last bn param in pytorch
        if 'running_var' in name:
            # let us merge bn ~
            tmp = bn_param['weight'] / torch.sqrt(bn_param['running_var'] + 1e-5)
            weights = tmp.view(tmp.size()[0], 1, 1, 1) * weights
            bias = tmp*(bias - bn_param['running_mean']) + bn_param['bias']

            return weights, bias

    return None, None


def main():
    args = parse_args()
    model_path = args.model_path
    save_path = args.save_dir
    pytorch_net = lite_faster_rcnn.eval()
    # load weights
    print('Finding trained model weights...')
    print('Loading weights from %s ...' % model_path)
    trained_weights = torch.load(model_path)
    # pytorch_net.load_state_dict(trained_weights)
    print('Weights load success')
    # go through pytorch net
    print('Going through pytorch net weights...')
    new_weights = OrderedDict()
    inner_product_flag = False
    for name, params in trained_weights.items():
        if len(params.size()) == 4:
            _, _ = merge(params, name, 'Convolution')
            prev_layer = name
        elif len(params.size()) == 1 and not inner_product_flag:
            w, b = merge(params, name, 'BatchNorm')
            if w is not None:
                new_weights[prev_layer] = w
                new_weights[prev_layer.replace('weight', 'bias')] = b
        else:
            # inner product layer
            # if meet inner product layer,
            # the next bias weight can be misclassified as 'BatchNorm' layer as len(params.size()) == 1
            new_weights[name] = params
            inner_product_flag = True

    # align names in new_weights with pytorch model
    # after move BatchNorm layer in pytorch model,
    # the layer names between old model and new model will mis-align
    print('Aligning weight names...')
    pytorch_net_key_list = list(pytorch_net.state_dict().keys())
    new_weights_key_list = list(new_weights.keys())
    assert len(pytorch_net_key_list) == len(new_weights_key_list)
    for index in range(len(pytorch_net_key_list)):
        new_weights[pytorch_net_key_list[index]] = new_weights.pop(new_weights_key_list[index])
    name = os.path.basename(model_path)
    torch.save(new_weights, save_path + '/' + name.replace('.pth', '_merged.pth'))

if __name__ == "__main__":
    main()
