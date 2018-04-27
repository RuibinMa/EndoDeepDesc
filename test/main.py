"""
    PyTorch training code for TFeat shallow convolutional patch descriptor:
    http://www.iis.ee.ic.ac.uk/%7Evbalnt/shallow_descr/TFeat_paper.pdf

    The code reproduces *exactly* it's lua anf TF version:
    https://github.com/vbalnt/tfeat

    2017 Edgar Riba
"""

from __future__ import print_function
import os
import sys
import cv2
import numpy as np
import lutorpy as lua
import torchfile
import torch
import torch.nn as nn
from scipy.io import loadmat
from torch.autograd import Variable
require('nn')
require('cunn')
require('cudnn')

resume = 'testmodel.pth'

class TNet(nn.Module):
    """TFeat model definition
    """
    def __init__(self):
        super(TNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=6),
            nn.Tanh()
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

TFEAT_PATCH_SIZE = 32
TFEAT_DESC_SIZE = 128
TFEAT_BATCH_SIZE = 1000
MEAN = 0.48544601108437
STD = 0.18649942105166


def preprocess_patch(patch):
    out = cv2.resize(patch, (TFEAT_PATCH_SIZE, TFEAT_PATCH_SIZE)).astype(np.float32) / 255;
    out = (out - MEAN) / STD
    return out.reshape(1, TFEAT_PATCH_SIZE, TFEAT_PATCH_SIZE)

def extract_tfeats(net,patches):
    num,channels,h,w = patches.shape
    patches_t = torch.FloatTensor(patches)
    patches_t.view(num, 1, TFEAT_PATCH_SIZE, TFEAT_PATCH_SIZE)
    patches_t = patches_t.split(TFEAT_BATCH_SIZE)
    descriptors = []
    for i in range(int(np.ceil(float(num) / TFEAT_BATCH_SIZE))):
        prediction_t = net(Variable(patches_t[i].cuda()))
        prediction = prediction_t.data.cpu().numpy()
        descriptors.append(prediction)
    out =  np.concatenate(descriptors)
    return out.reshape(num, TFEAT_DESC_SIZE)


def main():
    # print the experiment configuration

    # instantiate model and initialize weights
    model = TNet()
    model.cuda()

    #optimizer = create_optimizer(model, args.lr)

    # optionally resume from a checkpoint
    import torch
    if os.path.isfile(resume):
        print('=> loading checkpoint {}'.format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print('=> no checkpoint found at {}'.format(args.resume))

    raw_patches = loadmat('testpatches.mat')['patches']

    n_patches = raw_patches.shape[0]
    patches = np.zeros((n_patches,1,TFEAT_PATCH_SIZE, TFEAT_PATCH_SIZE))
    for i in range(n_patches):
        patches[i,0,:,:] = preprocess_patch(raw_patches[i, :, :]) 

    model.eval()
    descriptors = extract_tfeats(model, patches)

    print(descriptors.shape)

if __name__ == '__main__':
    main()
