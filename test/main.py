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
import torch
import torch.nn as nn
import argparse
from scipy.io import loadmat
from torch.autograd import Variable

def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('patchesfile', type=str)
    parser.add_argument('image_name', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('--checkpoint', type=str, default='./testmodel.pth')
    args = parser.parse_args()
    return args


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
    args = parse_args()
    resume = args.checkpoint
    print('patchesfile: %s'%args.patchesfile)
    print('image_name : %s'%args.image_name)
    print('output_path: %s'%args.output_path)
    print('checkpoint : %s'%resume)
    # instantiate model and initialize weights
    model = TNet()
    model.cuda()

    #optimizer = create_optimizer(model, args.lr)

    # resume from a checkpoint
    if os.path.isfile(resume):
        print('=> loading checkpoint {}'.format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print('=> no checkpoint found at {}'.format(resume))

    raw_patches = loadmat(args.patchesfile)['patches']

    n_patches = raw_patches.shape[0]
    patches = np.zeros((n_patches,1,TFEAT_PATCH_SIZE, TFEAT_PATCH_SIZE))
    for i in range(n_patches):
        patches[i,0,:,:] = preprocess_patch(raw_patches[i, :, :]) 

    model.eval()
    descriptors = extract_tfeats(model, patches)

    print(descriptors.shape)
    
    # write descriptor to .bin
    desc_file_name = args.output_path
    with open(desc_file_name, 'wb') as file:
        np.asarray(descriptors.shape, dtype=np.int32).tofile(file)
        descriptors.astype(np.float32).tofile(file)

if __name__ == '__main__':
    main()
