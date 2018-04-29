"""
    PyTorch training code for TFeat shallow convolutional patch descriptor:
    http://www.iis.ee.ic.ac.uk/%7Evbalnt/shallow_descr/TFeat_paper.pdf

    The code reproduces *exactly* it's lua anf TF version:
    https://github.com/vbalnt/tfeat

    2017 Edgar Riba
"""

from __future__ import print_function
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import argparse
import matlab.engine
from torch.autograd import Variable
from shutil import rmtree


def parse_args():
    parser = argparse.ArgumentParser(description='tfeat calculate descriptors.')
    parser.add_argument('data_folder', type=str)
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
    
#     print('patchesfile: %s'%args.patchesfile)
#     print('image_name : %s'%args.image_name)
#     print('output_path: %s'%args.output_path)
    print('checkpoint : %s'%resume)
    
    data_folder = args.data_folder
    image_folder = os.path.join(data_folder, 'images')
    keypoints_folder = os.path.join(data_folder, 'keypoints')
    descriptor_folder = os.path.join(data_folder, 'descriptors')
    database_path = os.path.join(data_folder, 'database.db')
    
    if os.path.exists(descriptor_folder):
        rmtree(descriptor_folder)
    os.mkdir(descriptor_folder)  
    
    image_names = os.listdir(image_folder)
    image_names.sort()
    keypoint_paths = []
    descriptor_paths = []
    image_paths = []
    patches_path = os.path.join(data_folder, 'temp_patches.bin')
    n_images = len(image_names)
    for image_name in image_names:
        image_paths.append(os.path.join(image_folder, image_name))
        keypoint_paths.append(os.path.join(keypoints_folder, image_name + '.bin'))
        descriptor_paths.append(os.path.join(descriptor_folder, image_name + '.bin'))
    
    # start matlab engine
    eng = matlab.engine.start_matlab()
    eng.bootstrap(nargout=0)
    
    # instantiate model and initialize weights
    model = TNet()
    model.cuda()
    # resume from a checkpoint
    if os.path.isfile(resume):
        print('=> loading checkpoint {}'.format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        #checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print('=> no checkpoint found at {}'.format(resume))
    model.eval()
    
    for i in range(n_images):
        # pass patches to .bin then to python
        eng.extract2(image_paths[i], keypoint_paths[i], patches_path, nargout=0)
        with open(patches_path, 'rb') as f:
            shape = np.fromfile(f, count=3, dtype=np.int32)
            patches = np.fromfile(f, count=shape[0]*shape[1]*shape[2], dtype=np.float32).reshape(shape)
        
        # directly pass patches from matlab to python: this is much slower by test
        #patches = eng.extract(image_paths[i], keypoint_paths[i], 32)
        #patches = np.array(patches._data).reshape(patches.size, order='F')
        print('verification key2: %.6f'%patches[200,20,50])
        print('%s : %d (%dx%d) image patches'%(image_names[i], patches.shape[0], patches.shape[1], patches.shape[2]))
        
        n_patches = patches.shape[0]
        patches_ = np.zeros((n_patches,1,TFEAT_PATCH_SIZE, TFEAT_PATCH_SIZE))
        for j in range(n_patches):
            patches_[j,0,:,:] = preprocess_patch(patches[j, :, :]) 
        descriptors = extract_tfeats(model, patches_)
        print(descriptors.shape)
        with open(descriptor_paths[i], 'wb') as f:
            np.asarray(descriptors.shape, dtype=np.int32).tofile(f)
            descriptors.astype(np.single).tofile(f)
        
    os.remove(patches_path)
    
    eng.match(data_folder, nargout=0)
    
if __name__ == '__main__':
    main()
