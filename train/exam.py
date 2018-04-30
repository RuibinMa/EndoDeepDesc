
'''
Created on Apr 23, 2018

@author: ruibinma
'''


from __future__ import print_function
import os
import numpy as np
import sqlite3
import cv2
from shutil import rmtree
from randomdict import RandomDict

grid_sz = 16
input_sz = 64
input_channels = 1
ones_arr = np.ones((input_sz, input_sz), dtype=np.uint8)

def blob_to_array(blob, dtype, shape=(-1,)):
    return np.frombuffer(blob, dtype).reshape(*shape)

def extract_patch(img, kp):
    """
    Extract the patch and subtract the mean

    :param img: The input image
    :param kp: OpenCV keypoint objects
    
    :return: An array with the patches with zero mean
    """
    # extract patch
    sub = cv2.getRectSubPix(img, (int(round(kp.size*20)), int(round(kp.size*20))), kp.pt)
    #sub = self.rectify_patch(img, kp, self.input_sz)

    # resize the patch
    res = cv2.resize(sub, (64, 64))
    # subtract mean
    #nmean = res - (ones_arr * cv2.mean(res)[0])
    #nmean = nmean.reshape(input_channels, input_sz, input_sz)
    #return nmean
    return res

def extract_patch_from_grid(patch_id):
    grid_id = patch_id / (grid_sz * grid_sz)
    in_grid_n = patch_id % (grid_sz * grid_sz)
    grid_row = in_grid_n / grid_sz
    grid_col = in_grid_n % grid_sz
    
    img = cv2.imread(os.path.join(output_folder, 'patches%04d.bmp'%grid_id), cv2.IMREAD_GRAYSCALE)
    
    patch = img[grid_row*input_sz:(grid_row+1)*input_sz, grid_col*input_sz:(grid_col+1)*input_sz]
    return patch

data_folder = '../data/sample'
image_folder = os.path.join(data_folder, 'images')
sfm_folder = os.path.join(data_folder, 'sfm_results')
output_folder = os.path.join(data_folder, 'output')
database_path = os.path.join(data_folder, 'database.db')
record_file_name = os.path.join(output_folder, 'record.txt')

matches_file_name = os.path.join(output_folder, 'm50_000000_000000_0.txt')
mfile = open(matches_file_name, 'r')
for i in range(350000):
    mfile.readline()
for i in range(100):
    line = mfile.readline().split()
    print(line)
    
    patch_id1 = np.int(line[0])
    patch_id2 = np.int(line[3])
    point3D_id1 = np.int(line[1])
    point3D_id2 = np.int(line[4])
    
    img = np.zeros(shape=(input_sz, input_sz*2), dtype=np.uint8)
    patch1 = extract_patch_from_grid(patch_id1)
    patch2 = extract_patch_from_grid(patch_id2)
    img[0:input_sz,0:input_sz] = patch1
    img[0:input_sz, input_sz:input_sz*2] = patch2 
    if point3D_id1 == point3D_id2:       
        cv2.imwrite('%04d_match.jpg'%i, img)
    else:
        cv2.imwrite('%04d_nonmatch.jpg'%i, img)
    
    
    
    
    
    