'''
Created on Apr 23, 2018

@author: ruibinma
'''


from __future__ import print_function
import os
import numpy as np
import sqlite3
import cv2
import matplotlib.pyplot as plt

input_sz = 32
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
    sub = cv2.getRectSubPix(img, (int(kp.size*20), int(kp.size*20)), kp.pt)
    #sub = self.rectify_patch(img, kp, self.input_sz)

    # resize the patch
    res = cv2.resize(sub, (32, 32))
    # subtract mean
    #nmean = res - (ones_arr * cv2.mean(res)[0])
    #nmean = nmean.reshape(input_channels, input_sz, input_sz)
    #return nmean
    return res

data_folder = '../data/sample'
image_folder = os.path.join(data_folder, 'images')
sfm_folder = os.path.join(data_folder, 'sfm_results')
database_path = os.path.join(data_folder, 'database.db')

from pycolmap.scene_manager import SceneManager
loader = SceneManager(sfm_folder)
loader.load_cameras()
loader.load_images()
loader.load_points3D()

db = sqlite3.connect(database_path)
kps = dict(
    (image_id, blob_to_array(data, np.float32, (-1, 6)))
    for image_id, data in db.execute(
        "SELECT image_id, data FROM keypoints"))

image_list = dict(
    (image_id, name)
    for image_id, name in db.execute(
        "SELECT image_id, name FROM images"))

print('%d 3D points in total'%len(loader.point3D_ids))

count = 0
for point3D_id in loader.point3D_ids:
    image_point = loader.point3D_id_to_images[point3D_id]
    print('%d: point3D_id = %d, appeared in %d images'%(loader.point3D_id_to_point3D_idx[point3D_id], point3D_id, len(image_point)))
    
    for image_id, point2D_idx in image_point:
        print('image_id = %d  point2D_idx = %d'%(image_id, point2D_idx))

        kp = kps[image_id][point2D_idx]
             
        ##### verification
        #print(kp)
        # image = loader.images[image_id]
        # k = list(image.point3D_ids).index(point3D_id)
        # print(image.points2D[k])
        ##### verification ends
        
        x , y, a11, a12, a21, a22 = kp
        scalex = np.sqrt(a11*a11 + a21*a21)
        scaley = np.sqrt(a12*a12 + a22*a22)
        scale  = (scalex + scaley) / 2.0
        orientation = np.arctan2(a21, a11)
        print('x = %.2f  y = %.2f  scale = %.2f  orientation = %.2f'%(x, y, scale, orientation))
        
        kp = cv2.KeyPoint(x = x, y = y, _size = scale, _angle = orientation)
        image_name = os.path.join(image_folder, image_list[image_id])
        img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        
        patch = extract_patch(img, kp)
        
        #cv2.circle(img, (x,y), 5, (255,0,0))
        if count <= 20:
            cv2.imwrite('%03d_%05d.jpg'%(count, point3D_id), patch)
            count += 1
        
        
        
        
        
        
        
        
        
