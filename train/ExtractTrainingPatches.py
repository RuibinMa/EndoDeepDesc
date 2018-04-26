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

def extract_patch_sz(img, sz, xy):
    sub = cv2.getRectSubPix(img, (int(round(sz)), int(round(sz))), xy)
    res = cv2.resize(sub, (input_sz, input_sz))
    return res

data_folder = '../data/sample'
image_folder = os.path.join(data_folder, 'images')
sfm_folder = os.path.join(data_folder, 'sfm_results')
output_folder = os.path.join(data_folder, 'output')
database_path = os.path.join(data_folder, 'database.db')

if os.path.exists(output_folder):
    rmtree(output_folder)
os.mkdir(output_folder)

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
# only one camera, camera intrinsics
camera = loader.cameras[1]
fx, fy, cx, cy = camera.fx, camera.fy, camera.cx, camera.cy
K = np.zeros(shape=(3,3))
K[0,0] = fx
K[1,1] = fy
K[0,2] = cx
K[1,2] = cy
K[2,2] = 1

#output grid settings
grid_row = 0
grid_col = 0
grid_id = 0
grid = np.zeros(shape=(grid_sz*input_sz, grid_sz*input_sz), dtype=np.uint8)
info_file = open(os.path.join(output_folder, 'info.txt'), 'w')
position_file = open(os.path.join(output_folder, 'position%04d.txt'%grid_id), 'w')

#

matches = RandomDict()
possible_n_matches = 0
nonmatches = {}

def init_choice_set(image_point):
    choices = RandomDict()
    n = len(image_point)
    for i in range(n):
        for j in range(i+1, n):
            image_id1, _ = image_point[i]
            image_id2, _ = image_point[j]
            choices[len(choices)] = (image_id1, image_id2)
    return choices

# extract according to the z-value the 3D point in each picture
count_debug = 0
count = 0
track_len = 0

point3D_id_image_id_to_patch_id = {}
patch_id_to_point3D_id_image_id = {}

for point3D_id in loader.point3D_ids:
    image_point = loader.point3D_id_to_images[point3D_id]
    
    
    #========================================debug
    image_point_unique = []
    unique_set = set() # this is purely to deal with the bug in colmap
    for image_id, point2D_idx in image_point:
        if image_id not in unique_set:
            unique_set.add(image_id)
            image_point_unique.append((image_id, point2D_idx))
        else:
            count_debug += 1
    image_point = image_point_unique
    #========================================debug     
    
       
    track_len += len(image_point)
    print('\n%d: point3D_id = %d, appeared in %d images'%(loader.point3D_id_to_point3D_idx[point3D_id], point3D_id, len(image_point)))
    z_values = []
    points2D_proj = []
    for image_id, point2D_idx in image_point:
        print('image_id = %d  point2D_idx = %d'%(image_id, point2D_idx))

        kp = kps[image_id][point2D_idx]
             
        ##### verification
#         print(kp)
#         image = loader.images[image_id]
#         k = list(image.point3D_ids).index(point3D_id)
#         print(image.points2D[k])
        ##### verification ends
        
        # reproject 3D points
        point3D_idx = loader.point3D_id_to_point3D_idx[point3D_id]
        point3D = loader.points3D[point3D_idx]
        image = loader.images[image_id]
        
        R = image.R().T # you need to transpose R, which is different from the COMP776 slides
        C = image.C()
        RTC = np.dot(R.T, C)
        RTC = np.expand_dims(RTC, axis=1)
        P = np.concatenate([R.T, -RTC], axis = 1)
        point3D1 = np.ones(shape=(4,))
        point3D1[0:3] = point3D
        point3D_cam = np.dot(P, point3D1)
        point2D1_proj = np.dot(K, point3D_cam / point3D_cam[2])
        z_values.append(point3D_cam[2])
        points2D_proj.append((point2D1_proj[0], point2D1_proj[1]))
        print('point2D detected: %.4f, %.4f '%(kp[0], kp[1]))
        print('        after BA: %.4f, %.4f '%(point2D1_proj[0], point2D1_proj[1]))
        print('z-value(depth)  : %.4f'%point3D_cam[2])
    
    z_max = np.max(z_values)
    print('largest distance: %.4f (extract 64x64 patch under this scale)'%z_max)
    assert(len(points2D_proj) == len(image_point))
    for i in range(len(points2D_proj)):
        x, y = points2D_proj[i]
        image_id, point2D_idx = image_point[i]
        image_name = os.path.join(image_folder, image_list[image_id])
        img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        
        # extract the information of the DoG keypoints for writing to interest.txt
        #kp = kps[image_id][point2D_idx]
        #x_kp , y_kp, a11, a12, a21, a22 = kp
        #scalex = np.sqrt(a11*a11 + a21*a21)
        #scaley = np.sqrt(a12*a12 + a22*a22)
        #scale  = (scalex + scaley) / 2.0
        #orientation = np.arctan2(a21, a11)
        
        # crop the patch
        crop_sz = input_sz * z_max / z_values[i] 
        patch = extract_patch_sz(img=img, sz=crop_sz, xy=(x,y))
        point3D_id_image_id_to_patch_id[(point3D_id, image_id)] = count
        patch_id_to_point3D_id_image_id[count] = (point3D_id, image_id)
        # write metadata to text files
        info_file.write('%d 0\n'%point3D_id)
        
        # cv2.circle(img, (int(round(x)), int(round(y))), 5, (255,0,0))
        #if count <= grid_sz * grid_sz:
        #    cv2.imwrite('%03d_%04d_%05d.jpg'%(count, image_id, point3D_id), patch)
        count += 1
        
        grid[grid_row*input_sz:(grid_row+1)*input_sz, grid_col*input_sz:(grid_col+1)*input_sz] = patch
        # record the 3D position to determine whether consider as non-match
        point3D_idx = loader.point3D_id_to_point3D_idx[point3D_id]
        point3D = loader.points3D[point3D_idx]
        position_file.write('(%d %d) %04d %05d %.2f %.2f %.2f\n'%(grid_row, grid_col, image_id, point3D_id, point3D[0], point3D[1], point3D[2]))
        
        grid_col += 1
        # create a new grid if the previous one is full
        if grid_row == grid_sz - 1 and grid_col == grid_sz:
            cv2.imwrite(os.path.join(output_folder, 'patches%04d.bmp'%grid_id), grid)
            grid = np.zeros(shape=(grid_sz*input_sz, grid_sz*input_sz), dtype=np.uint8)
            position_file.close()
            grid_row = 0
            grid_col = 0
            grid_id += 1
            position_file = open(os.path.join(output_folder, 'position%04d.txt'%(grid_id)), 'w')
        elif grid_col == grid_sz:
            assert(grid_row < grid_sz-1)
            grid_row += 1
            grid_col = 0         
    
#     print(image_point)
#     if point3D_id == 4740:
#         kp1 = kps[20][653]
#         kp2 = kps[20][2170]
#         print(kp1)
#         print(kp2)
#         cv2.imwrite(os.path.join(output_folder, 'patches%04d.bmp'%grid_id), grid)
#         position_file.close()
    matches[point3D_id] = init_choice_set(image_point)
    print('initialized %d choices for matches'%len(matches[point3D_id])) 
    assert(len(matches[point3D_id]) == (len(image_point)*(len(image_point)-1))/2)
    possible_n_matches += len(matches[point3D_id])
       
if grid_row != 0 or grid_col != 0:
    cv2.imwrite(os.path.join(output_folder, 'patches%04d.bmp'%grid_id), grid)
    position_file.close()
else:
    position_file.close()
    os.remove(position_file)
            


# # extract according to sift scale
# count = 0
# for point3D_id in loader.point3D_ids:
#     image_point = loader.point3D_id_to_images[point3D_id]
#     print('\n%d: point3D_id = %d, appeared in %d images'%(loader.point3D_id_to_point3D_idx[point3D_id], point3D_id, len(image_point)))
#      
#     for image_id, point2D_idx in image_point:
#         print('image_id = %d  point2D_idx = %d'%(image_id, point2D_idx))
#  
#         kp = kps[image_id][point2D_idx]
#               
#         ##### verification
# #         print(kp)
# #         image = loader.images[image_id]
# #         k = list(image.point3D_ids).index(point3D_id)
# #         print(image.points2D[k])
#         ##### verification ends
#          
#         # compute the opencv KeyPoint
#         x , y, a11, a12, a21, a22 = kp
#         scalex = np.sqrt(a11*a11 + a21*a21)
#         scaley = np.sqrt(a12*a12 + a22*a22)
#         scale  = (scalex + scaley) / 2.0
#         orientation = np.arctan2(a21, a11)
#         print('x = %.2f  y = %.2f  scale = %.2f  orientation = %.2f'%(x, y, scale, orientation))
#            
#         kp = cv2.KeyPoint(x = x, y = y, _size = scale, _angle = orientation)
#         image_name = os.path.join(image_folder, image_list[image_id])
#         img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
#            
#         patch = extract_patch(img, kp)
#          
#  
#  
#         #cv2.circle(img, (x,y), 5, (255,0,0))
# #         if count <= 20:
# #             cv2.imwrite('%03d_%05d.jpg'%(count, point3D_id), patch)
# #         count += 1        


def extract_patch_from_grid(patch_id):
    grid_id = patch_id / (grid_sz * grid_sz)
    in_grid_n = patch_id % (grid_sz * grid_sz)
    grid_row = in_grid_n / grid_sz
    grid_col = in_grid_n % grid_sz
    
    img = cv2.imread(os.path.join(output_folder, 'patches%04d.bmp'%grid_id), cv2.IMREAD_GRAYSCALE)
    
    patch = img[grid_row*input_sz:(grid_row+1)*input_sz, grid_col*input_sz:(grid_col+1)*input_sz]
    return patch
    

# establish matches
assert(len(matches) == len(loader.point3D_ids))

N_matches = 100000
max_n_try = 50000
n = 0

count_match = 0
while(n < min(max_n_try, possible_n_matches)):
    rand_point3D_id, rand_pair_choices = matches.random_item()
    assert(len(rand_pair_choices) > 0)
    rand_pair_key, rand_pair = rand_pair_choices.random_item()
    rand_pair_choices.pop(rand_pair_key)
    if len(rand_pair_choices) == 0:
        matches.pop(rand_point3D_id)
    n += 1
    image_id1 = rand_pair[0]
    image_id2 = rand_pair[1]
    patch_id1 = point3D_id_image_id_to_patch_id[(rand_point3D_id, image_id1)]
    patch_id2 = point3D_id_image_id_to_patch_id[(rand_point3D_id, image_id2)]
    print('%d/%d %d (img:%d[patch:%d] img:%d[patch:%d])'%(
        n, possible_n_matches, rand_point3D_id, image_id1, patch_id1, image_id2, patch_id2))
    
    # visualize the matchings
    if n % 900 == 0:
        img = np.zeros(shape=(input_sz, input_sz*2), dtype=np.uint8)
        patch1 = extract_patch_from_grid(patch_id1)
        patch2 = extract_patch_from_grid(patch_id2)
        img[0:input_sz,0:input_sz] = patch1
        img[0:input_sz, input_sz:input_sz*2] = patch2        
        cv2.imwrite('match%04d.jpg'%n, img)

assert(count == len(patch_id_to_point3D_id_image_id))
assert(count == len(point3D_id_image_id_to_patch_id))
print('============================\n%d patches in total'%count)
print('%d 3D points'%len(loader.point3D_ids))
print('%.2f average track length'%(float(track_len) / float(len(loader.point3D_ids))))        
info_file.close()
        
from numpy.linalg import norm
xyz_std = np.std(loader.points3D, axis = 0)
print('positional std = %.4f'%norm(xyz_std))
print('%d repeated images in point3D_id_to_images'%count_debug)




    
