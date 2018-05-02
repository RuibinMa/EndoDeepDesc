'''
Created on May 1, 2018

@author: ruibinma
'''
import os
import numpy as np
from pycolmap.scene_manager import SceneManager
sfm_folder = '/media/ruibinma/RUIBIN/trainset/sfm_results'
N_target = 250000
n_cases = len(os.listdir(sfm_folder))
case_list = []
ori_list = []

case_record = '/media/ruibinma/RUIBIN/trainset/sfm_record.txt'
record_file = open(case_record, 'r')
for i in range(n_cases):
    line = record_file.readline().split()
    print(line)
    case_list.append(line[0])
    ori_list.append(line[1][1:3])
record_file.close()

distribution_file = open('/media/ruibinma/RUIBIN/trainset/distribution.txt', 'w')
n_matches_list = []
distribution = []
total_n_matches = 0
for i in range(n_cases):
    loader = SceneManager(os.path.join(sfm_folder, case_list[i]))
    loader.load_cameras()
    loader.load_images()
    loader.load_points3D()
    
    possible_n_matches = 0
    for point3D_id, images in loader.point3D_id_to_images.iteritems():
        possible_n_matches += len(images)*(len(images)-1)/2
    
    print('possible_n_matches: %d'%possible_n_matches)
    n_matches_list.append(possible_n_matches)
    total_n_matches += possible_n_matches

remain = N_target
for i in range(n_cases):
    if i < n_cases-1:
        n = np.int(round(np.float(N_target) / np.float(total_n_matches) * np.float(n_matches_list[i])))
        distribution.append(n)
        remain -= n
    else:
        n = remain
        distribution.append(n)
        remain = 0
    
    assert(n <= n_matches_list[i])
    distribution_file.write('%s %s %d\n'%(case_list[i], ori_list[i], n))

distribution_file.close()
print(distribution)
    
    
    
    
    
    