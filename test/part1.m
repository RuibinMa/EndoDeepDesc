clear all; close all; clc;
PATCH_RADIUS = 32;
VLFEAT_PATH = '/home/ruibinma/vlfeat';
run(fullfile(VLFEAT_PATH, 'toolbox/vl_setup'));
TESTSCRIPT_PATH = '/home/ruibinma/EndoDeepDesc/test';

data_folder = '../data';
image_folder = fullfile(data_folder, 'images');
keypoints_folder = fullfile(data_folder, 'keypoints');
sfm_folder = fullfile(data_folder, 'sfm_results');
patches_folder = fullfile(data_folder, 'patches');
descriptor_folder = fullfile(data_folder, 'descriptors');
database_path = fullfile(data_folder, 'database.db');


if exist(keypoints_folder, 'dir')
    rmdir(keypoints_folder, 's')
end
mkdir(keypoints_folder)

if exist(patches_folder, 'dir')
    rmdir(patches_folder, 's')
end
mkdir(patches_folder)

if exist(descriptor_folder, 'dir')
    rmdir(descriptor_folder, 's')
end
mkdir(descriptor_folder)

cmd = ['python colmap_export.py --database_path ', database_path, ...
        ' --output_path ', keypoints_folder];
system(cmd);

image_files = dir(image_folder);
image_files = image_files(3:end);
num_images = length(image_files);
image_names = cell(num_images, 1);
image_paths = cell(num_images, 1);
keypoint_paths = cell(num_images, 1);
descriptor_paths = cell(num_images, 1);

for i=1:num_images
    image_names{i} = image_files(i).name;
    image_paths{i} = fullfile(image_folder, image_files(i).name);
    keypoint_paths{i} = fullfile(keypoints_folder, [image_files(i).name, '.bin']);
    descriptor_paths{i} = fullfile(descriptor_folder, [image_files(i).name, '.bin']);
end

% pool = gcp('nocreate');
% if isempty(pool)
%     pool = parpool(maxNumCompThreads());
% end
