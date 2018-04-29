function match(DATASET_PATH)

% Copyright 2017: Johannes L. Schoenberger <jsch at inf.ethz.ch>


%% Set the pipeline parameters.

% Whether to run matching on GPU.
MATCH_GPU = gpuDeviceCount() > 0;
fprintf('MATCH_GPU: %d\n', MATCH_GPU);

% Number of images to match in one block.
MATCH_BLOCK_SIZE = 50;

% Maximum distance ratio between first and second best matches.
MATCH_MAX_DIST_RATIO = 0.8;

% Mnimum number of matches between two images.
MIN_NUM_MATCHES = 15;

%% Setup the pipeline environment.

IMAGE_PATH = fullfile(DATASET_PATH, 'images');
KEYPOINT_PATH = fullfile(DATASET_PATH, 'keypoints');
DESCRIPTOR_PATH = fullfile(DATASET_PATH, 'descriptors');
MATCH_PATH = fullfile(DATASET_PATH, 'matches');
DATABASE_PATH = fullfile(DATASET_PATH, 'database.db');

%% Create the output directories.

if ~exist(KEYPOINT_PATH, 'dir')
    mkdir(KEYPOINT_PATH);
end
if ~exist(DESCRIPTOR_PATH, 'dir')
    mkdir(DESCRIPTOR_PATH);
end
if ~exist(MATCH_PATH, 'dir')
    mkdir(MATCH_PATH);
end

%% Extract the image names and paths.

image_files = dir(IMAGE_PATH);
num_images = length(image_files) - 2;
image_names = cell(num_images, 1);
image_paths = cell(num_images, 1);
keypoint_paths = cell(num_images, 1);
descriptor_paths = cell(num_images, 1);
for i = 3:length(image_files)
    image_name = image_files(i).name;
    image_names{i-2} = image_name;
    image_paths{i-2} = fullfile(IMAGE_PATH, image_name);
    keypoint_paths{i-2} = fullfile(KEYPOINT_PATH, [image_name '.bin']);
    descriptor_paths{i-2} = fullfile(DESCRIPTOR_PATH, [image_name '.bin']);
end

%% Compute the keypoints and descriptors.

%feature_extraction

%% Match the descriptors.
%
%  NOTE: - You must exhaustively match Fountain, Herzjesu, South Building,
%          Madrid Metropolis, Gendarmenmarkt, and Tower of London.
%        - You must approximately match Alamo, Roman Forum, Cornell.

if num_images < 2000
    exhaustive_matching
else
    % TODO: Change this to where you saved the vocabulary tree that was
    %       built using the features of the Oxford5k dataset.
    VOCAB_TREE_PATH = 'path/to/Oxford5k/vocab-tree.bin';
    approximate_matching
end
