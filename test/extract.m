function patches = extract(image_path, keypoint_path, PATCH_RADIUS)
    PATCH_RADIUS = 32;
    image = imread(image_path);
    if ismatrix(image)
        image = single(image);
    else
        image = single(rgb2gray(image));
    end
    
    keypoints = read_keypoints(keypoint_path);
    patches = extract_patches(image, keypoints, PATCH_RADIUS);
    
    fprintf('verification key1: %.6f\n', patches(201,21,51));
end