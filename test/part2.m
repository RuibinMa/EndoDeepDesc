
for i = 1:num_images
    fprintf('Computing features for %s [%d/%d]', ...
            image_names{i}, i, num_images);

    if exist(keypoint_paths{i}, 'file') ...
            && exist(descriptor_paths{i}, 'file')
        fprintf(' -> skipping, already exist\n');
        continue;
    end

    tic;

    % Read the image for keypoint detection, patch extraction and
    % descriptor computation.
    image = imread(image_paths{i});
    if ismatrix(image)
        image = single(image);
    else
        image = single(rgb2gray(image));
    end

    % TODO: Replace this with your keypoint detector. The resultding
    %       keypoints matrix should have shape N x 4, where each row
    %       contains the keypoint properties x, y, scale, orientation.
    %       Note that only x and y are necessary for the reconstruction
    %       benchmark while scale and orientation are only used for
    %       extracting local patches around each detected keypoint.
    %       If you implement your own keypoint detector and patch
    %       extractor, then you can simply set scale and orientation to 0.
    %       Here, we simply detect SIFT keypoints using VLFeat.
    % keypoints = vl_sift(image)';
    % write_keypoints(keypoint_paths{i}, keypoints);
    keypoints = read_keypoints(keypoint_paths{i});

    % Extract the local patches for all keypoints.
    patches = extract_patches(image, keypoints, PATCH_RADIUS);
    %patch_file_name = fullfile(patches_folder, [image_names{i},'.mat']);
    %save(patch_file_name, 'patches')
    patch_file_name = fullfile(patches_folder, [image_names{i},'.bin']);
    write_patches(patches, patch_file_name)

%     % TODO: Compute the descriptors for the extracted patches. Here, we
%     %       simply compute SIFT descriptors for all patches using VLFeat.
%     cmd = ['python ', fullfile(TESTSCRIPT_PATH, 'main.py'), ...
%             ' ', patch_file_name, ...
%             ' ', image_names{i}, ...
%             ' ', descriptor_paths{i}, ...
%             ' --checkpoint ', fullfile(TESTSCRIPT_PATH, 'testmodel.pth')];
%     system(cmd);
%     
%     cmd = ['rm ', patch_file_name];
%     system(cmd);

    fprintf(' in %.3fs\n', toc);
end
