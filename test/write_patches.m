function write_patches( patches, patch_file_name )
    fid = fopen(patch_file_name, 'w');
    fwrite(fid, size(patches), 'int32');
    fwrite(fid, permute(patches, [3,2,1]), 'single');
    fclose(fid);
end

