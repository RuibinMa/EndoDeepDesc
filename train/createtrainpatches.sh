#!/bin/bash
data_dir=$1
if [[ ${data_dir: -1} == "/" ]]; then
	data_dir=${data_dir::-1}
fi
script=$2
filename="$data_dir/distribution.txt"
while read -r line
do
    stringarray=($line)
    sfm_id=${stringarray[0]}
    echo $sfm_id
    keyframe_id=${stringarray[1]}
    echo $keyframe_id
    N=${stringarray[2]}
    echo $N

    python $script \
    	--N $N \
    	--image_folder "$data_dir/keyframes/$keyframe_id" \
    	--database_path "$data_dir/databases/$keyframe_id/database.db" \
    	--sfm_folder "$data_dir/sfm_results/$sfm_id" \
    	--output_folder "$data_dir/output"

done < "$filename"