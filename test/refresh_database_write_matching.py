# Import the features and matches into a COLMAP database.
#
# Copyright 2017: Johannes L. Schoenberger <jsch at inf.ethz.ch>

from __future__ import print_function, division

import os
import glob
import argparse
import sqlite3
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    args = parser.parse_args()
    return args


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        return 2147483647 * image_id2 + image_id1
    else:
        return 2147483647 * image_id1 + image_id2


def read_matrix(path, dtype):
    with open(path, "rb") as fid:
        shape = np.fromfile(fid, count=2, dtype=np.int32)
        matrix = np.fromfile(fid, count=shape[0] * shape[1], dtype=dtype)
    return matrix.reshape(shape)


def main():
    args = parse_args()

    connection = sqlite3.connect(os.path.join(args.dataset_path, "database.db"))
    cursor = connection.cursor()

#    cursor.execute("DELETE FROM keypoints;")
    cursor.execute("DELETE FROM descriptors;")
    cursor.execute("DELETE FROM matches;")
    cursor.execute("DELETE FROM inlier_matches;")
    connection.commit()

    images = {}
    cursor.execute("SELECT name, image_id FROM images;")
    for row in cursor:
        images[row[0]] = row[1]

#     for image_name, image_id in images.items():
#         print("Importing features for", image_name)
#         keypoint_path = os.path.join(args.dataset_path, "keypoints",
#                                      image_name + ".bin")
#         keypoints = read_matrix(keypoint_path, np.float32)
#         descriptor_path = os.path.join(args.dataset_path, "descriptors",
#                                      image_name + ".bin")
#         descriptors = read_matrix(descriptor_path, np.float32)
#         assert keypoints.shape[1] == 6
#         assert keypoints.shape[0] == descriptors.shape[0]
# 
#         keypoints_str = np.getbuffer(keypoints)
#         cursor.execute("INSERT INTO keypoints(image_id, rows, cols, data) "
#                        "VALUES(?, ?, ?, ?);",
#                        (image_id, keypoints.shape[0], keypoints.shape[1],
#                         keypoints_str))
#         connection.commit()

    image_pairs = []
    f = open(os.path.join(args.dataset_path, "matches.txt"), "w")
    for match_path in glob.glob(os.path.join(args.dataset_path,
                                             "matches/*---*.bin")):
        image_name1, image_name2 = \
            os.path.basename(match_path[:-4]).split("---")
        image_pairs.append((image_name1, image_name2))
        print("Importing matches for", image_name1, "---", image_name2)
        image_id1, image_id2 = images[image_name1], images[image_name2]
        image_pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = read_matrix(match_path, np.uint32)
        assert matches.shape[1] == 2
        
        f.write('%s %s\n'%(image_name1, image_name2))
        for k in range(len(matches)):
            f.write('%d %d\n'%(matches[k,0], matches[k,1]))
        f.write('\n')
    f.close()

    cursor.close()
    connection.close()


if __name__ == "__main__":
    main()
