import numpy as np
import os
import sqlite3


#-------------------------------------------------------------------------------
# convert SQLite BLOBs to/from numpy arrays

def array_to_blob(arr):
    return np.getbuffer(arr)

def blob_to_array(blob, dtype, shape=(-1,)):
    return np.frombuffer(blob, dtype).reshape(*shape)


#-------------------------------------------------------------------------------
# convert to/from image pair ids

MAX_IMAGE_ID = 2**31 - 1

def get_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


def get_image_ids_from_pair_id(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    return (pair_id - image_id2) / MAX_IMAGE_ID, image_id2


#-------------------------------------------------------------------------------
# create table commands

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < 2147483647),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))"""

CREATE_INLIER_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS inlier_matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL)"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = \
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join([CREATE_CAMERAS_TABLE, CREATE_DESCRIPTORS_TABLE,
    CREATE_IMAGES_TABLE, CREATE_INLIER_MATCHES_TABLE, CREATE_KEYPOINTS_TABLE,
    CREATE_MATCHES_TABLE, CREATE_NAME_INDEX])


#-------------------------------------------------------------------------------
# add object commands

def add_camera(db, model, width, height, params, prior_focal_length=False,
        camera_id=None):
    # TODO: Parameter count checks
    params = np.asarray(params, np.float32)
    db.execute("INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
        (camera_id, model, width, height, array_to_blob(params),
         prior_focal_length))


def add_descriptors(db, image_id, descriptors):
    descriptors = np.asarray(descriptors, np.float32)
    db.execute("INSERT INTO descriptors VALUES (?, ?, ?, ?)",
        (image_id,) + descriptors.shape + (array_to_blob(descriptors),))


def add_image(db, name, camera_id, prior_q=np.zeros(4), prior_t=np.zeros(3),
        image_id=None):
    db.execute("INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (image_id, name, camera_id, prior_q[0], prior_q[1], prior_q[2],
         prior_q[3], prior_t[0], prior_t[1], prior_t[2]))


# config: defaults to fundamental matrix
def add_inlier_matches(db, image_id1, image_id2, matches, config=2):
    assert(len(matches.shape) == 2)
    assert(matches.shape[1] == 2)

    if image_id1 > image_id2:
        matches = matches[:,::-1]

    pair_id = get_pair_id(image_id1, image_id2)
    matches = np.asarray(matches, np.uint32)
    db.execute("INSERT INTO inlier_matches VALUES (?, ?, ?, ?, ?)",
        (pair_id,) + matches.shape + (array_to_blob(matches), config))


def add_keypoints(db, image_id, keypoints):
    assert(len(keypoints.shape) == 2)
    assert(keypoints.shape[1] in [2, 4, 6])

    keypoints = np.asarray(keypoints, np.float32)
    db.execute("INSERT INTO keypoints VALUES (?, ?, ?, ?)",
        (image_id,) + keypoints.shape + (array_to_blob(keypoints),))


# config: defaults to fundamental matrix
def add_matches(db, image_id1, image_id2, matches):
    assert(len(matches.shape) == 2)
    assert(matches.shape[1] == 2)

    if image_id1 > image_id2:
        matches = matches[:,::-1]

    pair_id = get_pair_id(image_id1, image_id2)
    matches = np.asarray(matches, np.uint32)
    db.execute("INSERT INTO matches VALUES (?, ?, ?, ?)",
        (pair_id,) + matches.shape + (array_to_blob(matches),))


#-------------------------------------------------------------------------------

def main(args):
    # TODO (True): delete this

    db = sqlite3.connect(args.database_path)

    #
    # check keypoints
    #

    kps = dict(
        (image_id, blob_to_array(data, np.float32, (-1, 2)))
        for image_id, data in db.execute(
            "SELECT image_id, data FROM keypoints"))

    print kps

    descs = dict(
        (image_id, blob_to_array(data, np.float32, (-1, 2)))
        for image_id, data in db.execute(
            "SELECT image_id, data FROM descriptors"))

    print descs
    db.close()

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--database_path", default='database.db')

    args = parser.parse_args()

    main(args)
