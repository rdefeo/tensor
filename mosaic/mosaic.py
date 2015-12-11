from collections import defaultdict
from os import listdir
from os.path import isfile, join, exists
from sys import exit
from pymongo import MongoClient
from bson.objectid import ObjectId

import numpy as np
import itertools as it

from cv2 import waitKey
from cv2 import imshow
from cv2 import imread

import improc.features.preprocess as preprocess
import logging

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
logging.info('Starting logger for image grabber.')

IMG_DIR = '../grabber/out'
MAX_NUM_ITERATIONS = 5
MAX_NUM_IMG = 5000


def chunk_list(seq, step):
    out = []
    last = 0.0
    while len(seq) - last >= step:
        out.append(seq[int(last):int(last + step)])
        last += step
    return out


def grouper(n, iterable, fillvalue=None):
    '''grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx'''
    args = [iter(iterable)] * n
    return it.izip_longest(fillvalue=fillvalue, *args)


def mosaic(w, imgs):
    '''Make a grid from images.

    w    -- number of grid columns
    imgs -- images (must have same size and format)
    '''
    imgs = iter(imgs)
    img0 = imgs.next()
    pad = np.zeros_like(img0)
    imgs = it.chain([img0], imgs)
    rows = grouper(w, imgs, pad)
    return np.vstack(map(np.hstack, rows))


def get_imgs_id(path):
    ids = []
    for i in listdir(path):
        product_id = i[:24]
        img_id = i[25:49]
        ids.append((product_id, img_id))
    return ids


def retrieve_img(img_id, product_id):
    db_products = collection.find({"_id": ObjectId(product_id)})
    for db_product in db_products:
        for db_img in db_product['images']:
            if str(db_img['_id']) == img_id:
                return db_img
        return None


def get_img_file_path(img_id, path):
    for i in listdir(path):
        if isfile(join(path, i)) and img_id in i:
            return join(path, i)
    return None


def get_imgs_block(img_ids, img_dir_path):
    images = []
    for img_id in img_ids:
        image = get_img_file_path(img_id, img_dir_path)
        if image is not None:
            cvimage = imread(image)
            cvimage = preprocess.scale_max(cvimage, 100, 100)
            images.append(cvimage)
        else:
            logging.error("Image %s is missing. Database and download folder are inconsistent.", img_id)
            images.append(np.ones((100, 100)))
    return images


# Initializing MongoDB client
client = MongoClient('localhost', 27017)
test_db = client.jemboo
collection = test_db.shoes

if not (exists(IMG_DIR)) or listdir(IMG_DIR) == []:
    print "Image folder does not exist or is empty."
    exit()

imgs_by_orientation = defaultdict(list)
local_imgs = get_imgs_id(IMG_DIR)

for (product, img) in local_imgs:
    img_db = retrieve_img(img, product)
    if img_db is not None:
        if 'x' in img_db and 'y' in img_db and 'z' in img_db:
            rpy = str(img_db['x']) + '_' + str(img_db['y']) + '_' + str(img_db['z'])
            imgs_by_orientation[rpy].append(str(img))
        else:
            imgs_by_orientation['invalid_orientation'].append(str(img))

del local_imgs

for orientation in imgs_by_orientation:
    LOGGER.info("%s : %i", orientation, len(imgs_by_orientation[orientation]))
    display_groups = chunk_list(imgs_by_orientation[orientation], 144)
    imgs_by_orientation[orientation] = display_groups

iteration = 1
while iteration <= MAX_NUM_ITERATIONS:
    for orientation in imgs_by_orientation:
        if iteration < len(imgs_by_orientation[orientation]):
            imgs = get_imgs_block(imgs_by_orientation[orientation][iteration], IMG_DIR)
            if imgs is not None:
                imgs_mosaic = mosaic(12, imgs)
                imshow(orientation, imgs_mosaic)
                waitKey(0)
    iteration += 1

