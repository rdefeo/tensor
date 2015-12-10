from os import listdir
from os.path import isfile, join, exists
from sys import exit
from pymongo import MongoClient

import numpy as np
import itertools as it

from cv2 import waitKey
from cv2 import imshow
from cv2 import imread

from orientations import orientations
import improc.features.preprocess as preprocess
import logging

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.ERROR)
logging.info('Starting logger for image grabber.')

IMG_DIR = '../grabber/out'
MAX_NUM_PRODUCTS = 100000


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


def get_img_file_path(img_id, path):

    for i in listdir(path):
        if isfile(join(path, i)) and img_id in i:
            return join(path, i)
        else:
            return None


def get_imgs_block(img_ids, img_dir_path):

    images = []
    for img_id in img_ids[-144:]:
        image = get_img_file_path(img_id, img_dir_path)
        if image is not None:
            cvimage = imread(image)
            imshow("sederino", imshow(cvimage))
            cvimage = preprocess.scale_max(cvimage, 100, 100)
            images.append(cvimage)
        else:
            logging.error("Image %s is missing. Database and download folder are inconsistent.", img_id)
            return None
    return images

imgs_by_orientation = {
                        'SHOE_ORIENTATION_1': [],
                        'SHOE_ORIENTATION_2': [],
                        'SHOE_ORIENTATION_3': [],
                        'SHOE_ORIENTATION_4': [],
                        'SHOE_ORIENTATION_5': [],
                        'SHOE_ORIENTATION_6': [],
                        'SHOE_ORIENTATION_7': [],
                        'SHOE_ORIENTATION_8': [],
                        'SHOE_ORIENTATION_9': [],
                        'SHOE_ORIENTATION_10': [],
                        'SHOE_ORIENTATION_11': [],
                        'SHOE_ORIENTATION_12': [],
                        'SHOE_ORIENTATION_13': [],
                        'SHOE_ORIENTATION_14': [],
                        'SHOE_ORIENTATION_15': [],
                        'INVALID_ORIENTATION': [],
                        }


# Initializing MongoDB client
client = MongoClient('localhost', 27017)
test_db = client.jemboo
collection = test_db.shoes

if not (exists(IMG_DIR)) or listdir(IMG_DIR) == []:
    print "Image folder does not exist or is empty."
    exit()

products = collection.find().batch_size(30)
orientation_show_counter = 1

for prd_index, product in enumerate(products):
    for img_index, img in enumerate(product['images']):

        img_id = img['_id']
        if 'image_processed_status' in img and img['image_processed_status'] == 'ok':
            if 'x' in img and 'y' in img and 'z' in img:
                rpy = str(img['x']) + '_' + str(img['y']) + '_' + str(img['z'])
                orientation = orientations[rpy]
                imgs_by_orientation[orientation].append(img_id)

            else:
                imgs_by_orientation['INVALID_ORIENTATION'].append(img_id)

        current_orientation_show = 'SHOE_ORIENTATION_' + str(orientation_show_counter)
        if (len(imgs_by_orientation[current_orientation_show]) % 144 == 0 and
                len(imgs_by_orientation[current_orientation_show]) % 144 != 0):
            imgs = get_imgs_block(imgs_by_orientation[current_orientation_show], IMG_DIR)
            imgs_mosaic = mosaic(12, imgs)
            imshow(current_orientation_show, imgs_mosaic)
            waitKey(0)
            if orientation_show_counter != 15:
                orientation_show_counter += 1
            else:
                orientation_show_counter = 1
