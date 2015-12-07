from cv2 import IMREAD_COLOR
from cv2 import imdecode
from cv2 import imwrite
from PIL import Image
import improc.features.preprocess as preprocess

import numpy as np

from os import mkdir
from os import path
from pymongo import MongoClient
import requests
from StringIO import StringIO
from collections import defaultdict
import logging

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.ERROR)
logging.info('Starting logger for image grabber.')

OUT_DIR = 'out'
MAX_NUM_PRODUCTS = 400000

EXCLUDED_IMAGE_ORIENTATIONS = [(270, 90),
                               (270, 270),
                               (180, 180),
                               (90, 180)]

def is_corrupted(stream):
    """Check if the data streamed by url request is a valid image
    :param stream: raw data from url
    :return bool: whether it is a valid image or not
    """
    try:
        img = Image.open(StringIO(stream.content))
        check = img.verify()
        del img
        return False
    except IOError as e:
        LOGGER.error(e)
        return True


def is_valid_orientation(img_dict):
    """Check that the shoe is not in a forbidden orientation.
    :param img_dict: image dictionary to test
    :return: false if orientation (pitch, yaw) is in EXCLUDED_IMAGE_ORIENTATIONS
    """
    if ('y' in img_dict and 'z' in img_dict and
            (img_dict['y'], img_dict['z']) in EXCLUDED_IMAGE_ORIENTATIONS):
        return False
    else:
        return True


def image_raw_preprocessing(img_stream):
    """Decode the raw data from url into an image, crops and makes square
    :param img_stream: raw data from url
    :return: processed img
    """
    image_squared = None
    image_decoded = imdecode(np.fromstring(img_stream.content, np.uint8), flags=IMREAD_COLOR)
    if image_decoded is not None:
        try:
            image_autocropped = preprocess.autocrop(image_decoded)
        except AttributeError as e:
            LOGGER.error(e)
            return image_squared
        if image_autocropped is not None:
            image_scaled_max = preprocess.scale_max(image_autocropped)
            image_squared = preprocess.make_square(image_scaled_max)
    return image_squared


def update_product(data, prd_id):
    """Updates a product in the collection with data stored in a dictionary
    :param data: dictionary of new data
    :param prd_id: id of the product to be updated
    :return:
    """
    if any(data):
        collection.update(
            {"_id": prd_id},
            {"$set": data},
            upsert=False
        )


def process_image(img, data):
    """Checks if an image in the db has not been processed and has a valid
       orientation, then processes the image.
    :param img: A dictionary from the database which stores data about the
                image to be processed.
    :param data: A dictionary where to store image status for future update.
    :return: The image status.
    """
    try:
        url = img['url']
        img_id = img['_id']
        img_filename = OUT_DIR + "/" + str(product_id) + "_" + str(img_id) + ".jpg"
        if "image_processed_status" not in img and is_valid_orientation(img):  # check if image already exist
            img_raw_data = requests.get(url, stream=True)
            if img_raw_data.status_code == 200:
                if not is_corrupted(img_raw_data):
                    processed_img = image_raw_preprocessing(img_raw_data)
                    if processed_img is not None:
                        imwrite(img_filename, processed_img)  # save image
                        image_status = "ok"
                    else:
                        image_status = "autocropped_failed"

                else:
                    image_status = "image_corrupted"

            else:
                print("Unable to retrieve image " + str(img_index) + "/" + str(prd_index))
                image_status = "http_fail"

            data["images.%s.image_processed_status" % img_index] = image_status  # update img status
        else:
            image_status = "already_processed"

        return image_status
    except Exception as e:
        LOGGER.error(e)
        return "exception"


# Initializing MongoDB client
client = MongoClient('localhost', 27017)
test_db = client.jemboo
collection = test_db.shoes

if not (path.exists(OUT_DIR)):
    mkdir(OUT_DIR)
    print "Created output folder"

products = collection.find().batch_size(50)

print "Downloading images..."
print_interval = min((products.count(), MAX_NUM_PRODUCTS)) / 50
counters = defaultdict(int)

for prd_index, product in enumerate(products):
    if counters["images_attempted"] < MAX_NUM_PRODUCTS:
        product_status = "ok"
        product_id = product['_id']

        if prd_index % print_interval == 0:
            print(str(prd_index / print_interval * 2) + "% of products scanned")

        db_changes = {}
        for img_index, img_data in enumerate(product['images']):
            counters["images_attempted"] += 1
            processed_image_status = process_image(img_data, db_changes)
            counters[processed_image_status] += 1
            if processed_image_status is not "ok":
                product_status = "failed"

        db_changes["processed_status"] = product_status  # update_product_status
        update_product(db_changes, product_id)
        counters["products_attempted"] += 1
    else:
        break

LOGGER.info("100% of products scanned.")
LOGGER.info("Total number of processed products: %i", counters["products_attempted"])
LOGGER.info("Total number of processed images: %i", counters["images_attempted"])
LOGGER.info("Number of images failed to crop: %i", counters["autocropped_failed"])
LOGGER.info("Number of corrupted images: %i", counters["image_corrupted"])

