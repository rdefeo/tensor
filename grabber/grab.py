from cv2 import IMREAD_COLOR
from cv2 import imdecode
from cv2 import imwrite
from PIL import Image

import numpy as np

from os import mkdir
from os import path
from pymongo import MongoClient
import requests
import improc.features.preprocess as preprocess
from StringIO import StringIO
from collections import defaultdict
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

def is_corrupted(stream):
    try:
        img = Image.open(StringIO(stream.content))
        check = img.verify()
        del img
        return False
    except IOError:
        return True


def image_raw_preprocessing(img_stream):
    image_squared = None
    image_decoded = imdecode(np.fromstring(img_stream.content, np.uint8), flags=IMREAD_COLOR)
    if image_decoded is not None:
        try:
            image_autocropped = preprocess.autocrop(image_decoded)
        except AttributeError:
            return image_squared
        if image_autocropped is not None:
            image_scaled_max = preprocess.scale_max(image_autocropped)
            image_squared = preprocess.make_square(image_scaled_max)
    return image_squared


def update_product(data, prd_id):
    if any(data):
        collection.update(
            {"_id": prd_id},
            {"$set": data},
            upsert=False
        )


out_dir = 'out'

# Initializing MongoDB client
client = MongoClient('localhost', 27017)
test_db = client.jemboo
collection = test_db.shoes

if not (path.exists(out_dir)):
    mkdir(out_dir)
    print "Created output folder"

products = collection.find().batch_size(50)

print "Downloading images..."
print_interval = products.count() / 20
num_processed_products = 0
counters = defaultdict(int)


def process_image(set_data):
    try:
        global product_status
        url = img['url']
        img_id = img['_id']
        img_filename = out_dir + "/" + str(product_id) + "_" + str(img_id) + ".jpg"
        if "image_processed_status" not in img:  # check if image already exist

            img_data = requests.get(url, stream=True)
            if img_data.status_code == 200:
                if not is_corrupted(img_data):
                    processed_img = image_raw_preprocessing(img_data)
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

            set_data["images.%s.image_processed_status" % img_index] = image_status  # update img status
        else:
            image_status = "already_processed"

        return image_status
    except Exception as e:
        LOGGER.error(e)
        return "exception"


while counters["images_attempted"] < 400000:
    for prd_index, product in enumerate(products):
        product_status = "ok"
        product_id = product['_id']

        if prd_index % print_interval == 0:
            print(str(prd_index / print_interval * 4) + "% of products scanned")

        set_data = {}
        for img_index, img in enumerate(product['images']):
            counters["images_attempted"] += 1
            processed_image_status = process_image(set_data)
            counters[processed_image_status] += 1

        set_data["processed_status"] = product_status  # update_product_status
        update_product(set_data, product_id)
        num_processed_products += 1

LOGGER.info("100% of products scanned\n")
LOGGER.info("Total number of processed products: %i", num_processed_products)
LOGGER.info("Total number of processed images: %i", counters["images_attempted"])
LOGGER.info("Number of images failed to crop: %i", counters["autocropped_failed"])

LOGGER.info("Number of corrupted images: %i", counters["image_corrupted"])

