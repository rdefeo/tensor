from os import mkdir
from os import path
from pymongo import MongoClient
from cv2 import IMREAD_COLOR
from cv2 import waitKey
from cv2 import imshow
from cv2 import imdecode
from cv2 import imwrite
import requests
import numpy as np

import improc.features.preprocess as preprocess

out_dir = 'out'

# Initializing MongoDB client
client = MongoClient('localhost', 27017)
test_db = client.jemboo_test
collection = test_db.shoes

if not(path.exists(out_dir)):
    mkdir(out_dir)
    print "Created output folder"

products = collection.find()

print "Downloading images..."
prd = 0; print_interval = products.count()/20

for product in products:
    product_name = product['_id']

    if prd % print_interval == 0:
        print(str(prd / print_interval * 5) + "% of products scanned")

    product_img_num = -1
    for img in product['images']:
        url = img['url']
        product_img_num += 1

        # image fetch
        if not(path.isfile(out_dir + "/" + str(product_name) + "_" + str(product_img_num) + ".jpg")):  # check img exist
            img_data = requests.get(url, stream=True)
            if img_data.status_code == 200:

                # image processing
                image = imdecode(np.fromstring(img_data.content, np.uint8), flags=IMREAD_COLOR)
                image = preprocess.autocrop(image)
                image = preprocess.scale_max(image)
                image = preprocess.make_square(image)
                imshow("img", image)
                waitKey(0)

                image_name = out_dir + "/" + str(product_name) + "_" + str(product_img_num) + ".jpg"  # save image
                imwrite(image_name, image)

    prd += 1
