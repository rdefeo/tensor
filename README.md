# Tensorflow-based shoe orientation classificator

This repository contains some modules to fetch, process and learn from the images listed in a mongodb database. In particular, we deal with shoes (**products**) and the images which depicts them. These images are initially labelled with their orientation expressed in **roll**, **pitch** and **yaw**. After a training phase, we want to obtain a model to automatically assign a label to each newly acquired image. This is done through the following packages:

  - **Grabber** - it uses pymongo to query *jemboo.shoes* collection, then download, preprocess and save e images to a folder
  - **Learner** -  learn from the images in a folder and the labes stored in jemboo.shoes collection

**Mosaic** is an extra testing utility which iterates over the downloaded images and shows grids of 12x12 images tagged with the same orientation. It can be used to visually check the quality of the dataset (**image - label coherency**).

### Version
1.0.0

### Tech

We use the following modules:

* [Numpy] - efficient numerical and array calculations in Python
* [Requests] - seamless integration with web services
* [OpenCV] - open source computer vision library
* [Pymongo] - MongoDB interface written in Python
* [Image_processing] - implements several preprocessing functions based on opencv and scipy

The **official project repository** can be found [here].

### Installation

To work, image_processing module must be installed on the system.

```sh
cd image_processsing_dir
install sudo python setup.py install
```

### Grabber

This script connects to *jemboo.shoes* collection and downloads to a folder all images that:

* are not already present in the folder
* are not flagged with **_image_status_** in the db (not previously processed)

Downloaded images are squared ad centered before being saved. They are being saved with such naming convention:
**_productID_imageID_**.

It is possibe to set three parameters:

* **OUT_DIR** - is the directory where to save images
* **MAX_NUM_PRODUCTS** - maximum number of products to process (*warning*: not images)
* **EXCLUDED_IMAGE_ORIENTATIONS** - list of tuples describing the images rpy labels to ignore (e.g. [(270, 90),
                               (270, 270)], granted roll = 0)

Processed images **_image_status_** flag is updated so that it is possible to resume a grabber session later. 
**_image_status_** flag can be set to the following values:

* *ok*
* *autocropped_failed*
* *image_corupted*
* *http_fail*

A **_product_status_** flag is updated when all images belonging to a product have been processed. it is set to *ok* if and only if all images belonging to that product have **_image_status_** = *ok*.

### Learner

The learner module purpose is to actually train a model to classify images by their orientation. The module process the images in a directory, format them in a suitable dataset form, and train the model against them. Once it is done, it saves a tensorflow checkpoint with the obtained model parameters, which can be used to categorize images or as a starting point for further training.

The following parameters can be set by the user before execution:

* **TRAINING_SET_DIR** - the directory containing the training images
* **TEMP_DATASET_DIR** - the directory where to save intermediate products such as the datasets and the dict of already processed images
* **TF_LOG_DIR** - the directory where to save logging data (they can be visualized through *TensorBoard*)
* **MODEL_DIR** - the directory where to save the resulting checkpoint


* **TRAIN_SET_SIZE** - number of images in *TRAINING_SET_DIR* to be reserved for training and validation
* **BATCH_SIZE** and **MAX_ITERATIONS** - to ensure computational efficiency, training is performed in batches. When all images have been processed once, they are shuffled and new batches are generated. These two parameters sets the size of the batches and how many batches to process. Increasing *MAX_ITERATIONS* usually leads to better result but depending on the batch and train set size may also lead to overfitting in our experience (to verify). A good guess at the moment is:

    > MAX_ITERATIONS * BATCH_SIZE / (increment of TEST_SET_SIZE) ~ 18-20

*Warning*: the following two parameter are not supposed to be changed in further retrainings so they should be set straight away to the final desired value. This is to enforce the concept that changing these paramentes would void the validity of any result comparison.

* **TEST_SET_SIZE** - number of images to process and reserve for testing. 
* **VALIDATION_PERCENTAGE** - percentage of the *TRAIN_SET* to be reserved for validation.

The last two parameters sets the operating mode of the script:

* **DATASETS_AVAILABLE** - set to *True* if you want to use the datasets in *TEMP_DATASETS_DIR* instead of generating new ones. It is useful when training fails or crashes.
* **RETRAIN_SESSION** - set to *True* if you want to further train a previosly generated model starting from its checkpoint. In this case you also have to set a ne *TRAIN_SET_SIZE* which will indicate the *final* size of the set (not the increment). Do not change *TEST_SET_SIZE* and *VALIDATION_PERCENTEAGE*. It overrides *DATASETS_AVAILABLE*. In this case the script will process new images for the training set till the new test set size and skipping the already processed ones. It will also update data in the *TEMP_DATASET_DIR* and *MODEL_DIR*.


### Input_data.py

This file provides functions and classes used in the learner script. It mainly defines the data strutures, the preprocessing routines and the save/load funtcions. There are only two parameters to set and it is adviced not to do so.

* **IMG_SIZE** - images are squared, centered, made grayscale and resized. This parameter defines the edge of the square.
* **ALLOWED_ORIENTATIONS** - list of labels to be accepted. It is possible the datasets provides more but for many reasons it is better to affirmatively declare the ones we are interested in. At the moment the following orientations are taken into account:


> ALLOWED_ORIENTATIONS = ['0_0_270',
   '0_0_0',
    '0_0_90',
    '0_0_45',
    '0_0_180']


   [Numpy]: <http://www.numpy.org/r>
   [Requests]: <http://docs.python-requests.org/en/latest/>
   [OpenCV]: <http://opencv.org/>
   [Pymongo]: <https://api.mongodb.org/python/current/>
   [Image_processing]: <https://gitlab.com/jemboo/image_processing>
   [here]: <https://gitlab.com/jemboo/tensor/tree/master>