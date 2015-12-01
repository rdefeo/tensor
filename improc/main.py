from os import listdir
from os.path import isfile, join

import itertools as it
import numpy as np

from scipy.ndimage import imread
from scipy.misc import imshow
# from cv2 import imread
from scipy.misc import imsave

import improc.features.preprocess as preprocess
from improc.features.descriptor import ZernikeDescriptor


def make_square(img):
    '''Make an image square by adding white pixels to the smaller dimension,
    It keeps the original image centered'''
    img_size = max(img.shape[1], img.shape[0])
    whitebar_size = (img_size - min(img.shape[1], img.shape[0]))/2
    new_img = img

    if img.shape[0] > img.shape[1]:
        whitebar = np.ones((img_size, whitebar_size, 3))*255
        new_img = np.concatenate((whitebar, img, whitebar), axis=1)

    if img.shape[0] < img.shape[1]:
        whitebar = np.ones((whitebar_size, img_size, 3))*255
        new_img = np.concatenate((whitebar, img, whitebar), axis=0)

    if new_img.shape[0] - new_img.shape[1] == 1:
        new_img = np.concatenate((new_img, np.ones((img_size, 1, 3))*255), axis=1)

    if new_img.shape[1] - new_img.shape[0] == 1:
        new_img = np.concatenate((new_img, np.ones((1, img_size, 3))*255), axis=0)

    return new_img


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


def preprocess_plain(_image_files):
    processed_images = []

    for image_file in _image_files:
        img = imread(image_file)
        img = preprocess.autocrop(img)
        img = preprocess.scale_max(img)
        img = make_square(img)
        processed_images.append(img)

    mosaic_image = mosaic(len(processed_images), processed_images)
    # TODO cant remember the syntax
    imsave("out/before_processing.jpg", mosaic_image)


def preprocess_super_simple(_image_files):
    processed_images = []
    for image_file in _image_files:
        img = imread(image_file)
        img = preprocess.autocrop(img)
        img = preprocess.blur(
            img, gaussian_blur={"enabled": True, "ksize_width": 5, "ksize_height": 5,
            "sigmaX": 0}
        )
        img = preprocess.grey(img)
        img = preprocess.bitwise(img)
        img = preprocess.canny(img)

        img = preprocess.scale_max(img)

        processed_images.append(img)

    mosaic_image = mosaic(len(processed_images), processed_images)
    # TODO cant remember the syntax
    imsave("out/simple processing.jpg", mosaic_image)


shoes_path = 'shoes'
image_files = [join(shoes_path, f) for f in listdir(shoes_path) if isfile(join(shoes_path, f))]

preprocess_plain(image_files)
# preprocess_super_simple(image_files)
# img = imread('/home/alessio/Desktop/shoe.jpg')
#
# img = preprocess.autocrop(img)
# img = preprocess.blur(
#     img, gaussian_blur={"enabled": True, "ksize_width": 5, "ksize_height": 5,
#     "sigmaX": 0}
# )
# imshow(img)
# img = preprocess.grey(img)
# img = preprocess.bitwise(img)
# img = preprocess.canny(img)
# imshow(img)

# imshow(img)
# img = img.convertTo(img, CV_32SC1)
# img = preprocess.autocrop(img)
# img = preprocess.blur(
#     img, gaussian_blur={"enabled": True, "ksize_width": 5, "ksize_height": 5,
#     "sigmaX": 0}
# )

# descriptor = ZernikeDescriptor(
#     preprocess=True,
#     radius=84,
#     resize={"enabled": False, "width": 250, "height": 250},
#     grey={"enabled": True},
#     autocrop={"enabled": True},
#     outline_contour={"enabled": True},
#     add_border={"enabled": True, "color_value": 0, "border_size": 15,
#                 "fill_dimensions": True},
#     bitwise_info={"enabled": True},
#     thresh={"enabled": False},
#     scale_max={"enabled": True, "width": 250, "height": 250},
#     dilate={"enabled": True, "width": 7, "height": 7, "iterations": 1},
#     closing={"enabled": True, "width": 5, "height": 5},
#     canny={"enabled": True, "threshold1": 100, "threshold2": 200},
#     gaussian_blur={"enabled": True, "ksize_width": 5, "ksize_height": 5,
#                    "sigmaX": 0},
#     laplacian={"enabled": False})
#
# img = descriptor.do_preprocess(img)
# imshow(img)
