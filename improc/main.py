# from scipy.ndimage import imread
import itertools as it
import numpy as np




import improc.features.preprocess as preprocess
from improc.features.descriptor import ZernikeDescriptor


from scipy.misc import imshow
from cv2 import imread

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

from os import listdir
from os.path import isfile, join
shoes_path = 'shoes'
image_files = [join(shoes_path, f) for f in listdir(shoes_path) if isfile(join(shoes_path, f))]


def preprocess_plain(_image_files):
    processed_images = []

    for image_file in _image_files:
        img = imread(image_file)
        img = preprocess.autocrop(img)
        img = preprocess.scale_max(img)

        processed_images.append(img)

    mosaic_image = mosaic(len(processed_images), processed_images)
    # TODO cant remember the syntax
    imwrite(mosaic_image, "out/super_simple.jpg")


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
    imwrite(mosaic_image, "out/super_simple.jpg")


preprocess_plain(image_files)
preprocess_super_simple(image_files)



img = imread('/home/alessio/Desktop/shoe.jpg')
# imshow(img)
# img = img.convertTo(img, CV_32SC1)
# img = preprocess.autocrop(img)
# img = preprocess.blur(
#     img, gaussian_blur={"enabled": True, "ksize_width": 5, "ksize_height": 5,
#     "sigmaX": 0}
# )

img = preprocess.autocrop(img)
img = preprocess.blur(
    img, gaussian_blur={"enabled": True, "ksize_width": 5, "ksize_height": 5,
    "sigmaX": 0}
)
imshow(img)
img = preprocess.grey(img)
img = preprocess.bitwise(img)
img = preprocess.canny(img)
imshow(img)

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
