import os.path
from os import listdir
from os.path import isfile, join

import cv2
import numpy as np


def get_list_file(img, mask):
    imgs = [os.path.join(img, f) for f in listdir(img) if isfile(join(img, f))]
    masks = [os.path.join(mask, f) for f in listdir(img) if isfile(join(img, f))]

    return imgs, masks


def open_img(path):
    img = cv2.imread(path)

    values, counts = np.unique(img, return_counts=True)
    if len(values) == 2 and counts[1] > 1000:
        return True
    else:
        return False


def compare(list_img, list_mask):
    for (img, mask) in zip(list_img, list_mask):
        status = open_img(mask)

        if status:
            continue
        else:
            os.remove(img)
            os.remove(mask)


if __name__ == '__main__':
    path_raster = r'C:\Users\art\Desktop\dataset\SKOL_DATA\SKOL_DATA\valid\images'
    path_mask = r'C:\Users\art\Desktop\dataset\SKOL_DATA\SKOL_DATA\valid\masks'
    list_img, list_mask = get_list_file(path_raster, path_mask)
    # print(len(list_img), len(list_mask))
    compare(list_img, list_mask)
