import math
from copy import deepcopy

import cv2
import numpy as np
import os.path
from os import listdir
from os.path import isfile, join
from pathlib import Path


def get_list_file(img, mask):
    imgs = [os.path.join(img, f) for f in listdir(img) if isfile(join(img, f))]
    masks = [os.path.join(mask, f) for f in listdir(mask) if isfile(join(mask, f))]

    return imgs, masks


def get_pad_img(image, shape_tile, step):
    shape = image.shape
    last_px_x = ((shape[0] - shape_tile) // step + 1 + step // shape_tile) * shape_tile
    last_px_y = ((shape[1] - shape_tile) // step + 1 + step // shape_tile) * shape_tile
    step_h = range(0, last_px_x, step)
    step_w = range(0, last_px_y, step)
    pad_shape = (step_h[-1] + shape_tile, step_w[-1] + shape_tile)
    if len(image.shape) == 2:
        image_pad = np.pad(image, ((0, pad_shape[0] - shape[0]), (0, pad_shape[1] - shape[1])))
    else:
        image_pad = np.pad(image, ((0, pad_shape[0] - shape[0]), (0, pad_shape[1] - shape[1]), (0, 0)))
    return image_pad, step_h, step_w


def get_tiles(image, step_h, step_w, dim=3, shape=512):
    result_im = []
    for i in step_h:
        for j in step_w:
            if dim == 3:
                sub_im = image[i: shape + i, j: j + shape, :]
            else:
                sub_im = image[i: shape + i, j: j + shape]
            if sub_im.shape[:2] == (shape, shape):
                result_im.append(sub_im)

    return result_im


def open_image(path):
    img = cv2.imread(path)
    return img


def open_mask(path):
    mask_rgb = cv2.imread(path)
    # mask_gray = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(mask_gray, 10, 255, cv2.THRESH_BINARY)
    mask = mask_rgb == 1
    return np.uint8(mask *255)


def save_img(array_image, path_save):
    cv2.imwrite(path_save, array_image)


if __name__ == '__main__':
    shape_tile = 256
    step = 256
    path_img = r'C:\Users\art\Desktop\dataset\SKOL_DATA\images'
    path_mask = r'C:\Users\art\Desktop\dataset\SKOL_DATA\masks'

    base_path_save_img = r'C:\Users\art\Desktop\dataset\SKOL_DATA\SKOL_DATA\valid\images'
    base_path_save_mask = r'C:\Users\art\Desktop\dataset\SKOL_DATA\SKOL_DATA\valid\masks'

    list_img, list_mask = get_list_file(path_img, path_mask)
    for (img, mask) in zip(list_img, list_mask):

        array_image = open_image(img)
        array_mask = open_mask(mask)
        image_pad, step_h, step_w = get_pad_img(array_image, shape_tile, step)
        mask_pad, _, _ = get_pad_img(array_mask, shape_tile, step)

        tiles_images = get_tiles(image_pad, step_h, step_w, dim=3, shape=shape_tile)
        tiles_mask = get_tiles(mask_pad, step_h, step_w, dim=1, shape=shape_tile)
        for i, (t_img, t_mask) in enumerate(zip(tiles_images, tiles_mask)):
            save_img(t_img, os.path.join(base_path_save_img, str(i) + "_" + os.path.basename(img)))
            save_img(t_mask, os.path.join(base_path_save_mask, str(i) + "_" + os.path.basename(mask)))
