import os.path
from os import listdir
from os.path import isfile, join
from pathlib import Path

import cv2
import numpy as np
from osgeo import gdal


def get_list_file(img, mask):
    imgs = [os.path.join(img, f) for f in listdir(img) if isfile(join(img, f))]
    masks = [os.path.join(mask, f) for f in listdir(img) if isfile(join(img, f))]

    return imgs, masks


def warp_raster(path, pix_size):
    ds_time_1 = gdal.Open(path, gdal.GA_ReadOnly)
    name_raster = '/'.join(Path(path).parts[-2:])
    path_temp_1 = "/vsimem/1_" + name_raster
    gdal.Warp(path_temp_1, ds_time_1, xRes=pix_size, yRes=pix_size)
    return path_temp_1


def open_raster(path):
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    r_time_1 = ds.GetRasterBand(1).ReadAsArray()
    g_time_1 = ds.GetRasterBand(2).ReadAsArray()
    b_time_1 = ds.GetRasterBand(3).ReadAsArray()
    image = np.dstack((r_time_1, g_time_1, b_time_1))
    return image


def open_mask(path):
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    r_time_1 = ds.GetRasterBand(1).ReadAsArray()

    return r_time_1


def save_raster(array_image, path_save):
    path_save = path_save.replace(".tif", ".png")
    cv2.imwrite(path_save, array_image)


if __name__ == '__main__':
    pix_size = 1
    path_raster = r'C:\Users\art\Desktop\dataset\train_inria\images'
    path_mask = r'C:\Users\art\Desktop\dataset\train_inria\gt'
    base_path_save_img = r'C:\Users\art\Desktop\dataset\train_inria_1m\images'
    base_path_save_mask = r'C:\Users\art\Desktop\dataset\train_inria_1m\gt'
    list_img, list_mask = get_list_file(path_raster, path_mask)

    for (img, mask) in zip(list_img, list_mask):
        path_save_img = os.path.join(base_path_save_img, os.path.basename(img))
        path_save_mask = os.path.join(base_path_save_mask, os.path.basename(mask))

        new_img = warp_raster(img, pix_size)
        new_mask = warp_raster(mask, pix_size)

        img_array = open_raster(new_img)
        img_mask = open_mask(new_mask)

        save_raster(img_array, path_save_img)
        save_raster(img_mask, path_save_mask)

