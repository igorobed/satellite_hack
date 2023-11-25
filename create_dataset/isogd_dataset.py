import os.path
from os import listdir
from os.path import isfile, join
from pathlib import Path

import cv2
import numpy as np
from osgeo import gdal


def get_list_file(img, mask):
    imgs = [os.path.join(img, f) for f in listdir(img) if isfile(join(img, f))]
    masks = [os.path.join(mask, f) for f in listdir(mask) if isfile(join(mask, f))]

    return imgs, masks


def clear_data(list_img, list_mask):
    for img in list_img:
        mask = img.replace("image", 'mask')
        if os.path.isfile(mask):
            pass
        else:
            os.remove(img)


def set_no_data(path_mask):
    nodata = 255
    ds = gdal.Open(path_mask, gdal.GA_Update)
    for i in range(1, ds.RasterCount + 1):
        ds.GetRasterBand(i).SetNoDataValue(nodata)
    return ds


def open_mask(ds):
    r_time_1 = ds.GetRasterBand(1).ReadAsArray()
    g_time_1 = ds.GetRasterBand(2).ReadAsArray()
    b_time_1 = ds.GetRasterBand(3).ReadAsArray()
    image = np.dstack((r_time_1, g_time_1, b_time_1))
    ds = None
    return image


def get_treshold(image):
    mask = image < 255

    return mask


def save_mask(array_image, path_save):
    path_save = path_save.replace(".tiff", ".png")
    print(path_save)
    cv2.imwrite(path_save, np.uint8(array_image * 255))


def open_raster(path_img):
    ds = gdal.Open(path_img, gdal.GA_ReadOnly)
    r_time_1 = ds.GetRasterBand(3).ReadAsArray()
    g_time_1 = ds.GetRasterBand(2).ReadAsArray()
    b_time_1 = ds.GetRasterBand(1).ReadAsArray()
    image = np.dstack((r_time_1, g_time_1, b_time_1))
    ds = None
    return image

def save_image(array_image, path_save):
    path_save = path_save.replace(".tiff", ".png")

    cv2.imwrite(path_save, array_image)

def rename_all(list_img, list_mask):
    for i, (img, mask) in enumerate(zip(list_img, list_mask)):
        os.rename(img, os.path.join(path_raster, str(i) + '.png'))
        os.rename(mask, os.path.join(path_mask, str(i) + '.png'))
if __name__ == '__main__':
    path_raster = r'C:\Users\art\Desktop\dataset\isogd\images'
    path_mask = r'C:\Users\art\Desktop\dataset\isogd\masks'
    list_img, list_mask = get_list_file(path_raster, path_mask)
    rename_all(list_img, list_mask)
    # clear_data(list_img, list_mask)
    #
    #
    # list_img, list_mask = get_list_file(path_raster, path_mask)
    #
    # for mask in list_mask:
    #     ds = set_no_data(mask)
    #     array_mask = open_mask(ds)
    #     treshold = get_treshold(array_mask)
    #     save_mask(treshold, os.path.join(path_mask, os.path.basename(mask)))
    #
    # for img in list_img:
    #     array_img = open_raster(img)
    #     save_image(array_img, os.path.join(path_raster, os.path.basename(img)))


