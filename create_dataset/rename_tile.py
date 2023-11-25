import os
import os.path
from os import listdir
from os.path import isfile, join


def get_list_file(img, mask):
    imgs = [os.path.join(img, f) for f in listdir(img) if isfile(join(img, f))]
    masks = [os.path.join(mask, f) for f in listdir(mask) if isfile(join(mask, f))]

    return imgs, masks
def rename_all(list_img, list_mask):
    for i, (img, mask) in enumerate(zip(list_img, list_mask)):
        os.rename(img, os.path.join(path_raster, str(i) + '.png'))
        os.rename(mask, os.path.join(path_mask, str(i) + '.png'))
if __name__ == '__main__':
    path_raster = r'C:\Users\art\Desktop\dataset\SKOL_DATA\SKOL_DATA\valid\images'
    path_mask = r'C:\Users\art\Desktop\dataset\SKOL_DATA\SKOL_DATA\valid\masks'
    list_img, list_mask = get_list_file(path_raster, path_mask)
    rename_all(list_img, list_mask)