import os.path
import time
from os import listdir
from os.path import isfile, join

from tqdm import tqdm

from infer import run


def get_list_file(img):
    imgs = [os.path.join(img, f) for f in listdir(img) if isfile(join(img, f))]
    return imgs


if __name__ == '__main__':
    path_image = r'C:\Users\DungeonMaster3000\Desktop\temp_folder\satellite_hack\test_data\images'
    path_save_result = r'C:\Users\DungeonMaster3000\Desktop\temp_folder\satellite_hack\test_data\masks'
    device = 'cuda:0'
    model_path = r'C:\Users\DungeonMaster3000\Desktop\temp_folder\satellite_hack\models'
    list_img_path = get_list_file(path_image)
    start_time = time.time()

    for img in list_img_path:
        path_save = os.path.join(path_save_result, os.path.basename(img))
        ret = run(model_path, img, path_save, device)
    print('work is completed with the code', ret)
    print("--- %s seconds ---" % (time.time() - start_time))
