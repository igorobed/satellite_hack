import math
import os.path
from copy import deepcopy

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from albumentations.pytorch.transforms import ToTensorV2
import time

from tqdm import tqdm


def get_model(path, devices, type="unet"):
    try:
        if type == "unet":
            model = smp.UnetPlusPlus()
        elif type == "fpn":
            model = smp.FPN()
        model.load_state_dict(
            torch.load(path, map_location=torch.device('cpu')))
        model.eval()
        dev = torch.device(devices if torch.cuda.is_available() else "cpu")
        model.to(dev)
        return model
    except Exception as error:
        print(error)
        return 1


def get_image(image_path):
    try:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        (h, w) = (img.shape[0], img.shape[1])
        return img, h, w
    except Exception as error:
        print(error)
        return 2

# получаем количество тайлов
def getDelta(dim, slidwinDim):
    count = math.ceil(dim / slidwinDim)
    return count

# тайлинг с перекрытием
def get_tiles_overlap(inputImage, input_shape, overlap):
    splitted = []

    Nx = getDelta(inputImage.shape[1], overlap)
    Ny = getDelta(inputImage.shape[0], overlap)

    for i in range(Ny):
        for j in range(Nx):
            x_start = overlap * j
            y_start = overlap * i

            x_end = x_start + input_shape
            y_end = y_start + input_shape

            sub_img = inputImage[max(0, y_start): min(y_end, inputImage.shape[0]),
                      max(0, x_start):min(x_end, inputImage.shape[1]), :]

            if x_start < 0:
                x_crop = np.hstack((np.zeros((sub_img.shape[0], abs(x_start), 3), dtype=np.uint8), sub_img))
            else:
                x_crop = deepcopy(sub_img)
            if x_end > inputImage.shape[1]:
                x_crop = np.hstack(
                    (sub_img, np.zeros((sub_img.shape[0], abs(input_shape - sub_img.shape[1]), 3),
                                       dtype=np.uint8)))
            if y_start < 0:
                y_crop = np.vstack((np.zeros((abs(y_start), x_crop.shape[1], 3), dtype=np.uint8), x_crop))
            else:
                y_crop = deepcopy(x_crop)
            if y_end > inputImage.shape[0]:
                y_crop = np.vstack((x_crop, np.zeros((input_shape - sub_img.shape[0], x_crop.shape[1], 3),
                                                     dtype=np.uint8)))
            splitted.append(y_crop)

    return np.array(splitted, dtype=np.uint8), Nx, Ny


def get_transforms(input_shape):
    transforms = A.Compose([
        A.Normalize(),
        A.Resize(input_shape, input_shape),
        ToTensorV2()
    ])
    return transforms


def predict(model_lst, curr_img, transforms, threshold=0.5, device="cpu"):
    # aug = transforms(image=curr_img)
    # proc_img = aug["image"]
    # proc_img = proc_img[np.newaxis, :, :, :].float()
    # forward_res = model(proc_img)
    # res_mask = (F.sigmoid(forward_res) > threshold).int().detach().cpu().numpy()[0, 0]
    aug = transforms(image=curr_img)
    proc_img = aug["image"]
    proc_img = proc_img[np.newaxis, :, :, :].float()
    forward_res_lst = []
    for model in model_lst:
        prob = F.sigmoid(model(proc_img.to(device))).detach().cpu().numpy()[0, 0]
        forward_res_lst.append(prob)
    res_mask = np.zeros((forward_res_lst[0].shape))
    for prob_res in forward_res_lst:
        res_mask += prob_res
    res_mask /= len(model_lst)
    res_mask = (res_mask > threshold).astype(np.uint8)
    return res_mask

# создаем ядро Гауса для сглаживания
def create_kernel(M, std):
    n = torch.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-n ** 2 / sig2)
    gkern2d = torch.outer(w, w)
    return gkern2d.numpy()


# объединяем тайлы с перекрытием и сглаживаеним
def merge_tiles_with_smooth(splitted, kernel, h, w, overlap, shape_tile, Nx, Ny):
    rec_1 = np.zeros((h + shape_tile, w + shape_tile), dtype=np.float16)
    rec_2 = np.zeros((h + shape_tile, w + shape_tile), dtype=np.float16)
    rec_3 = np.zeros((h + shape_tile, w + shape_tile), dtype=np.float16)
    rec_4 = np.zeros((h + shape_tile, w + shape_tile), dtype=np.float16)

    w_1 = np.zeros((h + shape_tile, w + shape_tile), dtype=np.float16)
    w_2 = np.zeros((h + shape_tile, w + shape_tile), dtype=np.float16)
    w_3 = np.zeros((h + shape_tile, w + shape_tile), dtype=np.float16)
    w_4 = np.zeros((h + shape_tile, w + shape_tile), dtype=np.float16)

    for i in range(Ny):
        for j in range(Nx):
            n = (i) * Nx + (j)
            x_start = overlap * j
            y_start = overlap * i

            x_end = x_start + shape_tile  # - delta
            y_end = y_start + shape_tile  # - delta

            sub_img = splitted[n]
            if i % 2 == 0 and j % 2 == 0:
                rec_1[y_start:y_end, x_start:x_end] = sub_img * kernel
                w_1[y_start:y_end, x_start:x_end] = kernel

            if i % 2 == 0 and j % 2 != 0:
                rec_2[y_start:y_end, x_start:x_end] = sub_img * kernel
                w_2[y_start:y_end, x_start:x_end] = kernel
            elif i % 2 != 0 and j % 2 == 0:
                rec_3[y_start:y_end, x_start:x_end] = sub_img * kernel
                w_3[y_start:y_end, x_start:x_end] = kernel
            elif i % 2 != 0 and j % 2 != 0:
                rec_4[y_start:y_end, x_start:x_end] = sub_img * kernel
                w_4[y_start:y_end, x_start:x_end] = kernel
    np.seterr(invalid='ignore')
    return np.true_divide((rec_1 + rec_2 + rec_3 + rec_4), (w_1 + w_2 + w_3 + w_4), dtype=np.float16)

# выбрасываем объекты чем заданная площадь
def sort_contours(img, area):
    # thr = img[:, :, 0] > 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours_list = []
    mask = np.zeros_like(img[:, :, 0])

    for contour in contours:
        if cv2.contourArea(contour) > area:
            contours_list.append(contour)

    for cnt in contours_list:
        cv2.drawContours(mask, [cnt], -1, 1, thickness=cv2.FILLED)

    return mask


def save_image(mask, path_save):
    try:
        cv2.imwrite(path_save, mask)
        return 0
    except Exception as error:
        print(error)
        return 3


def run(model_path, image_path, path_save, device, threshold=0.5):
    try:

        input_shape = 256
        overlap = input_shape // 2
        model_1 = get_model(os.path.join(model_path, "fpn_isogd.pt"), device, type="fpn")
        model_2 = get_model(os.path.join(model_path, "fpn_skoltech.pt"), device, type="fpn")
        model_3 = get_model(os.path.join(model_path, "unetpp_resnet_skolkovo.pt"), device,
                            type="unet")
        model_4 = get_model(os.path.join(model_path, "unet_resnet_isogd.pt"), device, type="unet")

        model_lst = [model_1, model_2, model_3, model_4]

        array_image, h, w = get_image(image_path)
        tiles, Nx, Ny = get_tiles_overlap(array_image, input_shape, overlap)
        transforms = get_transforms(input_shape)
        y_pred = []

        for tile in tqdm(tiles):
            pred = predict(model_lst, tile, transforms, device=device, threshold=threshold)
            y_pred.append(pred)

        kernel = create_kernel(input_shape, 48)
        mask_pred = merge_tiles_with_smooth(y_pred, kernel, h, w, overlap, input_shape, Nx, Ny)
        mask_pred_true_shape = mask_pred[:array_image.shape[0], :array_image.shape[1]]
        #sort_mask = sort_contours(mask_pred_true_shape, area=100)
        stat_save = save_image(mask_pred_true_shape, path_save)
        return stat_save
    except Exception as error:
        print(error)
        return 4


if __name__ == '__main__':
    path_image = r'C:\Users\art\Desktop\AIV_group\geo\infer\test_data\input\train_image_001.png'
    path_save_result = r'C:\Users\art\Desktop\AIV_group\geo\infer\test_data\output\train_image_test.png'

    device = 'cpu'
    model_path = r'C:\Users\art\Desktop\AIV_group\geo\infer\models'
    start_time = time.time()
    ret = run(model_path, path_image, path_save_result, device)
    print('work is completed with the code', ret)
    print("--- %s seconds ---" % (time.time() - start_time))
