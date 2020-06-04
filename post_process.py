import numpy as np
from matplotlib import pyplot as plt
from SAD_matching import SAD_2
import cv2
from PIL import Image
import time

def LRC(input_l, input_r, thres=20):
    mask = np.zeros_like(input_l)
    # print(mask.shape)
    h, w = mask.shape

    for i in range(h):
        for j in range(w):
            if abs(int(input_l[i, j]) - int(input_r[i, j-input_l[i, j]])) < thres and (j-input_l[i, j]) >= 0:
                mask[i, j] = 1
    return mask

def LRC_filling_1(input, mask):
    assert input.shape == mask.shape
    h, w = input.shape
    output = np.copy(input)
    count = 0
    for i in range(h):
        for j in range(0, w):
            if mask[i, j] == 0 and j > 0 and j < (w - 1):
                temp_l = (mask[i, 0:j][::-1] != 0).argmax()
                temp_r = (mask[i, (j+1):] != 0).argmax()
                if temp_l > 0 and temp_r > 0:
                    count += 1
                    output[i, j] = min(input[i, j + temp_r + 1], input[i, j - temp_l - 1])
                    print('get')
                elif temp_l > 0 and temp_r == 0:
                    output[i, j] = input[i, j - temp_l - 1]
                elif temp_l == 0 and temp_r > 0:
                    output[i, j] = input[i, j + temp_r + 1]
    print(count)
    return output

def LRC_filling_2(input, mask):
    assert input.shape == mask.shape
    h, w = input.shape
    output = np.copy(input)

    for i in range(h):
        for j in range(w):
            if mask[i, j] == 0 and j > 0 and j < (w - 1):
                flag_l = 0
                flag_r = 0
                for k in range(1, w):
                    if mask[i, j - k] == 1 and flag_l == 0 and (j - k) >= 0:
                        temp_l = input[i, j - k]
                        flag_l = 1
                    if flag_r == 0 and (j + k - w) < 0:
                        if mask[i, j + k] == 1:
                            temp_r = input[i, j + k]
                            flag_r = 1
                    if flag_l == 1 and flag_r == 1:
                        output[i, j] = min(temp_l, temp_r)
                        # output[i, j] = 1
                        break
    return output

def LRC_filling_3(input, mask):
    assert input.shape == mask.shape
    h, w = input.shape
    output = np.copy(input)

    for i in range(h):
        for j in range(0, w):
            if mask[i, j] == 0:
                find_1 = np.argwhere(mask[i, 0:j][::-1] != 0)
                find_2 = np.argwhere(mask[i, (j + 1):] != 0)
                size_1 = find_1.size
                size_2 = find_2.size
                if size_1 > 0 and size_2 > 0:
                    output[i, j] = min(input[i, j + find_2[0] + 1], input[i, j - find_1[0] - 1])
                elif find_1.size > 0:
                    output[i, j] = input[i, j - find_1[0] - 1]
                elif find_2.size > 0:
                    output[i, j] = input[i, j + find_2[0] + 1]
    return output

if __name__ == '__main__':
    img_root_1 = '/media/home_bak/swap/Robot_depth_completion/depth_completion-master/workspace/exp_stereo_lidar/val_output_1_epoch_11/0000000020.png'
    img_root_2 = '/media/home_bak/swap/Robot_depth_completion/depth_completion-master/workspace/exp_stereo_lidar/val_output_2_epoch_11/0000000020.png'
    # img_root_3 = '/home/ding/下载/teddy/disp2.png'
    # gt = cv2.imread(img_root_3)
    # gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    # gt = np.array(gt)

    # print(np.mean(gt), gt.shape)
    # disp_1 = SAD_2(img_root_1, img_root_2, 5, 30, K=0)
    # disp_2 = SAD_2(img_root_1, img_root_2, 5, 30, K=1)
    disp_1 = cv2.imread(img_root_1)
    disp_1 = cv2.cvtColor(disp_1, cv2.COLOR_BGR2GRAY)
    disp_1 = np.array(np.array(disp_1) / 4, dtype=int)
    disp_2 = cv2.imread(img_root_2)
    disp_2 = cv2.cvtColor(disp_2, cv2.COLOR_BGR2GRAY)
    disp_2 = np.array(np.array(disp_2) / 4, dtype=int)

    mask = LRC(disp_1, disp_2)

    '''plt.imshow(mask)
    plt.show()
    assert False'''

    '''start_time = time.time()
    for i in range(20):
        img_filled = LRC_filling_3(disp_1, mask)
        print('get')
    print((time.time() - start_time) / 20)
    assert False'''
    FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
    # img_filled = cv2.dilate(np.array(disp_1 * mask, dtype=np.float32), FULL_KERNEL_5)
    img_filled = LRC_filling_3(disp_1, mask)
    img_filled = np.array(img_filled, dtype=np.float32)
    img_filter = cv2.medianBlur(img_filled, 5)

    plt.figure()
    plt.subplot(4, 1, 1)
    plt.imshow(disp_1)
    plt.subplot(4, 1, 2)
    plt.imshow(mask)
    plt.subplot(4, 1, 3)
    plt.imshow(img_filled)
    plt.subplot(4, 1, 4)
    plt.imshow(img_filter)
    plt.show()
