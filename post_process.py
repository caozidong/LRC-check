import numpy as np
from matplotlib import pyplot as plt
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

def LRC_filling(input, mask):
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
    img_root_1 = ''
    img_root_2 = ''
   
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
        img_filled = LRC_filling(disp_1, mask)
        print('get')
    print((time.time() - start_time) / 20)
    assert False'''
    
    img_filled = LRC_filling(disp_1, mask)
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
