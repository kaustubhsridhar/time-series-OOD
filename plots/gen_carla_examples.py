import cv2
from glob import glob
import os
import numpy as np

def generate(type_of_imgs):
    if 'out' in type_of_imgs:
        second_half = type_of_imgs.split('_')[-1]
        location = f'../carla_data/testing/out_{second_half}/out/0'
        stride = 11
        total = 123
        # if 'rainy' in type_of_imgs:
        #     stride = 12
        #     total = 141
        if 'replay' in type_of_imgs:
            stride = 11
            total = 50
    else:
        second_half = 'in'
        location = '../carla_data/testing/in/0'
        stride = 11
        total = 123

    roll_of_images = []
    n_roll = 0
    for i in range(0, total, stride):
        img_loc = f'{location}/{i}.png'
        img = cv2.imread(img_loc)
        roll_of_images.append(img)
        n_roll += 1
    final_idx = i

    final_img = roll_of_images[-1]
    if 'replay' in type_of_imgs and n_roll != 12:
        while n_roll < 12:
            n_roll += 1
            roll_of_images.append(final_img)
    
    H = 600
    W = 800

    print(type_of_imgs)
    print(n_roll)
    print(final_idx)

    concat_roll_top = roll_of_images[0]
    concat_roll_bottom = roll_of_images[int(n_roll/2)]
    for n, img in enumerate(roll_of_images[0+1 : int(n_roll/2)]):
        concat_roll_top = np.concatenate((concat_roll_top, img), axis=1)
    for n, img in enumerate(roll_of_images[int(n_roll/2)+1 :]):
        concat_roll_bottom = np.concatenate((concat_roll_bottom, img), axis=1)

    white_space_height = 30
    white_space_row = np.zeros([white_space_height, W*int(n_roll/2), 3],dtype=np.uint8)
    white_space_row[:] = 255

    concat_roll = np.concatenate((np.concatenate((concat_roll_top, white_space_row), axis=0), concat_roll_bottom), axis=0)

    try:
        os.mkdir('./plots_saved')
    except:
        pass
    cv2.imwrite(f"./plots_saved/{second_half}_example.png", concat_roll)

for t in ['in', 'out_replay', 'out_rainy', 'out_foggy', 'out_snowy', 'out_night']:
    generate(t)

