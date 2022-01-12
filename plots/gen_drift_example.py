import cv2
from glob import glob
import os
import numpy as np

imgs = sorted(glob('./example_trace_drift/*.jpg'))
H = 360
W = 640

x_start = int(W/4) - 20
x_end = int(W*3/4) + 80
y_start = int(H/3) - 10
y_end = int(H*2/3) + 50
crop_img_W = x_end-x_start

start_frame_idx = 18
last_frame_idx = start_frame_idx
stride = 3

roll_of_images = []
n_roll = 0

for idx, img_loc in enumerate(imgs):
    img_loc_second_part = img_loc.split('/')[-1]
    img = cv2.imread(img_loc)
    crop_img = img[y_start:y_end, x_start:x_end]
    try:
        os.mkdir('./cropped_example_trace_drift')
    except:
        pass
    cv2.imwrite(f"./cropped_example_trace_drift/{img_loc_second_part}", crop_img)

    if idx+1 >= start_frame_idx:
        if idx - last_frame_idx == stride:
            roll_of_images.append(crop_img)
            last_frame_idx = idx
            n_roll += 1

concat_roll_top = roll_of_images[0]
concat_roll_bottom = roll_of_images[int(n_roll/2)]
for n, img in enumerate(roll_of_images[0+1 : int(n_roll/2)]):
    concat_roll_top = np.concatenate((concat_roll_top, img), axis=1)
for n, img in enumerate(roll_of_images[int(n_roll/2)+1 :]):
    concat_roll_bottom = np.concatenate((concat_roll_bottom, img), axis=1)

print(n_roll)

white_space_height = 10
white_space_row = np.zeros([white_space_height, crop_img_W*int(n_roll/2), 3],dtype=np.uint8)
white_space_row[:] = 255

concat_roll = np.concatenate((np.concatenate((concat_roll_top, white_space_row), axis=0), concat_roll_bottom), axis=0)

try:
    os.mkdir('./plots_saved')
except:
    pass
cv2.imwrite(f"./plots_saved/drift_example.png", concat_roll)



