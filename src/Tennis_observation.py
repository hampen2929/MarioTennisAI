# インポート
import os
import numpy as np
from PIL import ImageGrab
import cv2
from Tennis_action import *
import time

## スコア取得

left_resize = 1010
top_resize = 32
width_resize = 182
height_resize = 140

#grab_screen
#1920×1200の画面において右上四分の一に配置した画面のゲーム部分だけを取得
left = 1087
top = 30
width = 705
height = 545

#少し大きめに画像をとってくる
size_delta = 0

fps    = 20
frame_per_sec = (1 / fps)


## video_record
video_name = 'Mario_tennis_AI'
path2video = '../videos/'+video_name+'.mp4'
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
stop_flag = {'flag':False}

#score_img
shape = (45, 70)

#lower left -- ok
xll_1 = 146
xll_2 = xll_1 + shape[1]
yll_1 = 435
yll_2 = yll_1 + shape[0]

#lower right --standard
xlr_1 = 585
xlr_2 = xlr_1 + shape[1]
ylr_1 = 435
ylr_2 = ylr_1 + shape[0]

#upper left
xul_1 = 146
xul_2 = xul_1 + shape[1]
yul_1 = 44
yul_2 = yul_1 + shape[0]

#upper right --ok
xur_1 = 585
xur_2 = xur_1 + shape[1]
yur_1 = 44
yur_2 = yur_1 + shape[0]

pos_list = [[yll_1, yll_2, xll_1, xll_2],
            [ylr_1, ylr_2, xlr_1, xlr_2],
            [yul_1, yul_2, xul_1, xul_2],
            [yur_1, yur_2, xur_1, xur_2]]

score_value_list = [0, 15, 30, 40, 60, 99]
path_to_temp = '../images/score_template/lower_right/'
name_temp_list = os.listdir(path_to_temp)

threshold = 0.7

img_temp_list = []

for path in name_temp_list:
    img = cv2.imread(path_to_temp + path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_temp_list.append(img)


def observation(left, top, width, height, frame, stop_flag):
    while True:
        #pillow Original RGB, cv2 originally GBR
        #cv2.COLOR_BGR2RGB?
        frame['frame'] = cv2.cvtColor(np.array(ImageGrab.grab(bbox=(left, top, left + width, top + height))), cv2.COLOR_BGR2RGB)
        if stop_flag['flag'] == True:
            break


def grab_screen(left, top, width, height, gray_flag):
    img = ImageGrab.grab(bbox=(left, top, left + width, top + height))
    img = np.array(img)

    if gray_flag == True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def score_check(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    temp_ll = frame[pos_list[0][0]:pos_list[0][1], pos_list[0][2]:pos_list[0][3]]
    temp_lr = frame[pos_list[1][0]:pos_list[1][1], pos_list[1][2]:pos_list[1][3]]
    temp_ul = frame[pos_list[2][0]:pos_list[2][1], pos_list[2][2]:pos_list[2][3]]
    temp_ur = frame[pos_list[3][0]:pos_list[3][1], pos_list[3][2]:pos_list[3][3]]

    img_score_list = [temp_ll, temp_lr, temp_ul, temp_ur]

    loc_list = []
    for img_score_num in range(len(img_score_list)):  # my and enemy
        loc_list_buf = []
        for img_temp_num in range(len(img_temp_list)):  # 0 ~ adv
            result = cv2.matchTemplate(img_score_list[img_score_num], img_temp_list[img_temp_num], cv2.TM_CCOEFF_NORMED)

            loc = len(np.where(result >= threshold)[0])
            loc_list_buf.append(loc)

        loc_list.append(loc_list_buf)

    my_score_array = np.array([loc_list[0], loc_list[1]]).sum(axis=0)
    enemy_score_array = np.array([loc_list[2], loc_list[3]]).sum(axis=0)

    if my_score_array.sum() > 0:
        idx_my_score = np.argmax(my_score_array)
        my_score = score_value_list[idx_my_score]
    else:
        my_score = np.nan

    if enemy_score_array.sum() > 0:
        idx_enemy_score = np.argmax(enemy_score_array)
        enemy_score = score_value_list[idx_enemy_score]
    else:
        enemy_score = np.nan

    score = (my_score, enemy_score)

    return score


def image_gray_resize(state, width, height):

    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    state = cv2.resize(state, (width, height))
    state = np.reshape(state, (state.shape[0], state.shape[1], 1))
    return state


def image_resize(state, width, height):
    state = cv2.resize(state, (width, height))
    state = np.reshape(state, (state.shape[0], state.shape[1], 1))
    return state


def get_score(my_score, enemy_score, frame, stop_flag):
    while True:
        # score取得
        score = score_check(frame)
        my_score['my_score'] = float(score[0])
        enemy_score['enemy_score'] = float(score[1])

        if stop_flag['flag'] == True:
            print('stop getting score')
            break


def video_record(stop_flag, _):
    video = cv2.VideoWriter(path2video, fourcc, fps, (int(width), int(height)))
    try:
        while True:
            frame = cv2.cvtColor(np.array(ImageGrab.grab(bbox=(left, top, left + width, top + height))),
                                 cv2.COLOR_BGR2RGB)
            video.write(frame)
    except:
        print('key')
    else:
        print('else')
    finally:
        print('q')
        video.release()
    atexit.register(print('exit'))
0