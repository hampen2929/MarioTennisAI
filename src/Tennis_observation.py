# インポート
import os
import cv2
import numpy as np
import numpy as np
from PIL import ImageGrab
import cv2

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
size_delta = 5

fps    = 20
frame_per_sec = (1 / fps)

## video_record
video_name = 'Mario_tennis_AI'
path2video = '../videos/'+video_name+'.mp4'
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
stop_flag = {'flag':False}


def score_pos(width, height, size_delta):
    # ゲーム画面内のスコアの相対位置
    # lower left
    xll_1 = 0.191
    xll_2 = 0.319
    yll_1 = 0.789
    yll_2 = 0.899

    # lower right
    xlr_1 = 0.816
    xlr_2 = 0.943
    ylr_1 = 0.789
    ylr_2 = 0.899

    # upper left
    xul_1 = 0.191
    xul_2 = 0.319
    yul_1 = 0.073
    yul_2 = 0.183

    # upper right
    xur_1 = 0.816
    xur_2 = 0.943
    yur_1 = 0.073
    yur_2 = 0.183

    pos_list = [xll_1 * width - size_delta,xll_2 * width + size_delta, yll_1 * height - size_delta,yll_2 * height + size_delta,
                xlr_1 * width - size_delta,xlr_2 * width + size_delta, ylr_1 * height - size_delta,ylr_2 * height + size_delta,
                xul_1 * width - size_delta,xul_2 * width + size_delta, yul_1 * height - size_delta,yul_2 * height + size_delta,
                xur_1 * width - size_delta,xur_2 * width + size_delta, yur_1 * height - size_delta,yur_2 * height + size_delta]

    pos_list = list(np.array(pos_list).astype(int))

    return pos_list


pos_list = score_pos(width, height, size_delta)


score_value_list = ['0', '15', '30', '40', '60', '99']
path_to_temp = '../images/score_template/'
name_temp_list = os.listdir(path_to_temp)

threshold = 0.8

img_temp_list = []

for path in name_temp_list:
    img = cv2.imread(path_to_temp + path)
    img_temp_list.append(img)


def observation(left, top, width, height, frame, stop_flag):
    while True:
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
    temp_ll = frame[pos_list[2]:pos_list[3], pos_list[0]:pos_list[1]]
    temp_lr = frame[pos_list[6]:pos_list[7], pos_list[4]:pos_list[5]]
    temp_ul = frame[pos_list[10]:pos_list[11], pos_list[8]:pos_list[9]]
    temp_ur = frame[pos_list[14]:pos_list[15], pos_list[12]:pos_list[13]]

    img_my_score = cv2.hconcat([temp_ll, temp_lr])
    img_enemy_score = cv2.hconcat([temp_ul, temp_ur])

    img_score_list = [img_my_score, img_enemy_score]

    loc_list = []
    for img_score_num in range(len(img_score_list)):  # my and enemy
        loc_list_buf = []
        for img_temp_num in range(len(img_temp_list)):  # 0 ~ adv
            result = cv2.matchTemplate(img_score_list[img_score_num], img_temp_list[img_temp_num], cv2.TM_CCOEFF_NORMED)

            loc = np.where(result >= threshold)
            loc_list_buf.append(loc)

        loc_list.append(loc_list_buf)

    score_extract = []
    for loc_num in range(len(loc_list)):  # my and enemy
        my_score = np.nan

        score_pos_list_buf = []
        for num in range(len(loc_list[loc_num])):  # 0~adv
            if len(loc_list[loc_num][num][0]) > 0:
                my_score = score_value_list[num]
        score_extract.append(my_score)

    score = (score_extract[0], score_extract[1])
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


def score_check(frame):
    temp_ll = frame[pos_list[2]:pos_list[3], pos_list[0]:pos_list[1]]
    temp_lr = frame[pos_list[6]:pos_list[7], pos_list[4]:pos_list[5]]
    temp_ul = frame[pos_list[10]:pos_list[11], pos_list[8]:pos_list[9]]
    temp_ur = frame[pos_list[14]:pos_list[15], pos_list[12]:pos_list[13]]

    img_my_score = cv2.hconcat([temp_ll, temp_lr])
    img_enemy_score = cv2.hconcat([temp_ul, temp_ur])

    img_score_list = [img_my_score, img_enemy_score]

    loc_list = []
    for img_score_num in range(len(img_score_list)):  # my and enemy
        loc_list_buf = []
        for img_temp_num in range(len(img_temp_list)):  # 0 ~ adv
            result = cv2.matchTemplate(img_score_list[img_score_num], img_temp_list[img_temp_num], cv2.TM_CCOEFF_NORMED)

            loc = np.where(result >= threshold)
            loc_list_buf.append(loc)

        loc_list.append(loc_list_buf)

    score_extract = []
    for loc_num in range(len(loc_list)):  # my and enemy
        my_score = np.nan

        score_pos_list_buf = []
        for num in range(len(loc_list[loc_num])):  # 0~adv
            if len(loc_list[loc_num][num][0]) > 0:
                my_score = score_value_list[num]
        score_extract.append(my_score)

    score = (score_extract[0], score_extract[1])
    return score


def get_score(my_score, enemy_score, frame, stop_flag):
    while True:
        # score取得
        score = score_check(frame)
        my_score['my_score'] = float(score[0])
        enemy_score['enemy_score'] = float(score[1])

        if stop_flag['flag'] == True:
            print('stop getting score')
            break

import atexit


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
