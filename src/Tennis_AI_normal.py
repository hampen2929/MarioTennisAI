

#import
from Tennis_observation import *
from Tennis_action import *
from window_controlle import *
from Tennis_RL import *


import numpy as np
import pandas as pd
import copy
from threading import Thread
import cv2


# ゲームの位置とサイズ調整

#ゲーム画面の位置調整
#1920×1200の画面において右上四分の一に配置
win_left = 953
win_top = 0
win_width = 974
win_height = 587

#grab_screen
#1920×1200の画面において右上四分の一に配置した画面のゲーム部分だけを取得
left = 1087
top = 30
width = 705
height = 545

#少し大きめに画像をとってくる
size_delta = 5

#cnnに渡すときの画像サイズ
width_cnn = 84
height_cnn = 84

#試合終了のフラグ(PEACH DOME)
end_left = 240
end_top = 10
end_right = 470
end_bottom = 70

img_end = cv2.imread('../images/end_flag/end_flag.png')
img_end = cv2.cvtColor(img_end, cv2.COLOR_BGR2GRAY)
threshold = 0.8

#serverのフラグ
server_left = 85
server_top = 240
server_right = 155
server_bottom = 310

img_server_mario = cv2.imread('../images/server/server_mario.png')
img_server_mario = cv2.cvtColor(img_server_mario, cv2.COLOR_BGR2GRAY)

img_server_luigi = cv2.imread('../images/server/server_luigi.png')
img_server_luigi = cv2.cvtColor(img_server_luigi, cv2.COLOR_BGR2GRAY)


threshold_server = 0.9

adjust_window_pos_size(win_left, win_top, win_width, win_height)

frame = grab_screen(left, top, width, height, False)







def get_score(my_score, enemy_score, reward, game_end_flag):
    count_game = 0
    last_score = (99, 99)
    info_list = []

    #ゲーム開始
    while True:
        score = score_check(frame_current['frame'])
        
        #ポイント取得
        if (score != last_score) & (np.nan not in score):

            print('score:', str(score))
            print('server', server_flag['flag'])

            #reward['reward'] = score_reward_dict[score[0]] - score_reward_dict[score[1]]
            #print('reward: ', reward['reward'])

            my_init_judge = (score_reward_dict[score[0]] - score_reward_dict[last_score[0]])
            enemy_init_judge = (score_reward_dict[score[1]] - score_reward_dict[last_score[1]])

            # (0,0)で報酬清算
            # (0,0)読み取り損ねたと時の対応も込み
            if (score == init_flag) | (((my_init_judge < 0) | (enemy_init_judge < 0)) & (15 in score)):
                # 1ゲーム目の0-0は報酬計算対象外
                if count_game == 0:
                    reward['reward'] = -3
                    count_game += 10

                # 2ゲーム目以降
                else:
                    reward['reward'] = score_reward_dict[last_score[0]] - score_reward_dict[last_score[1]]
                    print('reward: ', reward['reward'])
                    count_game += 1

                    my_score['score'] = score[0]
                    enemy_score['score'] = score[1]

                    game_end_flag['flag'] = True

            info = [count_match['score'], server_flag['flag'], count_game, score, reward['reward']]
            info_list.append(info)

            info_df = pd.DataFrame(info_list, columns=column)
            info_df.to_csv('../data/CSV/info_df.csv', index=False)

            last_score = copy.copy(score)


def end_judge(end_flag, _):
    while True:
        frame = frame_current['frame'][end_top:end_bottom, end_left:end_right]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        result = cv2.matchTemplate(frame, img_end, cv2.TM_CCOEFF_NORMED)
        loc = np.where(result > threshold)

        if len(loc[0]) > 0:
            end_flag['flag'] = True

def end_action(end_flag, count_match):
    print('end game and next!')
    time.sleep(10)

    cv2.imwrite('../score/game_score_{0:04d}.png'.format(count_match['score']), frame_current['frame'])

    #stop record
    record()

    PressKey(A)
    time.sleep(0.2)
    ReleaseKey(A)

    time.sleep(1)

    PressKey(rightarrow)
    time.sleep(0.2)
    ReleaseKey(rightarrow)

    time.sleep(1)

    PressKey(A)
    time.sleep(0.2)
    ReleaseKey(A)

    end_flag['flag'] = False

    # restart record
    record()

def server_judge(server_flag, img_server_mario, img_server_luigi):
    while True:
        frame = frame_current['frame'][server_top:server_bottom, server_left:server_right]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        result_mario = cv2.matchTemplate(frame, img_server_mario, cv2.TM_CCOEFF_NORMED)
        result_luigi = cv2.matchTemplate(frame, img_server_luigi, cv2.TM_CCOEFF_NORMED)

        server_flag_mario = len(np.where(result_mario > threshold_server)[0])
        server_flag_luigi = len(np.where(result_luigi > threshold_server)[0])

        if server_flag_mario == 1:
            server_flag['flag'] = 1
        if server_flag_luigi == 1:
            server_flag['flag'] = 0

# スレッド

## 画像取得

frame_current = {'frame':grab_screen(left, top, width, height, False)}
stop_flag = {'flag':False}

frame_current = {'frame':grab_screen(left, top, width, height, False)}
t1 = Thread(target=observation, args=(left, top, width, height, frame_current, stop_flag))
t1.start()

## スコア
my_score = {'score':0}
enemy_score = {'score':0}
game_end_flag = {'flag':False}
count_match = {'score':1}
reward = {'reward':0}
column = ['match', 'server', 'game', 'score', 'reward']
server_flag = {'flag':1}

t2 = Thread(target=get_score, args=(my_score, enemy_score, reward, game_end_flag))
t2.start()

#end_flag
end_flag = {'flag':False}
t3 = Thread(target=end_judge, args=(end_flag, True))
t3.start()

#server_flag
t4 = Thread(target=server_judge, args=(server_flag, img_server_mario, img_server_luigi))
t4.start()


## 録画
# video_stop_flag = {'flag': False}
# t4 = Process(target=video_record, args=(stop_flag, True))  # argsで引数二つ以上じゃないとエラー出る
# t4.start()

# 学習

memory = ExperienceReplay(replay_mem_size)
qnet_agent = QNet_Agent()

reward_list = []
score_list = []

frames_total = 0
count_game = 0

active_window()

done = False

score_read_flag = False

num_match = 100
num_game = 1000

info_list = []


count_game = 0

new_state = image_gray_resize(frame_current['frame'], width_cnn, height_cnn)
state = new_state

# 1ゲーム終わるまで継続
print('match: ', count_match['score'])
record()
while True:
    frames_total += 1
    epsilon = calculate_epsilon(frames_total)

    #行動決定
    action = qnet_agent.select_action(state, epsilon)
    #行動
    take_action(action)

    #新しい環境の観測
    state = copy.copy(new_state)
    new_state = image_gray_resize(frame_current['frame'], width_cnn, height_cnn)

    #スコア取得
    score = (str(my_score['score']), str(enemy_score['score']))

    #ExperienceMemory
    memory.push(state, action, new_state, reward['reward'], server_flag['flag'])

    #ゲーム終了
    if game_end_flag['flag'] == True:
        # 最適化
        qnet_agent.optimize(memory)
        game_end_flag['flag'] = False

    #試合終了
    if end_flag['flag'] == True:

        end_action(end_flag, count_match)
        count_match['score'] = count_match['score'] + 1

        end_flag['flag'] == False

        print('match: ', count_match['score'])




