

#import
from Tennis_observation import *
from Tennis_action import *
from window_controlle import *

import numpy as np
import pandas as pd
import copy
import os
import random
import matplotlib.pyplot as plt
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

# DQN
## import
import torch
import torch.nn as nn
import torch.optim as optim
import math


## GPU判定

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
Tensor = torch.Tensor
LongTensor = torch.LongTensor

seed_value = 0
torch.manual_seed(seed_value)
random.seed(seed_value)


## パラメータ

#学習率
learning_rate = 0.001
#ゲーム数
num_episodes = 1000
gamma = 0.99

hidden_layer = 512

replay_mem_size = 100000
batch_size = 32

update_target_frequency = 5


double_dqn = False

egreedy = 0.9
egreedy_final = 0.01
egreedy_decay = 10000

report_interval = 10
score_to_solve = 18

clip_error = True
normalize_image = True
save_model_frequency = 50
resume_previous_training = False

position = 0
capacity = 4

output_list = ['state', 'action', 'new_state', 'reward', 'done', 'info']

number_of_inputs = frame.shape[0]
number_of_outputs = len(keys_to_press)
number_of_skips = 10

# reward
init_flag = (0,0)
score_reward_dict = {0:0, 15:1, 30:2, 40:3, 60:4, 99:99, np.nan : np.nan}


## 関数

def calculate_epsilon(steps_done):
    epsilon = egreedy_final + (egreedy - egreedy_final) * math.exp(-1. * steps_done / egreedy_decay)

    return epsilon


def load_model():
    return torch.load(file2save)


def save_model(model, file2save):
    torch.save(model.state_dict(), file2save)


def preprocess_frame(frame):
    frame = frame.transpose((2,0,1))
    frame = torch.from_numpy(frame)
    frame = frame.to(device, dtype=torch.float32)
    frame = frame.unsqueeze(1)

    return frame


def plot_results():
    plt.figure(figsize=(12,5))
    plt.title('Rewards')
    plt.plot(reards_total, alpha=0.6, color='red')
    plt.savefig('Tennis_result.png')
    plt.close()


class ExperienceReplay(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, new_state, reward, done):
        transition = (state, action, new_state, reward, done)

        if self.position >= len(self.memory):
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition

        self.position = ( self.position + 1 ) % self.capacity


    def sample(self, batch_size):
        return zip(*random.sample(self.memory, batch_size))


    def __len__(self):
        return len(self.memory)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.advantage1 = nn.Linear(7*7*64,hidden_layer)
        self.advantage2 = nn.Linear(hidden_layer, number_of_outputs)

        self.value1 = nn.Linear(7*7*64,hidden_layer)
        self.value2 = nn.Linear(hidden_layer,1)

        #self.activation = nn.Tanh()
        self.activation = nn.ReLU()


    def forward(self, x):

        if normalize_image:
            x = x / 255

        output_conv = self.conv1(x)
        output_conv = self.activation(output_conv)
        output_conv = self.conv2(output_conv)
        output_conv = self.activation(output_conv)
        output_conv = self.conv3(output_conv)
        output_conv = self.activation(output_conv)

        output_conv = output_conv.view(output_conv.size(0), -1) # flatten

        output_advantage = self.advantage1(output_conv)
        output_advantage = self.activation(output_advantage)
        output_advantage = self.advantage2(output_advantage)

        output_value = self.value1(output_conv)
        output_value = self.activation(output_value)
        output_value = self.value2(output_value)

        output_final = output_value + output_advantage - output_advantage.mean()

        return output_final


class QNet_Agent(object):
    def __init__(self):
        self.nn = NeuralNetwork().to(device)
        self.target_nn = NeuralNetwork().to(device)

        self.loss_func = nn.MSELoss()
        #self.loss_func = nn.SmoothL1Loss()

        self.optimizer = optim.Adam(params=self.nn.parameters(), lr=learning_rate)
        #self.optimizer = optim.RMSprop(params=mynn.parameters(), lr=learning_rate)

        self.number_of_games = 0

        if resume_previous_training and os.path.exists(file2save):
            print("Loading previously saved model ... ")
            self.nn.load_state_dict(load_model())

    def select_action(self,state,epsilon):

        random_for_egreedy = torch.rand(1)[0]

        if random_for_egreedy > epsilon:

            with torch.no_grad():

                state = preprocess_frame(state)
                action_from_nn = self.nn(state)

                action = torch.max(action_from_nn,1)[1]
                action = action.item()
        else: # random
            action = action_random()

        return action

    def optimize(self):
        print('optimize')

        if (len(memory) < batch_size):
            return

        state, action, new_state, reward, done = memory.sample(batch_size)

        state = [ preprocess_frame(frame) for frame in state ]
        state = torch.cat(state)

        new_state = [ preprocess_frame(frame) for frame in new_state ]
        new_state = torch.cat(new_state)

        reward = Tensor(reward).to(device)
        action = LongTensor(action).to(device)
        done = Tensor(done).to(device)


        if double_dqn:
            new_state_indexes = self.nn(new_state).detach()
            max_new_state_indexes = torch.max(new_state_indexes, 1)[1]

            new_state_values = self.target_nn(new_state).detach()
            max_new_state_values = new_state_values.gather(1, max_new_state_indexes.unsqueeze(1)).squeeze(1)
        else:
            new_state_values = self.target_nn(new_state).detach()
            max_new_state_values = torch.max(new_state_values, 1)[0]


        target_value = reward + ( 1 - done ) * gamma * max_new_state_values

        predicted_value = self.nn(state).gather(1, action.unsqueeze(1)).squeeze(1)

        loss = self.loss_func(predicted_value, target_value)

        self.optimizer.zero_grad()
        loss.backward()

        if clip_error:
            for param in self.nn.parameters():
                param.grad.data.clamp_(-1,1)

        self.optimizer.step()

        if self.number_of_games % update_target_frequency == 0:
            self.target_nn.load_state_dict(self.nn.state_dict())
            print('model updated')

        if self.number_of_games % save_model_frequency == 0:
            file2save = '../model/tennis_save_{0:04d}.pth'.format(self.number_of_games)
            save_model(self.nn, file2save)

        self.number_of_games += 1


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
            #print('server', server_flag['flag'])

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

                    info_df = pd.DataFrame(info_list, columns=column)
                    info_df.to_csv('../data/info_df.csv', index=False)

                    count_game += 1

                    my_score['score'] = score[0]
                    enemy_score['score'] = score[1]

                    game_end_flag['flag'] = True

            info = [count_match['score'], server_flag['flag'], count_game, score, reward['reward']]
            info_list.append(info)

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

t2 = Thread(target=get_score, args=(my_score, enemy_score, reward, game_end_flag))
t2.start()

#end_flag
end_flag = {'flag':False}
t3 = Thread(target=end_judge, args=(end_flag, True))
t3.start()

#server_flag
server_flag = {'flag':1}
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
column = ['match', 'server', 'game', 'score', 'reward']

count_game = 0

new_state = image_gray_resize(frame_current['frame'], width_cnn, height_cnn)
state = new_state

# 1ゲーム終わるまで継続
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
        qnet_agent.optimize()
        game_end_flag['flag'] = False

    #試合終了
    if end_flag['flag'] == True:

        end_action(end_flag, count_match)
        count_match['score'] = count_match['score'] + 1

        end_flag['flag'] == False




