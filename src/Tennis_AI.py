
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#コメント" data-toc-modified-id="コメント-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>コメント</a></span></li><li><span><a href="#import" data-toc-modified-id="import-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>import</a></span></li><li><span><a href="#ゲームの位置とサイズ調整" data-toc-modified-id="ゲームの位置とサイズ調整-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>ゲームの位置とサイズ調整</a></span></li><li><span><a href="#DQN" data-toc-modified-id="DQN-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>DQN</a></span><ul class="toc-item"><li><span><a href="#import" data-toc-modified-id="import-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>import</a></span></li><li><span><a href="#GPU判定" data-toc-modified-id="GPU判定-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>GPU判定</a></span></li><li><span><a href="#パラメータ" data-toc-modified-id="パラメータ-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>パラメータ</a></span></li><li><span><a href="#関数" data-toc-modified-id="関数-4.4"><span class="toc-item-num">4.4&nbsp;&nbsp;</span>関数</a></span></li></ul></li><li><span><a href="#スレッド" data-toc-modified-id="スレッド-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>スレッド</a></span><ul class="toc-item"><li><span><a href="#画像取得" data-toc-modified-id="画像取得-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>画像取得</a></span></li><li><span><a href="#スコア" data-toc-modified-id="スコア-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>スコア</a></span></li><li><span><a href="#報酬" data-toc-modified-id="報酬-5.3"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>報酬</a></span></li><li><span><a href="#録画" data-toc-modified-id="録画-5.4"><span class="toc-item-num">5.4&nbsp;&nbsp;</span>録画</a></span></li></ul></li><li><span><a href="#学習" data-toc-modified-id="学習-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>学習</a></span></li></ul></div>

# # コメント

# - 報酬を-1, 0, 1に変更せず、やはりスコアで傾斜つけた方がいい。理由はは-1,0,1だと点数取っただけだと報酬に反映されず、学習が遅くなる

# # import

# In[1]:


import numpy as np
import cv2
from Tennis_observation import *
from Tennis_action import *
from window_controlle import *
import re
import sys
import copy
import os
import random
import matplotlib.pyplot as plt


# In[2]:


import threading
import subprocess
import time
from win32 import win32gui


# # ゲームの位置とサイズ調整

# In[3]:


#ゲーム画面の位置調整
#1920×1200の画面において右上四分の一に配置
win_left = 953
win_top = 0
win_width = 974
win_height = 587


# In[4]:


#grab_screen
#1920×1200の画面において右上四分の一に配置した画面のゲーム部分だけを取得
left = 1087
top = 30
width = 705
height = 545

#少し大きめに画像をとってくる
size_delta = 5


# In[5]:


#cnnに渡すときの画像サイズ
width_cnn = 84
height_cnn = 84


# In[6]:


gray_frag = True


# In[7]:


adjust_window_pos_size(win_left, win_top, win_width, win_height)


# In[8]:


frame = grab_screen(left, top, width, height, False)


# imshow(frame)

# # DQN

# ## import 

# In[9]:


import torch
import torch.nn as nn
import torch.optim as optim
import math


# ## GPU判定

# In[10]:


use_cuda = torch.cuda.is_available()

device = torch.device("cuda:0" if use_cuda else "cpu")
Tensor = torch.Tensor
LongTensor = torch.LongTensor

seed_value = 23
torch.manual_seed(seed_value)
random.seed(seed_value)


# ## パラメータ

# In[11]:


#学習率
learning_rate = 0.0001
#ゲーム数
num_episodes = 500
gamma = 0.99

hidden_layer = 512

replay_mem_size = 100000
batch_size = 32

update_target_frequency = 5000

double_dqn = True

egreedy = 0.9
egreedy_final = 0.01
egreedy_decay = 10000

report_interval = 10
score_to_solve = 18

clip_error = True
normalize_image = True

file2save = '../model/tennis_save.pth'
save_model_frequency = 10000
resume_previous_training = False


# In[12]:


position = 0
capacity = 4


# In[13]:


output_list = ['state', 'action', 'new_state', 'reward', 'done', 'info']

number_of_inputs = frame.shape[0]
number_of_outputs = len(keys_to_press)
number_of_skips = 10


# In[14]:


# reward
init_flag = ('0.0','0.0')
score_reward_dict = {'0.0':0, '15.0':1, '30.0':2, '40.0':3, '60.0':4, '99':99, 'nan':np.nan}


# ## 関数

# In[15]:


def calculate_epsilon(steps_done):
    epsilon = egreedy_final + (egreedy - egreedy_final) * math.exp(-1. * steps_done / egreedy_decay)
    
    return epsilon


# In[16]:


def load_model():
    return torch.load(file2save)


# In[17]:


def save_model(model):
    torch.save(model.state_dict(), file2save)


# In[18]:


def preprocess_frame(frame):
    frame = frame.transpose((2,0,1))
    frame = torch.from_numpy(frame)
    frame = frame.to(device, dtype=torch.float32)
    frame = frame.unsqueeze(1)
    
    return frame


# In[19]:


def plot_results():
    plt.figure(figsize=(12,5))
    plt.title('Rewards')
    plt.plot(reards_total, alpha=0.6, color='red')
    plt.savefig('Tennis_result.png')
    plt.close()


# In[20]:


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


# In[21]:


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


# In[22]:


class QNet_Agent(object):
    def __init__(self):
        self.nn = NeuralNetwork().to(device)
        self.target_nn = NeuralNetwork().to(device)

        self.loss_func = nn.MSELoss()
        #self.loss_func = nn.SmoothL1Loss()
        
        self.optimizer = optim.Adam(params=self.nn.parameters(), lr=learning_rate)
        #self.optimizer = optim.RMSprop(params=mynn.parameters(), lr=learning_rate)
        
        self.number_of_frames = 0
        
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
        
        if self.number_of_frames % update_target_frequency == 0:
            self.target_nn.load_state_dict(self.nn.state_dict())
        
        if self.number_of_frames % save_model_frequency == 0:
            save_model(self.nn)
        
        self.number_of_frames += 1


# # スレッド

# ## 画像取得

# In[23]:


frame_current = {'frame':grab_screen(left, top, width, height, False)}
stop_flag = {'flag':False}


# In[24]:


t1 = threading.Thread(target=observation, args=(left, top, width, height, frame_current, stop_flag))
t1.start()


# ## スコア

# In[28]:


def get_score(my_score, enemy_score, stop_flag):
    while True:
        # score取得
        score = score_check(frame_current['frame'])
        my_score['my_score'] = float(score[0])
        enemy_score['enemy_score'] = float(score[1])

        if stop_flag['flag'] == True:
            print('stop getting score')
            break


# In[29]:


my_score = {'my_score':0}
enemy_score = {'enemy_score':0}

t2 = threading.Thread(target=get_score, args=(my_score, enemy_score, stop_flag))
t2.start()


# ## 報酬

# In[30]:


def reward_calculation(my_last_score, enemy_last_score, reward, stop_flag):
    #スコアと報酬初期化
    score = ('99', '99')
    last_score = ('99', '99')
    reward = -3
    count_game = 0
    
    while stop_flag['flag'] != True:
        #スコア取得
        score = (str(my_score['my_score']), str(enemy_score['enemy_score']))
        
        #ポイント終了
        if (score != last_score) & ('nan' not in score):
            point_fin_flag['point_fin_flag'] = True
            #print('score:', str(score))
            
            #(0,0)で報酬清算
            #1ゲーム目の0-0は報酬計算対象外
            if (score == init_flag) & (count_game == 0):
                count_game += 1
                #print('game: ', count_game)
            
            #2ゲーム目以降
            if (score == init_flag) & (count_game > 0):
                reward = score_reward_dict[last_score[0]] - score_reward_dict[last_score[1]]
                #print('reward:', str(reward))
                
                count_game += 1
                #print('game: ', count_game)
            
            #0-0の認識できなかった時の報酬計算処理
            #if
            

            last_score = copy.copy(score)
            my_last_score['my_last_score'] = last_score[0]
            enemy_last_score['enemy_last_score'] = last_score[1]
            
    
    print('fin')


# reward = {'reward':-3}
# my_last_score = {'my_last_score': '99'}
# enemy_last_score = {'enemy_last_score': '99'}
# 
# 
# t3 = threading.Thread(target=reward_calculation, args=(my_last_score, enemy_last_score, reward, stop_flag))
# t3.start()

# print(my_last_score['my_last_score'], enemy_last_score['enemy_last_score'])

# ## 録画

# video_stop_flag = {'flag':False}
# t4 = Process(target=video_record, args=(frame_current['frame'], stop_flag)) # argsで引数二つ以上じゃないとエラー出る
# t4.start()

# imshow(frame_current['frame'])

# # 学習

# In[33]:


# In[35]:


memory = ExperienceReplay(replay_mem_size)
qnet_agent = QNet_Agent()


reward_list = []
score_list = []

frames_total = 0
count_game = 0

active_window()
reward = 0.0

done = False

score = ('99', '99')
last_score = ('99', '99')49
num_episodes = 10000

# 0-0になる数(ゲーム数)
for i_episode in range(num_episodes):
    

    score_list_buf = []
    reward_list_buf = []
    
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
        
        #新しい環境
        state = new_state
        
        new_state = image_gray_resize(frame_current['frame'], width_cnn, height_cnn)
        
        score = (str(my_score['my_score']), str(enemy_score['enemy_score']))
        
        #ゲームが終わったかの確認               
        memory.push(state, action, new_state, reward, done)
        qnet_agent.optimize()
        
        #ポイント終了
        if (score != last_score) & ('nan' not in score):
            print('score:', str(score))
            
            score_list_buf.append(score)
            
            #(0,0)で報酬清算
            #1ゲーム目の0-0は報酬計算対象外
            if (score == init_flag) & (count_game == 0):
                count_game += 1
                print('game: ', count_game)
            
            #2ゲーム目以降
            if (score == init_flag) & (count_game > 0):
                reward = score_reward_dict[last_score[0]] - score_reward_dict[last_score[1]]
                reward_list.append(reward)
                print('reward:', str(reward))
                
                count_game += 1
                print('game: ', count_game)
            
            #0-0の認識できなかった時の報酬計算処理
            #if

            last_score = copy.copy(score)
            break
        
        score_list.append(score_list_buf)
        

