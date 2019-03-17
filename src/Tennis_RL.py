# DQN
## import
import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt
import random
import os
import numpy as np
import Tennis_action
from Tennis_action import action_random

keys_to_press = Tennis_action.keys_to_press

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
save_model_frequency = 100
resume_previous_training = False

position = 0
capacity = 4

output_list = ['state', 'action', 'new_state', 'reward', 'done', 'info']

#number_of_inputs = frame.shape[0]
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

    def optimize(self, memory):
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