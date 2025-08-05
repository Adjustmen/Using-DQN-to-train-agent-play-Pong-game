import ale_py
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
import collections
from collections import deque
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import torch
from gymnasium.wrappers import AtariPreprocessing
from preprocess import get_frames, stack_frames
import torch.optim as optim

env = gym.make("ALE/Pong-v5", render_mode="human")
#print(observer)
#前期调试相关的动作空间和观察的空间状体是什么
#print(env.action_space)  # 打印出动作空间
#print(observer.shape)  # 打印出状态空间的维度；
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
lr = 1e-3  # 学习率
gamma = 0.99  # 折扣因子
num_episodes = 100  # 训练的轮数
start_epsilon = 1 #探索率
epsilon_decay = 1e-5  # 不断缩减探索率的大小
final_epsilon = 0.02
batch_size = 32
update_frequency = 1000
max_steps = 1000  # 每个轮的最大步数
buffersize = 10000 #表示为能够得到的经验值是
##print(env.observation_space)
state_dim = env.observation_space.shape[0] #表示为状态空间的维度
action_dim = env.action_space.n #表示为动作空间的维度
observer, info= env.reset()
observer = get_frames(observer)
stack_frame = deque(maxlen=4)
stack_state, stack_frame = stack_frames(stack_frame, observer, True)
##为了能够更好的感知所依我们叠加四个帧在其中
##经验回放
class replaybuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen = capacity)  ##设置队列大小为capacity
    def add(self, state, action, reward, next_state, done):  ##将数据放在缓冲池中使得
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batchsize):
        transition = random.sample(self.buffer, batchsize)
        state, action, reward, next_state, done = zip(*transition)
        return np.array(state), action, reward, np.array(next_state), done
    def size(self):  ##得到现在缓冲队列的大小是多少
        return len(self.buffer)
##Q网络
class QNetwork(nn.Module):
    def __init__(self, action_dim, in_channel = 4):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 16, kernel_size=8, stride = 4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride = 2)
        ##全连接层
        self.fc3 = nn.Linear(9 * 9 * 32, 256) 
        self.fc4 = nn.Linear(256, action_dim)
    def forward(self, x):
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
##DQN
class DQN():
    def __init__(self, learning_rate, start_epsilon, espsilon_decay, final_epsilon, gamma):
        self.learning_rate = learning_rate
        self.start_epsilon = start_epsilon
        self.epsilon_decay = espsilon_decay
        self.final_epsilon = final_epsilon
        self.gamma = gamma
        self.q_network = QNetwork(action_dim, 4).to(device)
        self.target_network = QNetwork(action_dim, 4).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict()) # 同步初始权重
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
###经验回放
        self.memory = replaybuffer(buffersize)
        self.total_steps = 0
        self.losses = []
        self.rewards = []
    def get_action(self, state):
        if np.random.rand() < self.start_epsilon:
            return env.action_space.sample()
        else: 
            ##使用Q网络来预测动作
            return np.argmax(self.q_network(state).detach().numpy());
    def computer_loss(self):
        ##计算相应的TD损失
        if self.memory.size() < batch_size:
            return None
        state, action, reward, next_state, done= self.memory.sample(batch_size)
        state_batch = torch.FloatTensor(state).to(device)
        action_batch = torch.FloatTensor(action).to(device)
        reward_bacth = torch.FloatTensor(reward).to(device)
        next_state_batch = torch.FloatTensor(next_state).to(device)
        done_batch = torch.FloatTensor(done).to(device)
        ##获得当前动作采取这个action所能得到的价值
        Q_value = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
        ##求得期望的价值;
        expected_Qvalue = reward_bacth + gamma * next_q_values(1 - done_batch)
        ##使用均方误差;
        Loss = nn.MSELoss(Q_value, expected_Qvalue)
        return Loss
    def update_q_network(self):
         ##定义Q网络相关的逻辑
        loss = self.computer_loss()
        if loss == None:
            return 
        self.optimizer.zero_grad()
        loss.backward()
        ##进行梯度剪枝防止梯度爆炸
        for par in self.q_network.parameters():
            par.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # 定期更新目标网络
        if self.total_steps % update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, 
                       self.start_epsilon - self.epsilon_decay * self.total_steps)
#进行agent的训练;
for i in range(num_episodes):
    observer, info= env.reset()
    observer = get_frames(observer)
    stack_frame = deque(maxlen=4)
    stack_state, stack_frame = stack_frames(stack_frame, observer, True)
    #定义自己的智能体
    agent = DQN(lr, start_epsilon, epsilon_decay, final_epsilon, gamma)
    steps = 0
    total_reward = 0
    while steps< max_steps:
        
        action = agent.get_action(stack_state)
        next_obsever, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_obsever = get_frames(next_obsever)
        next_stack_state, next_stack_frame = stack_frames(stack_frame, next_obsever, False)
        
        ###设计相应的奖励机制;
        clipped_reward = np.clip(reward, -1, 1)
            
            # 存储经验
        agent.memory.add(stack_state, action, clipped_reward, next_stack_state, done)
            
            # 更新网络
        agent.update_q_network()
        agent.total_steps += 1
            
            # 更新状态
        stack_state = next_stack_state
        total_reward += reward
        steps += 1
            
        if done:
            break
        
        # 衰减epsilon
    agent.decay_epsilon()
        
        # 记录奖励
    agent.rewards.append(total_reward)
        
        # 打印进度
    if i % 10 == 0:
        avg_reward = np.mean(agent.rewards[-10:]) if len(agent.rewards) >= 10 else total_reward
        print(f"回合 {i:4d} | 奖励: {total_reward:6.1f} | "
                f"平均奖励: {avg_reward:6.1f} | ε: {agent.epsilon:.3f} | "
                f"步数: {steps:4d} | 缓冲区: {agent.memory.size():5d}")