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
import torch.optim as optim
import cv2

# 图像预处理函数
def get_frames(frame):
    """预处理帧：转换为灰度图并调整大小"""
    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # 调整大小到84x84
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized

def stack_frames(stack_frame, frame, is_new_episode):
    """堆叠帧以提供时间信息"""
    if is_new_episode:
        # 新回合开始，用同一帧填充整个堆栈
        stack_frame.clear()
        for _ in range(4):
            stack_frame.append(frame)
    else:
        # 添加新帧
        stack_frame.append(frame)
    
    # 返回堆叠的状态
    stacked_state = np.stack(stack_frame, axis=0)
    return stacked_state, stack_frame

# 环境设置
env = gym.make("ALE/Pong-v5", render_mode="human")

# 设备和超参数
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"使用设备: {device}")

lr = 1e-4  # 降低学习率
gamma = 0.99
num_episodes = 400 # 增加训练轮数
start_epsilon = 1.0
epsilon_decay = 0.001  # 修正epsilon衰减
final_epsilon = 0.02
batch_size = 64
update_frequency = 1000
max_steps = 1000  # 增加最大步数
buffer_size = 20000  # 增加缓冲区大小

action_dim = env.action_space.n
print(f"动作空间大小: {action_dim}")

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)
    
    def size(self):
        return len(self.buffer)
# Q网络
class QNetwork(nn.Module):  # 必须继承nn.Module
    def __init__(self, action_dim, in_channel=4):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        
        # 计算卷积输出大小: 9 * 9 * 32
        self.fc1 = nn.Linear(9 * 9 * 32, 256)
        self.fc2 = nn.Linear(256, action_dim)

    def forward(self, x):
        # 确保输入是正确的数据类型和范围
        x = x.float() / 255.0
        
        x = F.relu(self.conv1(x)) 
        x = F.relu(self.conv2(x))  
        
        # 展平
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # 不要在最后一层使用ReLU

# DQN代理
class DQN:
    def __init__(self, learning_rate, start_epsilon, epsilon_decay, final_epsilon, gamma):
        self.learning_rate = learning_rate
        self.start_epsilon = start_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.gamma = gamma
        self.epsilon = start_epsilon
        
        # 创建网络
        self.q_network = QNetwork(action_dim, 4).to(device)
        self.target_network = QNetwork(action_dim, 4).to(device)
        
        # 同步初始权重
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # 经验回放
        self.memory = ReplayBuffer(buffer_size)
        
        # 统计信息
        self.losses = []
        self.rewards = []
        self.total_steps = 0
    
    def get_action(self, state):
        """选择动作"""
        if np.random.rand() < self.epsilon:
            return env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.q_network(state_tensor)
                return q_values.cpu().data.numpy().argmax()
    
    def compute_loss(self):
        """计算TD损失"""
        if self.memory.size() < batch_size:
            return None
        
        # 采样经验
        state, action, reward, next_state, done = self.memory.sample(batch_size)
        
        # 转换为张量
        state_batch = torch.FloatTensor(state).to(device)
        action_batch = torch.LongTensor(action).to(device)  # 使用LongTensor
        reward_batch = torch.FloatTensor(reward).to(device)
        next_state_batch = torch.FloatTensor(next_state).to(device)
        done_batch = torch.BoolTensor(done).to(device)  # 使用BoolTensor
        
        # 当前Q值
        current_q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        
        # 下一状态的Q值（使用目标网络）
        with torch.no_grad():
            next_q_values = self.target_network(next_state_batch).max(1)[0]
            target_q_values = reward_batch + self.gamma * next_q_values * (~done_batch)
        
        # 计算损失
        loss = F.mse_loss(current_q_values, target_q_values)
        return loss
    
    def update_q_network(self):
        """更新Q网络"""
        loss = self.compute_loss()
        if loss is None:
            return
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        
        self.optimizer.step()
        
        # 记录损失
        self.losses.append(loss.item())
        
        # 定期更新目标网络
        if self.total_steps % update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            print(f"目标网络已更新 (步数: {self.total_steps})")
    
    def decay_epsilon(self):
        """衰减epsilon"""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
    
    def save_model(self, filepath):
        """保存模型"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'losses': self.losses,
            'rewards': self.rewards
        }, filepath)

# 训练函数
def train_dqn():
    """训练DQN代理"""
    
    # 创建DQN代理
    agent = DQN(lr, start_epsilon, epsilon_decay, final_epsilon, gamma)
    
    print("开始训练...")
    print(f"总回合数: {num_episodes}")
    print(f"设备: {device}")
    print("=" * 50)
    
    for episode in range(num_episodes):
        # 重置环境
        observer, info = env.reset()
        observer = get_frames(observer)
        
        # 初始化帧堆栈
        stack_frame = deque(maxlen=4)
        stack_state, stack_frame = stack_frames(stack_frame, observer, True)
        
        total_reward = 0
        steps = 0   
        while steps < max_steps:
            # 选择动作
            action = agent.get_action(stack_state)
            
            # 执行动作
            next_observer, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 预处理下一帧
            next_observer = get_frames(next_observer)
            next_stack_state, stack_frame = stack_frames(stack_frame, next_observer, False)
            
            # 奖励裁剪 Atair
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
        if episode % 10 == 0:
            avg_reward = np.mean(agent.rewards[-10:]) if len(agent.rewards) >= 10 else total_reward
            print(f"回合 {episode:4d} | 奖励: {total_reward:6.1f} | "
                  f"平均奖励: {avg_reward:6.1f} | ε: {agent.epsilon:.3f} | "
                  f"步数: {steps:4d} | 缓冲区: {agent.memory.size():5d}")
        
        # 保存模型
        if episode % 100 == 0 and episode > 0:
            agent.save_model(f"pong_dqn_episode_{episode}.pth")
    
    # 保存最终模型
    agent.save_model("pong_dqn_final.pth")
    
    # 绘制训练曲线
    plot_training_results(agent)
    
    return agent

def plot_training_results(agent):
    """绘制训练结果"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 奖励曲线
    if agent.rewards:
        axes[0, 0].plot(agent.rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True)
    
    # 损失曲线
    if agent.losses:
        axes[0, 1].plot(agent.losses)
        axes[0, 1].set_title('Training Loss')
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
    
    # 移动平均奖励
    if len(agent.rewards) > 50:
        window = 50
        moving_avg = []
        for i in range(window, len(agent.rewards)):
            moving_avg.append(np.mean(agent.rewards[i-window:i]))
        axes[1, 0].plot(range(window, len(agent.rewards)), moving_avg)
        axes[1, 0].set_title('Moving Average Reward (50 episodes)')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Average Reward')
        axes[1, 0].grid(True)
    
    # Epsilon衰减
    epsilons = []
    for step in range(agent.total_steps):
        eps = start_epsilon - step * epsilon_decay
        eps = max(final_epsilon, eps)
        epsilons.append(eps)
    
    axes[1, 1].plot(epsilons)
    axes[1, 1].set_title('Epsilon Decay')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Epsilon')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('pong_training_results.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 开始训练
    trained_agent = train_dqn()
