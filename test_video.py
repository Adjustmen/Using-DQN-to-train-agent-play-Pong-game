import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
from gymnasium.wrappers import  RecordVideo
import numpy as np
from collections import deque

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
class Qnetwork(nn.Module):
    def __init__(self, action_dim, in_channel = 4):

model.eval()
env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=True, resize = 84 * 84)
print(env)
env = deque(maxlen=4)

#env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda episode_id: True)

state, _ = env.reset()
done = False
""""
while not done:
    # 状态格式处理
    state_tensor = torch.tensor(np.array(state), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = model(state_tensor)
        action = torch.argmax(q_values).item()

    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    state = next_state

env.close()
"""""