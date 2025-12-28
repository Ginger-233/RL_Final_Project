import os
# --- 核心修复：设置 MuJoCo 使用 EGL 后端进行无头渲染 ---
# 必须在导入 gymnasium/mujoco 之前设置
os.environ["MUJOCO_GL"] = "egl"

import argparse
import gymnasium as gym
import torch
import numpy as np
from gymnasium.wrappers import RecordVideo, AtariPreprocessing
import ale_py

# 处理 FrameStack 的导入兼容性
try:
    from gymnasium.wrappers import FrameStack
except ImportError:
    from gymnasium.wrappers import FrameStackObservation as FrameStack

from networks import DuelingDQN, PPOActorCritic

# 注册环境
gym.register_envs(ale_py)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_atari_env(env_name):
    """
    复用训练时的环境预处理逻辑，确保模型输入一致
    """
    if "NoFrameskip" not in env_name:
        env_id = env_name.replace("-v5", "NoFrameskip-v4")
    else:
        env_id = env_name

    # 关键：render_mode="rgb_array" 允许在后台录制视频而不弹窗
    env = gym.make(env_id, render_mode="rgb_array")
    env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=False)
    env = FrameStack(env, 4)
    return env

def evaluate_dqn(env_name, model_path, video_dir):
    print(f"Loading DQN model from {model_path} for {env_name}...")
    
    # 1. 创建带录制功能的环境
    env = make_atari_env(env_name)
    # episode_trigger=lambda x: True 表示录制每一局
    env = RecordVideo(env, video_folder=video_dir, name_prefix=f"dqn_{env_name}", episode_trigger=lambda x: True)
    
    # 2. 初始化网络并加载权重
    input_shape = env.observation_space.shape
    num_actions = env.action_space.n
    model = DuelingDQN(input_shape, num_actions).to(device)
    
    # 加载模型，添加 weights_only=False 以兼容旧版保存方式并消除警告（视 PyTorch 版本而定）
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    except TypeError:
         # 如果 PyTorch 版本较低不支持 weights_only 参数
        model.load_state_dict(torch.load(model_path, map_location=device))
        
    model.eval() # 切换到评估模式
    
    # 3. 运行游戏
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    print("Start recording...")
    while not done:
        with torch.no_grad():
            # 预处理：numpy -> tensor (uint8) -> unsqueeze batch dim
            state_tensor = torch.tensor(np.array(state), device=device, dtype=torch.uint8).unsqueeze(0)
            q_values = model(state_tensor)
            # 测试时选择 Q 值最大的动作（Greedy）
            action = q_values.argmax().item()
            
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
    print(f"Episode finished. Total Reward: {total_reward}")
    env.close()
    print(f"Video saved to {video_dir}/")

def evaluate_ppo(env_name, model_path, video_dir):
    print(f"Loading PPO model from {model_path} for {env_name}...")
    
    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(env, video_folder=video_dir, name_prefix=f"ppo_{env_name}", episode_trigger=lambda x: True)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    model = PPOActorCritic(state_dim, action_dim).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    except TypeError:
        model.load_state_dict(torch.load(model_path, map_location=device))
        
    model.eval()
    
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    print("Start recording...")
    while not done:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            # PPO 动作选择
            action, _, _, _ = model.get_action_and_value(state_tensor)
            action = action.cpu().numpy()[0]
            
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
    print(f"Episode finished. Total Reward: {total_reward}")
    env.close()
    print(f"Video saved to {video_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize RL Agent")
    parser.add_argument("--env_name", type=str, required=True, help="Environment name (e.g., PongNoFrameskip-v4)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the .pth model file")
    parser.add_argument("--video_dir", type=str, default="videos", help="Directory to save videos")
    
    args = parser.parse_args()
    
    # 简单的环境类型判断逻辑，与 run.py 保持一致
    if any(k in args.env_name for k in ["NoFrameskip", "Pong", "Breakout", "Boxing", "VideoPinball"]):
        evaluate_dqn(args.env_name, args.model_path, args.video_dir)
    else:
        evaluate_ppo(args.env_name, args.model_path, args.video_dir)
