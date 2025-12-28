import argparse
import gymnasium as gym
import numpy as np
import torch
import os
import time
import matplotlib.pyplot as plt # 用于生成静态图片 
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import ale_py
from agents import DuelingDDQNAgent, PPOAgent
from gymnasium.wrappers import AtariPreprocessing

try:
    from gymnasium.wrappers import FrameStack
except ImportError:
    from gymnasium.wrappers import FrameStackObservation as FrameStack

gym.register_envs(ale_py)

# --- 修改功能：保存曲线图片到指定文件夹 ---
def save_plot(data, label, env_name, save_dir, filename_prefix):
    plt.figure(figsize=(10, 5))
    plt.plot(data)
    plt.title(f"{label} over Time - {env_name}")
    plt.xlabel("Episodes/Updates")
    plt.ylabel(label)
    plt.grid(True)
    # 图片保存路径：save_dir/prefix_envname.png
    save_path = os.path.join(save_dir, f"{filename_prefix}_{env_name}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Graph saved as: {save_path}")

def make_atari_env(env_name):
    # 1. 确保 env_name 包含 NoFrameskip。
    if "NoFrameskip" not in env_name:
        env_id = env_name.replace("-v5", "NoFrameskip-v4")
    else:
        env_id = env_name

    # 2. 创建环境
    env = gym.make(env_id, render_mode="rgb_array")
    
    # 3. AtariPreprocessing 处理
    env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=False)
    env = FrameStack(env, 4)
    return env

def train_dqn(env_name, writer):
    print(f"Starting Dueling Double DQN on {env_name}")
    env = make_atari_env(env_name)
    agent = DuelingDDQNAgent(env.observation_space.shape, env.action_space.n)
    
    total_steps = 1000000
    target_update_freq = 1000
    
    rewards_history = []
    loss_history = []
    
    state, _ = env.reset()
    episode_reward = 0
    
    pbar = tqdm(range(total_steps), desc="DQN Training")
    for global_step in pbar:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        agent.store(state, action, reward, next_state, done)
        loss = agent.update()
        
        if loss is not None:
            loss_history.append(loss)
            if global_step % 100 == 0:
                writer.add_scalar("losses/td_loss", loss, global_step)
        
        state = next_state
        episode_reward += reward
        
        if global_step % target_update_freq == 0:
            agent.update_target_network()
            
        if done:
            rewards_history.append(episode_reward)
            writer.add_scalar("charts/episodic_return", episode_reward, global_step)
            pbar.set_postfix({"Rew": episode_reward, "Eps": f"{agent.epsilon:.2f}"})
            state, _ = env.reset()
            episode_reward = 0

    # --- 修改部分：保存模型与图片到 save/timestamp_envname 文件夹 ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = os.path.join("save", f"{timestamp}_{env_name}")
    os.makedirs(save_dir, exist_ok=True) # 创建文件夹

    model_path = os.path.join(save_dir, f"dqn_{env_name}.pth")
    torch.save(agent.online_net.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    save_plot(rewards_history, "Reward", env_name, save_dir, "reward_curve")
    save_plot(loss_history, "Loss", env_name, save_dir, "loss_curve")

def train_ppo(env_name, writer):
    print(f"Starting PPO on {env_name}")
    env = gym.make(env_name, render_mode="rgb_array")
    agent = PPOAgent(env.observation_space.shape[0], env.action_space.shape[0])
    
    num_steps = 2048 
    total_timesteps = 300000
    num_updates = total_timesteps // num_steps
    
    rewards_history = []
    loss_history = []
    
    global_step = 0
    pbar = tqdm(range(num_updates), desc="PPO Updating")
    
    for update in pbar:
        rollouts = {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'dones': [], 'values': []}
        state, _ = env.reset()
        curr_reward = 0
        
        for step in range(num_steps):
            global_step += 1
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            rollouts['states'].append(state)
            rollouts['actions'].append(action)
            rollouts['log_probs'].append(log_prob)
            rollouts['rewards'].append(reward)
            rollouts['dones'].append(done)
            rollouts['values'].append(value)
            
            state = next_state
            curr_reward += reward
            if done:
                rewards_history.append(curr_reward)
                writer.add_scalar("charts/episodic_return", curr_reward, global_step)
                curr_reward = 0
                state, _ = env.reset()
        
        loss = agent.update(rollouts)
        loss_history.append(loss)
        writer.add_scalar("losses/policy_loss", loss, global_step)
        pbar.set_postfix({"AvgRew": f"{np.mean(rewards_history[-5:]):.2f}" if rewards_history else 0})

    # --- 修改部分：保存模型与图片到 save/timestamp_envname 文件夹 ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = os.path.join("save", f"{timestamp}_{env_name}")
    os.makedirs(save_dir, exist_ok=True) # 创建文件夹

    model_path = os.path.join(save_dir, f"ppo_{env_name}.pth")
    torch.save(agent.network.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    save_plot(rewards_history, "Reward", env_name, save_dir, "reward_curve")
    save_plot(loss_history, "Loss", env_name, save_dir, "loss_curve")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL Final Project")
    parser.add_argument("--env_name", type=str, required=True, help="Environment name")
    args = parser.parse_args()

    exp_name = f"{args.env_name}__{datetime.now().strftime('%m%d_%H%M%S')}"
    writer = SummaryWriter(f"runs/{exp_name}")

    if any(k in args.env_name for k in ["NoFrameskip", "Pong", "Breakout", "Boxing", "VideoPinball"]): 
        train_dqn(args.env_name, writer)
    else:
        train_ppo(args.env_name, writer)
        
    writer.close()
