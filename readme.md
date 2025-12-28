# 强化学习项目：Atari 与 MuJoCo 训练框架

本项目实现了一个通用的强化学习框架，支持两种经典算法：**Dueling Double DQN (D3QN)** 用于处理离散动作空间的 Atari 游戏，以及 **PPO (Proximal Policy Optimization)** 用于处理连续控制任务。

## 🛠 环境要求与安装

项目依赖 `Gymnasium` 生态系统及其相关的模拟器。建议使用 Python 3.8+ 环境。

### 1. 安装基础依赖

```bash
pip install torch torchvision torchaudio
pip install gymnasium[atari,accept-rom-license]
pip install gymnasium[mujoco]
pip install numpy matplotlib tqdm tensorboard ale-py

```

### 2. 系统级依赖 (针对 MuJoCo 渲染)

如果你在服务器（无显示器）环境下运行可视化脚本，项目已内置 EGL 后端支持。请确保系统安装了相关的 OpenGL 库（如 `libegl1-mesa-dev`）。

---

## 📂 文件结构说明

- `run.py`: **项目主入口**。负责解析参数、初始化环境、启动 DQN 或 PPO 训练循环，并定期保存模型和性能图表。
- `agents.py`: **智能体逻辑**。
- `ReplayBuffer`: 针对图像数据优化的经验回放池（使用 `uint8` 存储以节省内存）。
- `DuelingDDQNAgent`: 实现了双路网络（状态价值与优势函数）和双 Q 学习逻辑。
- `PPOAgent`: 实现了基于 GAE（广义优势估计）的策略梯度算法。

- `networks.py`: **神经网络架构**。
- `DuelingDQN`: 包含卷积层（CNN）提取特征，适用于图像输入。
- `PPOActorCritic`: 全连接层架构，适用于连续向量输入。

- `visualize.py`: **测试与可视化**。加载训练好的 `.pth` 模型，运行游戏并保存录像至 `videos/` 文件夹。

---

## 🚀 运行指南

### 1. 训练模型

脚本会根据 `env_name` 自动选择算法：

- **Atari 游戏** (如 Pong, Breakout, Boxing): 使用 **DQN**。
- **连续控制** (如 HalfCheetah, Hopper): 使用 **PPO**。

```bash
# 训练 Atari 游戏 (DQN)
python run.py --env_name "PongNoFrameskip-v4"

# 训练 MuJoCo 环境 (PPO)
python run.py --env_name "HalfCheetah-v4"

```

**训练产出：**

- **权重保存**: 自动创建 `save/时间戳_环境名/` 文件夹，保存 `.pth` 模型。
- **训练曲线**: 在上述文件夹中生成 `reward_curve.png` 和 `loss_curve.png`。
- **日志**: 运行 `tensorboard --logdir=runs` 查看实时训练指标。

### 2. 模型评估与视频录制

使用 `visualize.py` 加载保存的模型并查看表现。

```bash
# 评估 DQN 模型
python visualize.py --env_name "PongNoFrameskip-v4" --model_path "save/your_path/dqn_PongNoFrameskip-v4.pth"

# 评估 PPO 模型
python visualize.py --env_name "HalfCheetah-v4" --model_path "save/your_path/ppo_HalfCheetah-v4.pth"

```

**结果**: 录像文件将保存在 `videos/` 目录下。

---

## 💡 技术细节亮点

- **内存优化**: 在 `agents.py` 中，Atari 图像以 `uint8` 格式存储在 Buffer 中，仅在上传 GPU 时转换为 `float32`，这使得在 16GB 内存的机器上运行大规模回放池成为可能。
- **无头渲染**: `visualize.py` 强制开启 `os.environ["MUJOCO_GL"] = "egl"`，支持在没有物理显示器的 Linux 服务器上直接合成视频。
- **双路竞争架构 (Dueling)**: 网络在预测 Q 值时分离了状态价值 和动作优势 ，在动作较多时能更快收敛。
