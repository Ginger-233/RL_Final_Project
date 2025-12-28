python run.py --env_name PongNoFrameskip-v4

tensorboard --logdir runs

python run.py --env_name HalfCheetah-v4

python visualize.py --env_name "HalfCheetah-v4" --model_path "save/sota/mybaby.pth"

python visualize.py --env_name "PongNoFrameskip-v4" --model_path "save/sota/mybaby.pth"
python visualize.py --env_name "HalfCheetah-v4" --model_path "ppo_HalfCheetah-v4_20251228_1655.pth"