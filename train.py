import os
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from src.environment.soccer_env import SoccerEnv
from src.environment.render_callback import RenderCallback

import torch

print("CUDA disponível:", torch.cuda.is_available())
print("Dispositivo atual:", torch.cuda.current_device())
print("Nome do dispositivo:", torch.cuda.get_device_name(0))


def main():
    env = SoccerEnv()
    env = Monitor(env, "./logs/")
    check_env(env)

    render_callback = RenderCallback(render_freq=1)

    checkpoint_dir = "./checkpoints/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=5000,  
        save_path=checkpoint_dir,
        name_prefix="sac_goalkeeper",
        save_replay_buffer=True,  
        save_vecnormalize=True,   
    )

    model = SAC("MlpPolicy", env, verbose=2, batch_size=16384, tensorboard_log="./logs/tensorboard/")

    model.learn(total_timesteps=100000, callback=[render_callback, checkpoint_callback])
    model.save(os.path.join(checkpoint_dir, "sac_goalkeeper_final"))

    env.close()


if __name__ == "__main__":
    main()
