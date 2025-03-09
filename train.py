import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
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

    render_callback = RenderCallback(render_freq=100)

    model = SAC("MlpPolicy", env, verbose=2, batch_size=4096, tensorboard_log="./logs/tensorboard/")
    
    model.learn(total_timesteps=2000000, callback=render_callback)
    
    model.save("sac_goalkeeper")
    env.close()

if __name__ == "__main__":
    main()
