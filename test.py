import cv2
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from src.environment.soccer_env import SoccerEnv  

def evaluate(model, num_episodes=5):
    env = SoccerEnv()
    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(state, deterministic=True)
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            env.render()
            time.sleep(0.05)
        print(f"Episódio {ep+1}: recompensa total = {total_reward}")
        env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model = SAC.load("sac_goalkeeper.zip") 
    evaluate(model, num_episodes=5)
