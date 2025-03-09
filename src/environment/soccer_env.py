import cv2
import numpy as np
import random
import math
import gymnasium as gym
from gymnasium import spaces

class SoccerEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, width=640, height=480):
        super(SoccerEnv, self).__init__()
        self.width = width
        self.height = height


        self.goal_width = 200  
        self.goal_height = 50  
        self.goal_x = width // 2 
        self.goal_y = height     
        self.goal_left = self.goal_x - self.goal_width / 2
        self.goal_right = self.goal_x + self.goal_width / 2

        self.action_space = spaces.Box(low=-50.0, high=50.0, shape=(1,), dtype=np.float32)
        
        obs_low = np.array([0, 0, 0, -np.inf, -np.inf], dtype=np.float32)
        obs_high = np.array([width, width, height, np.inf, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.goalkeeper_pos = np.array([self.goal_x, height - 30], dtype=np.float32)

        self.ball_pos = None
        self.ball_vel = None
        self.done = False
        self.last_reward = 0.0
        self.reset()

    def reset(self, seed=None, options=None):
        """
        Reinicia o ambiente para o início de um episódio:
        - Reposiciona o goleiro no centro do gol.
        - Posiciona a bola no centro do topo do campo.
        - Chuta a bola em direção a uma posição aleatória dentro do gol.
        """
        self.goalkeeper_pos = np.array([self.goal_x, self.height - 30], dtype=np.float32)
        self.done = False

        self.ball_pos = np.array([self.width / 2, 0], dtype=np.float32)
        
        goal_left = self.goal_x - self.goal_width / 2
        goal_right = self.goal_x + self.goal_width / 2
        target_x = random.uniform(goal_left, goal_right)
        target_y = self.height
        speed = 10.0
        vec = np.array([target_x, target_y]) - self.ball_pos
        norm = np.linalg.norm(vec) + 1e-5
        self.ball_vel = speed * vec / norm

        self.last_reward = 0.0
        return self._get_state(), {}


    def _get_state(self):
        return np.array([
            self.goalkeeper_pos[0],
            self.ball_pos[0],
            self.ball_pos[1],
            self.ball_vel[0],
            self.ball_vel[1]
        ], dtype=np.float32)

    def step(self, action):
        # Atualiza a posição do goleiro e restringe à área do gol
        self.goalkeeper_pos[0] += action[0]
        self.goalkeeper_pos[0] = np.clip(self.goalkeeper_pos[0], self.goal_left, self.goal_right)
        
        # Atualiza a posição da bola
        self.ball_pos += self.ball_vel
        reward = 0.0

        # Se a bola atingir a linha do gol
        if self.ball_pos[1] >= self.height:
            # Calcula a distância Euclidiana entre o goleiro e a bola
            d = np.linalg.norm(self.goalkeeper_pos - self.ball_pos)
            # Se a distância for menor que 23 pixels (15+8), considere que houve colisão/defesa
            if d < 23:
                reward = 1.0  # Defesa bem-sucedida
            else:
                reward = -1.0  # Gol sofrido
            self.done = True

        # Se a bola sair lateralmente do campo, penaliza
        if self.ball_pos[0] < 0 or self.ball_pos[0] > self.width:
            reward = -0.5
            self.done = True

        self.last_reward = reward
        return self._get_state(), reward, self.done, False, {}

    def render(self, mode="human"):
        img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        cv2.rectangle(img, (0, 0), (self.width, self.height), (0, 255, 0), 2)
        goal_left = int(self.goal_x - self.goal_width / 2)
        goal_right = int(self.goal_x + self.goal_width / 2)
        cv2.rectangle(img, (goal_left, self.height - self.goal_height),
                      (goal_right, self.height), (0, 0, 255), 2)
        cv2.circle(img, (int(self.goalkeeper_pos[0]), int(self.goalkeeper_pos[1])), 15, (255, 0, 0), -1)
        cv2.circle(img, (int(self.ball_pos[0]), int(self.ball_pos[1])), 8, (0, 0, 0), -1)
        cv2.putText(img, f"Reward: {self.last_reward:.2f}", 
                    (self.width - 220, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow("Soccer Field", img)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()
