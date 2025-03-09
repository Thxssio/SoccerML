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

        # Configurações do gol
        self.goal_width = 200  
        self.goal_height = 50  
        self.goal_x = width // 2 
        self.goal_y = height     
        self.goal_left = self.goal_x - self.goal_width / 2
        self.goal_right = self.goal_x + self.goal_width / 2

        # Espaço de ação: deslocamento horizontal do goleiro (valor contínuo)
        self.action_space = spaces.Box(low=-50.0, high=50.0, shape=(1,), dtype=np.float32)
        
        # Espaço de observação: [goalkeeper_x, ball_x, ball_y, ball_vel_x, ball_vel_y]
        obs_low = np.array([0, 0, 0, -100.0, -100.0], dtype=np.float32)
        obs_high = np.array([width, width, height, 100.0, 100.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Posição inicial do goleiro (centrado na área do gol)
        self.goalkeeper_pos = np.array([self.goal_x, height - 30], dtype=np.float32)

        self.ball_pos = None
        self.ball_vel = None
        self.done = False
        self.last_reward = 0.0
        self.reset()

    def reset(self, seed=None, options=None):
        """
        Reinicia o ambiente:
         - Reposiciona o goleiro no centro do gol.
         - Posiciona a bola no centro do topo do campo.
         - Chuta a bola em direção a uma posição aleatória dentro do gol.
        """
        self.goalkeeper_pos = np.array([self.goal_x, self.height - 30], dtype=np.float32)
        self.done = False

        # A bola sempre começa no centro do topo
        self.ball_pos = np.array([self.width / 2, 0], dtype=np.float32)
        # O atacante chuta a bola sempre em direção ao gol
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
        # Atualiza a posição do goleiro e restringe o movimento à área do gol
        self.goalkeeper_pos[0] += action[0]
        self.goalkeeper_pos[0] = np.clip(self.goalkeeper_pos[0], self.goal_left, self.goal_right)
        
        # Atualiza a posição da bola
        self.ball_pos += self.ball_vel
        reward = 0.0

        # Quando a bola atinge a área do gol (parte inferior do campo)
        if self.ball_pos[1] >= self.height - self.goal_height:
            goalie_radius = 20
            goalie_box = {
                "x_min": self.goalkeeper_pos[0] - goalie_radius,
                "x_max": self.goalkeeper_pos[0] + goalie_radius,
                "y_min": self.goalkeeper_pos[1] - goalie_radius,
                "y_max": self.goalkeeper_pos[1] + goalie_radius
            }
            # Se o centro da bola estiver dentro do bounding box, defesa ocorreu
            if (goalie_box["x_min"] <= self.ball_pos[0] <= goalie_box["x_max"] and
                goalie_box["y_min"] <= self.ball_pos[1] <= goalie_box["y_max"]):
                reward = 1.0  # Defesa bem-sucedida
                # Congela a bola no ponto de contato
                self.ball_vel = np.array([0.0, 0.0])
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
        # Cria uma imagem branca minimalista
        img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        
        # Desenha as bordas do campo (linha verde)
        cv2.rectangle(img, (0, 0), (self.width, self.height), (0, 255, 0), 2)
        
        # Desenha o gol (retângulo vermelho simples)
        goal_left = int(self.goal_x - self.goal_width / 2)
        goal_right = int(self.goal_x + self.goal_width / 2)
        cv2.rectangle(img, (goal_left, self.height - self.goal_height),
                      (goal_right, self.height), (0, 0, 255), 2)
        
        # Desenha o goleiro (círculo azul)
        cv2.circle(img, (int(self.goalkeeper_pos[0]), int(self.goalkeeper_pos[1])), 15, (255, 0, 0), -1)
        # Desenha a bola (círculo preto)
        cv2.circle(img, (int(self.ball_pos[0]), int(self.ball_pos[1])), 8, (0, 0, 0), -1)
        
        # Exibe a recompensa no canto superior direito
        cv2.putText(img, f"Reward: {self.last_reward:.2f}", 
                    (self.width - 220, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        cv2.imshow("Soccer Field", img)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()
