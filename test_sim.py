import cv2
import numpy as np
import time
from src.environment.soccer_env import SoccerEnv  # Ajuste o caminho se necessário

def main():
    env = SoccerEnv(width=640, height=480)
    state, _ = env.reset()
    done = False
    while not done:
        # Para testar, usamos ação zero: o goleiro não se move
        action = np.array([0.0], dtype=np.float32)
        state, reward, done, truncated, info = env.step(action)
        print("Estado:", state, "Recompensa:", reward, "Done:", done)
        env.render()
        time.sleep(0.05)  # pequena pausa para visualização
    print("Episódio finalizado com recompensa:", reward)
    env.close()

if __name__ == "__main__":
    main()
