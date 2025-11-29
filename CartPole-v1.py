import gymnasium as gym
from gymnasium.utils.play import play
import pygame

# --- Configurações ---
ENV_NAME = "CartPole-v1"

# 1. Obter o modelo Gymnasium
env = gym.make(ENV_NAME, render_mode="rgb_array")

# 2. Definir o mapeamento de teclas (Controle Bang-Bang)
mapping = {
    pygame.K_LEFT: 0,  # Tecla Seta Esquerda -> Ação 0
    pygame.K_RIGHT: 1, # Tecla Seta Direita -> Ação 1
}

# 3. Rodar a função de 'play'
play(
    env,               
    keys_to_action=mapping, 
    fps=30              
)

env.close()