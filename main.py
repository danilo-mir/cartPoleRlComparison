#!/usr/bin/env python3
"""
Main training script for comparing Q-Learning and SARSA on Gymnasium's
CartPole-v1 environment with state discretization.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import gymnasium as gym
import matplotlib

# Use a non-interactive backend by default to avoid display issues.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

# --- Hyperparameters and global configuration ---
ENV_NAME = "CartPole-v1"
NUM_EPISODES = 5000
MAX_STEPS_PER_EPISODE = 500
ALPHA = 0.1  # Learning rate
GAMMA = 0.99  # Discount factor
EPSILON_START = 1.0
EPSILON_MIN = 0.02
EPSILON_DECAY = 0.995
SEED = 42
MOVING_AVG_WINDOW = 100
RENDER_FINAL_POLICY = True  # Set True to watch the trained agent
FINAL_POLICY_EPISODES = 1

# Bucketing configuration for each dimension:
#   cart position, cart velocity, pole angle, pole angular velocity
NUM_BINS = (8, 8, 16, 16)
OBSERVATION_BOUNDS = np.array(
    [
        (-4.8, 4.8),  # cart position
        (-3.0, 3.0),  # cart velocity (clipped)
        (-0.418, 0.418),  # pole angle (approx. 24 degrees)
        (-3.5, 3.5),  # pole angular velocity (clipped)
    ],
    dtype=np.float32,
)


def build_bin_edges(
    num_bins: Sequence[int], bounds: np.ndarray
) -> List[np.ndarray]:
    """Create equally spaced bin edges for each observation dimension."""
    edges = []
    for bin_count, (low, high) in zip(num_bins, bounds):
        if bin_count < 2:
            raise ValueError("Each dimension must have at least two bins.")
        edges.append(np.linspace(low, high, bin_count - 1))
    return edges


BIN_EDGES = build_bin_edges(NUM_BINS, OBSERVATION_BOUNDS)


def discretize_state(observation: np.ndarray) -> Tuple[int, ...]:
    """Convert a continuous observation into a tuple of discrete indices."""
    clipped = np.clip(
        observation,
        OBSERVATION_BOUNDS[:, 0],
        OBSERVATION_BOUNDS[:, 1],
    )
    indices = [
        int(np.digitize(value, edges))
        for value, edges in zip(clipped, BIN_EDGES)
    ]
    return tuple(indices)


def epsilon_greedy(
    q_table: np.ndarray, state: Tuple[int, ...], epsilon: float, n_actions: int
) -> int:
    """Select an action using the epsilon-greedy policy."""
    if np.random.random() < epsilon:
        return int(np.random.randint(n_actions))
    return int(np.argmax(q_table[state]))


def initialize_q_table(n_actions: int) -> np.ndarray:
    """Create a zero-initialized Q-table with the correct shape."""
    return np.zeros(NUM_BINS + (n_actions,), dtype=np.float32)


def run_episode_q_learning(
    env: gym.Env, q_table: np.ndarray, epsilon: float, seed: int
) -> float:
    """Run a single Q-Learning episode and update the Q-table in-place."""
    observation, _ = env.reset(seed=seed)
    state = discretize_state(observation)
    total_reward = 0.0
    for _ in range(MAX_STEPS_PER_EPISODE):
        action = epsilon_greedy(q_table, state, epsilon, env.action_space.n)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_state = discretize_state(next_obs)
        done = terminated or truncated

        best_next_q = float(np.max(q_table[next_state]))
        target = reward + GAMMA * best_next_q * (not done)
        # Q-Learning update (off-policy): Q(s,a) ← Q(s,a) + α [target − Q(s,a)]
        q_table[state + (action,)] += ALPHA * (
            target - q_table[state + (action,)]
        )

        state = next_state
        total_reward += reward
        if done:
            break
    return total_reward


def run_episode_sarsa(
    env: gym.Env, q_table: np.ndarray, epsilon: float, seed: int
) -> float:
    """Run a single SARSA episode and update the Q-table in-place."""
    observation, _ = env.reset(seed=seed)
    state = discretize_state(observation)
    action = epsilon_greedy(q_table, state, epsilon, env.action_space.n)
    total_reward = 0.0
    for _ in range(MAX_STEPS_PER_EPISODE):
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_state = discretize_state(next_obs)
        done = terminated or truncated
        next_action = epsilon_greedy(
            q_table, next_state, epsilon, env.action_space.n
        )

        target = reward + GAMMA * q_table[next_state + (next_action,)] * (
            not done
        )
        # SARSA update (on-policy): Q(s,a) ← Q(s,a) + α [target − Q(s,a)]
        q_table[state + (action,)] += ALPHA * (
            target - q_table[state + (action,)]
        )

        state = next_state
        action = next_action
        total_reward += reward
        if done:
            break
    return total_reward


def decay_epsilon(epsilon: float) -> float:
    """Apply exponential decay to epsilon while enforcing the minimum."""
    return max(EPSILON_MIN, epsilon * EPSILON_DECAY)


def train_agent(
    algorithm: str,
) -> Tuple[np.ndarray, List[float]]:
    """Train either a Q-Learning or SARSA agent."""
    env = gym.make(ENV_NAME)
    env.reset(seed=SEED)
    np.random.seed(SEED)

    q_table = initialize_q_table(env.action_space.n)
    rewards: List[float] = []
    epsilon = EPSILON_START

    for episode in range(NUM_EPISODES):
        episode_seed = SEED + episode
        if algorithm == "q_learning":
            episode_reward = run_episode_q_learning(
                env, q_table, epsilon, seed=episode_seed
            )
        elif algorithm == "sarsa":
            episode_reward = run_episode_sarsa(
                env, q_table, epsilon, seed=episode_seed
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        rewards.append(episode_reward)
        epsilon = decay_epsilon(epsilon)
        if (episode + 1) % 100 == 0:
            print(
                f"[{algorithm.upper()}] Episode {episode+1}/{NUM_EPISODES} | "
                f"Reward: {episode_reward:.1f} | Epsilon: {epsilon:.3f}"
            )

    env.close()
    return q_table, rewards


def moving_average(data: Sequence[float], window: int) -> np.ndarray:
    """Compute a centered moving average for smoothing learning curves."""
    if not data:
        return np.array([], dtype=np.float32)
    window = min(window, len(data))
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def plot_learning_curves(
    q_learning_rewards: Sequence[float], sarsa_rewards: Sequence[float]
) -> None:
    """Plot and save the comparative learning curves."""
    q_curve = moving_average(q_learning_rewards, MOVING_AVG_WINDOW)
    s_curve = moving_average(sarsa_rewards, MOVING_AVG_WINDOW)
    q_x = np.arange(len(q_curve)) + MOVING_AVG_WINDOW - 1
    s_x = np.arange(len(s_curve)) + MOVING_AVG_WINDOW - 1

    plt.figure(figsize=(10, 6))
    plt.plot(q_x, q_curve, label="Q-Learning")
    plt.plot(s_x, s_curve, label="SARSA")
    plt.xlabel("Episódio")
    plt.ylabel(f"Recompensa (média móvel {MOVING_AVG_WINDOW})")
    plt.title("CartPole-v1: Q-Learning vs SARSA")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = Path("cartpole_q_vs_sarsa.png").resolve()
    plt.savefig(output_path)
    print(f"Gráfico salvo em: {output_path}")

    try:
        plt.show()
    except Exception:
        # On headless servers show() might fail; saving the plot is enough.
        pass


def run_trained_agent(q_table: np.ndarray, episodes: int = 1) -> None:
    """Render a trained policy for qualitative inspection."""
    env = gym.make(ENV_NAME, render_mode="human")
    try:
        for ep in range(episodes):
            observation, _ = env.reset(seed=SEED + 10_000 + ep)
            state = discretize_state(observation)
            total_reward = 0.0
            for _ in range(MAX_STEPS_PER_EPISODE):
                action = int(np.argmax(q_table[state]))
                observation, reward, terminated, truncated, _ = env.step(action)
                state = discretize_state(observation)
                total_reward += reward
                if terminated or truncated:
                    break
            print(f"Episódio de demonstração {ep + 1}: recompensa={total_reward}")
    finally:
        env.close()


def main() -> None:
    """Train both agents, compare performance, and optionally render."""
    q_table, q_learning_rewards = train_agent("q_learning")
    sarsa_table, sarsa_rewards = train_agent("sarsa")
    plot_learning_curves(q_learning_rewards, sarsa_rewards)

    if RENDER_FINAL_POLICY:
        print("Renderizando política final (Q-Learning).")
        run_trained_agent(q_table, episodes=FINAL_POLICY_EPISODES)


if __name__ == "__main__":
    main()

