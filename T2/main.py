import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.agent import Agent
from src.wrappers import create_env

# Hyperparameters
ENV_NAME = "Pong-v5"
N_STACK_FRAMES = 4
N_EPISODES = 500
LR = 1e-4
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 100000
TARGET_UPDATE_FREQUENCY = 1000
SHAPE = 84


def plot_rewards(rewards):
    plt.figure(figsize=(10, 5))
    plt.title("Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.plot(rewards)
    # Calculate and plot a moving average
    moving_avg = np.convolve(rewards, np.ones(100) / 100, mode="valid")
    plt.plot(np.arange(99, len(rewards)), moving_avg, label="100-episode MA")
    plt.legend()
    plt.grid(True)
    plt.savefig("rewards.png")
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = create_env(ENV_NAME, SHAPE, N_STACK_FRAMES)
    n_actions = env.action_space.n

    agent = Agent(
        n_stack_frames=N_STACK_FRAMES,
        n_actions=n_actions,
        device=device,
        lr=LR,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        batch_size=BATCH_SIZE,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        target_update_frequency=TARGET_UPDATE_FREQUENCY,
    )

    episode_rewards = []

    for episode in tqdm(range(N_EPISODES)):
        state, _ = env.reset()
        state = torch.from_numpy(state).float().to(device)
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            total_reward += reward

            reward = torch.tensor([reward], device=device)
            done_tensor = torch.tensor([done], device=device, dtype=torch.bool)
            next_state = torch.from_numpy(next_state).float().to(device)

            agent.memory.push(state, action, next_state, reward, done_tensor)

            state = next_state
            agent.train_step()

        episode_rewards.append(total_reward)

    env.close()
    plot_rewards(episode_rewards)
    torch.save(agent.policy_net.state_dict(), f"{ENV_NAME}_dqn_model.pth")
    print("Training complete. Model saved.")


if __name__ == "__main__":
    main()
