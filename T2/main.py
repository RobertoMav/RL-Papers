import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.agent import Agent
from src.wrappers import create_env

# Hyperparameters

ENV_NAME = "ALE/Pong-v5"
N_STACK_FRAMES = 4
N_EPISODES = 100
LR = 1e-4
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 100000
TARGET_UPDATE_FREQUENCY = 1000
SHAPE = 84


def plot_rewards(rewards, filename_prefix: str = "rewards"):
    """
    Plot rewards with timestamp to avoid overwriting files.

    Args:
        rewards: List of rewards to plot
        filename_prefix: Prefix for the filename (default: "rewards")
    """
    plt.figure(figsize=(10, 5))
    plt.title("Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.plot(rewards, label="Episode Rewards")

    # Calculate and plot a moving average only if we have enough episodes
    if len(rewards) >= 100:
        window_size = 100
        moving_avg = np.convolve(rewards, np.ones(window_size) / window_size, mode="valid")
        plt.plot(np.arange(window_size - 1, len(rewards)), moving_avg, label="100-episode MA")
    elif len(rewards) >= 10:
        # Use a smaller window if we have at least 10 episodes
        window_size = min(10, len(rewards))
        moving_avg = np.convolve(rewards, np.ones(window_size) / window_size, mode="valid")
        plt.plot(
            np.arange(window_size - 1, len(rewards)), moving_avg, label=f"{window_size}-episode MA"
        )

    plt.legend()
    plt.grid(True)

    # Create plots directory if it doesn't exist
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)

    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{plots_dir}/{filename_prefix}_{timestamp}.png"
    plt.savefig(filename)
    print(f"Plot saved as: {filename}")
    plt.close()  # Close the figure to free memory


def demonstrate_agent(model_path: str, n_demo_episodes: int = 3, save_results: bool = True):
    """
    Load the trained model and demonstrate the agent playing the game.

    Args:
        model_path: Path to the saved model weights
        n_demo_episodes: Number of episodes to demonstrate
        save_results: Whether to save demonstration results to CSV
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Demonstrating trained agent on device: {device}")

    # Create environment with rendering
    env = create_env(ENV_NAME, SHAPE, N_STACK_FRAMES)
    n_actions = env.action_space.n

    # Create agent and load trained weights
    agent = Agent(
        n_stack_frames=N_STACK_FRAMES,
        n_actions=n_actions,
        device=device,
        lr=LR,
        gamma=GAMMA,
        epsilon_start=0.0,  # No exploration during demonstration
        epsilon_end=0.0,
        epsilon_decay=EPSILON_DECAY,
        batch_size=BATCH_SIZE,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        target_update_frequency=TARGET_UPDATE_FREQUENCY,
    )

    # Load the trained model
    agent.policy_net.load_state_dict(torch.load(model_path, map_location=device))
    agent.policy_net.eval()  # Set to evaluation mode

    print(f"\nDemonstrating agent for {n_demo_episodes} episodes...")
    demo_rewards = []
    episode_data = []  # Store detailed episode information

    for episode in range(n_demo_episodes):
        state, _ = env.reset()
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)
        total_reward = 0
        done = False
        step_count = 0

        print(f"\nEpisode {episode + 1}/{n_demo_episodes}")

        while not done:
            # Select action (pure exploitation - no exploration)
            with torch.no_grad():
                action = agent.policy_net(state).max(1)[1].view(1, 1)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            total_reward += reward
            step_count += 1

            state = torch.from_numpy(next_state).float().to(device).unsqueeze(0)

            # Print progress every 100 steps
            if step_count % 100 == 0:
                print(f"  Step {step_count}, Current reward: {total_reward}")

        demo_rewards.append(total_reward)
        episode_data.append(
            {"episode": episode + 1, "steps": step_count, "total_reward": total_reward}
        )
        print(
            f"  Episode {episode + 1} finished: {step_count} steps, Total reward: {total_reward}"
        )

    env.close()

    # Print demonstration statistics
    print(f"\n{'=' * 50}")
    print("DEMONSTRATION RESULTS")
    print(f"{'=' * 50}")
    print(f"Episodes played: {n_demo_episodes}")
    print(f"Average reward: {np.mean(demo_rewards):.2f}")
    print(f"Best episode reward: {np.max(demo_rewards):.2f}")
    print(f"Worst episode reward: {np.min(demo_rewards):.2f}")
    print(f"Reward std: {np.std(demo_rewards):.2f}")

    # Save results to CSV if requested
    if save_results:
        import csv
        import os

        # Create results folder
        results_folder = "demonstration_results"
        os.makedirs(results_folder, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{results_folder}/demo_results_{timestamp}.csv"

        # Write episode data to CSV
        with open(csv_filename, "w", newline="") as csvfile:
            fieldnames = ["episode", "steps", "total_reward"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(episode_data)

        # Write summary statistics to separate CSV
        summary_filename = f"{results_folder}/demo_summary_{timestamp}.csv"
        summary_data = [
            {"metric": "Episodes played", "value": n_demo_episodes},
            {"metric": "Average reward", "value": f"{np.mean(demo_rewards):.2f}"},
            {"metric": "Best episode reward", "value": f"{np.max(demo_rewards):.2f}"},
            {"metric": "Worst episode reward", "value": f"{np.min(demo_rewards):.2f}"},
            {"metric": "Reward std", "value": f"{np.std(demo_rewards):.2f}"},
        ]

        with open(summary_filename, "w", newline="") as csvfile:
            fieldnames = ["metric", "value"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_data)

        print("\nDemonstration results saved to:")
        print(f"  Episode details: {csv_filename}")
        print(f"  Summary statistics: {summary_filename}")

    return demo_rewards


def create_render_env(
    env_name: str,
    shape: int = 84,
    k: int = 4,
    record_video: bool = False,
    video_folder: str = "videos",
    video_name_prefix: str = "demo",
):
    """Create environment with rendering enabled for demonstration."""
    import ale_py
    import gymnasium as gym
    from gymnasium.wrappers import RecordVideo

    from src.wrappers import FrameStack, GrayScaleObservation, ResizeObservation

    gym.register_envs(ale_py)

    # Create environment with human rendering
    env = gym.make(env_name, render_mode="rgb_array" if record_video else "human")

    # Add video recording wrapper if requested
    if record_video:
        # Create unique video name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = f"{video_name_prefix}_{timestamp}"
        env = RecordVideo(
            env, video_folder=video_folder, episode_trigger=lambda x: True, name_prefix=video_name
        )

    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape)
    env = FrameStack(env, k)
    return env


def demonstrate_agent_visual(model_path: str, n_demo_episodes: int = 5, save_video: bool = True):
    """
    Load the trained model and demonstrate the agent playing with visual rendering.

    Args:
        model_path: Path to the saved model weights
        n_demo_episodes: Number of episodes to demonstrate with rendering
        save_video: Whether to save video recordings of the episodes
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Demonstrating trained agent visually on device: {device}")

    # Create video folder if saving videos
    video_folder = "demonstration_videos"
    if save_video:
        import os

        os.makedirs(video_folder, exist_ok=True)
        print(f"Videos will be saved to: {video_folder}/")

    # Create environment with visual rendering
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_prefix = f"visual_demo_{timestamp}"
    env = create_render_env(
        ENV_NAME,
        SHAPE,
        N_STACK_FRAMES,
        record_video=save_video,
        video_folder=video_folder,
        video_name_prefix=video_prefix,
    )
    n_actions = env.action_space.n

    # Create agent and load trained weights
    agent = Agent(
        n_stack_frames=N_STACK_FRAMES,
        n_actions=n_actions,
        device=device,
        lr=LR,
        gamma=GAMMA,
        epsilon_start=0.0,  # No exploration during demonstration
        epsilon_end=0.0,
        epsilon_decay=EPSILON_DECAY,
        batch_size=BATCH_SIZE,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        target_update_frequency=TARGET_UPDATE_FREQUENCY,
    )

    # Load the trained model
    agent.policy_net.load_state_dict(torch.load(model_path, map_location=device))
    agent.policy_net.eval()  # Set to evaluation mode

    print(f"\nStarting visual demonstration for {n_demo_episodes} episodes...")
    print("Close the game window to end the demonstration early.")

    demo_rewards = []

    try:
        for episode in range(n_demo_episodes):
            state, _ = env.reset()
            state = torch.from_numpy(state).float().to(device).unsqueeze(0)
            total_reward = 0
            done = False
            step_count = 0

            print(f"\nEpisode {episode + 1}/{n_demo_episodes} - Watch the game window!")

            while not done:
                # Select action (pure exploitation - no exploration)
                with torch.no_grad():
                    action = agent.policy_net(state).max(1)[1].view(1, 1)

                next_state, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated
                total_reward += reward
                step_count += 1

                state = torch.from_numpy(next_state).float().to(device).unsqueeze(0)

                # Add small delay to make it watchable (optional)
                import time

                time.sleep(0.01)  # 10ms delay

            demo_rewards.append(total_reward)
            print(
                f"  Episode {episode + 1} finished: {step_count} steps, "
                f"Total reward: {total_reward}"
            )

    except Exception as e:
        print(f"Demonstration ended: {e}")
    finally:
        env.close()

    # Print video saving information
    if save_video:
        print(f"\nVideos saved to: {video_folder}/")
        print(f"Video files use prefix: {video_prefix}")
        print("You can find the recorded episodes as MP4 files in the videos folder.")

    return demo_rewards


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
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

    print("Starting training...")
    for episode in tqdm(range(N_EPISODES), desc="Training Episodes"):
        state, _ = env.reset()
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)  # Add batch dimension
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            total_reward += reward

            reward = torch.tensor([reward], device=device)
            done_tensor = torch.tensor([done], device=device, dtype=torch.bool)
            next_state = (
                torch.from_numpy(next_state).float().to(device).unsqueeze(0)
            )  # Add batch dimension

            agent.memory.push(
                state.squeeze(0), action, next_state.squeeze(0), reward, done_tensor
            )  # Remove batch dimension for storage

            state = next_state
            agent.train_step()

        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1} reward: {total_reward}")

    env.close()

    # Plot training results
    plot_rewards(episode_rewards, "training_rewards")

    # Save the trained model
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    clean_env_name = ENV_NAME.replace("/", "_")
    model_path = f"{model_dir}/{clean_env_name}_dqn_model.pth"
    torch.save(agent.policy_net.state_dict(), model_path)
    print(f"Training complete. Model saved to: {model_path}")

    # Demonstrate the trained agent
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - DEMONSTRATING TRAINED AGENT")
    print("=" * 60)

    # Non-visual demonstration (faster, with statistics)
    demo_rewards = demonstrate_agent(model_path, n_demo_episodes=2)
    visual_rewards = demonstrate_agent_visual(model_path, n_demo_episodes=2)
    print(
        f"\nAll done! Check '{model_path}' for the trained model and "
        "timestamped reward plots for training curves."
    )

    plot_rewards(demo_rewards, "demonstration_rewards")
    plot_rewards(visual_rewards, "visual_demonstration_rewards")


if __name__ == "__main__":
    main()
