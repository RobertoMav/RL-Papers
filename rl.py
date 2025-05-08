import math
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # For a nicer heatmap aesthetic
from gymnasium.wrappers import RecordVideo


def value_iteration(
    env, gamma: float = 0.99, theta: float = 1e-9
) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs Value Iteration to find the optimal value function and policy
    for a given environment.

    Args:
        env: The OpenAI Gym environment. It must have `env.observation_space.n`
             for the number of states, `env.action_space.n` for the number of
             actions, and `env.P` for the transition dynamics.
        gamma: The discount factor (0 < gamma <= 1).
        theta: A small positive number, the threshold for convergence.
               Iteration stops when the maximum change in the value function
               across all states is less than theta.

    Returns:
        A tuple (optimal_value_function, optimal_policy):
        - optimal_value_function: A 1D NumPy array where V[s] is the
                                  optimal value of state s.
        - optimal_policy: A 1D NumPy array where P[s] is the optimal
                          action to take in state s.
    """
    num_states: int = env.observation_space.n
    num_actions: int = env.action_space.n

    # Initialize value function V(s) to zeros
    value_function: np.ndarray = np.zeros(num_states)

    # Iteratively update the value function
    while True:
        delta: float = 0.0  # Stores the maximum change in V(s) during this iteration
        for s in range(num_states):
            old_v: float = value_function[s]
            action_values: np.ndarray = np.zeros(num_actions)

            # Calculate the value for each action in the current state s
            for a in range(num_actions):
                # env.P[s][a] is a list of (probability, next_state, reward, terminated) tuples
                # Accessing P from the unwrapped environment
                for prob, next_state, reward, terminated in env.unwrapped.P[s][a]:
                    action_values[a] += prob * (reward + gamma * value_function[next_state])

            # Update V(s) to the maximum action value
            value_function[s] = np.max(action_values)
            delta = max(delta, abs(old_v - value_function[s]))

        # Check for convergence
        if delta < theta:
            break

    # Extract the optimal policy once the value function has converged
    optimal_policy: np.ndarray = np.zeros(num_states, dtype=int)
    for s in range(num_states):
        action_values = np.zeros(num_actions)
        for a in range(num_actions):
            # Accessing P from the unwrapped environment
            for prob, next_state, reward, terminated in env.unwrapped.P[s][a]:
                action_values[a] += prob * (reward + gamma * value_function[next_state])

        # Choose the action that maximizes the expected value
        optimal_policy[s] = np.argmax(action_values)

    return value_function, optimal_policy


def print_policy_grid(
    policy: np.ndarray, env_description: np.ndarray, grid_shape: tuple[int, int]
) -> None:
    """
    Prints the policy in a human-readable grid format.
    Action mapping: 0: Left (←), 1: Down (↓), 2: Right (→), 3: Up (↑)
    """
    action_symbols: list[str] = ["←", "↓", "→", "↑"]
    print("\nOptimal Policy Grid:")

    policy_reshaped = policy.reshape(grid_shape)

    for r in range(grid_shape[0]):
        row_str: list[str] = []
        for c in range(grid_shape[1]):
            state_char: str = env_description[r, c].decode("utf-8")
            if state_char in ("G", "H"):  # Goal or Hole states
                row_str.append(state_char)
            else:
                state_index: int = r * grid_shape[1] + c
                row_str.append(action_symbols[policy[state_index]])
        print(" ".join(row_str))


def print_value_function_grid(value_function: np.ndarray, grid_shape: tuple[int, int]) -> None:
    """Prints the value function in a grid format, rounded for readability."""
    print("\nOptimal Value Function Grid:")
    print(np.round(value_function.reshape(grid_shape), 3))


def plot_value_function_heatmap(
    value_function: np.ndarray,
    grid_shape: tuple[int, int],
    title: str,
    env_description: np.ndarray | None = None,
    save_path: str | None = None,  # New argument to specify where to save the plot
) -> None:
    """Plots the value function as a heatmap."""
    plt.figure(
        figsize=(grid_shape[1] * 1.0, grid_shape[0] * 1.0)
    )  # Slightly increased size for better saving
    reshaped_values = value_function.reshape(grid_shape)
    ax = sns.heatmap(reshaped_values, annot=True, fmt=".3f", cmap="viridis", cbar=True)
    plt.title(title)
    plt.xlabel("Column")
    plt.ylabel("Row")

    # Overlay S, F, H, G characters if env_description is available
    if env_description is not None:
        for r in range(grid_shape[0]):
            for c in range(grid_shape[1]):
                char_state = env_description[r, c].decode("utf-8")
                # Add text, ensuring visibility against heatmap colors
                text_color = "black" if reshaped_values[r, c] > reshaped_values.mean() else "white"
                if char_state in ["S", "G", "H"]:
                    ax.text(
                        c + 0.5,
                        r + 0.5,
                        char_state,
                        ha="center",
                        va="center",
                        color=text_color,
                        weight="bold",
                    )

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Heatmap saved to {save_path}")
    plt.show()


# --- Evaluation Functions ---


def run_single_episode(
    env, policy: np.ndarray, max_steps: int = 100
) -> tuple[list[tuple[int, int, float, int, bool]], float]:
    """Runs a single episode using the given policy.

    Args:
        env: The OpenAI Gym environment.
        policy: A 1D NumPy array where P[s] is the action to take in state s.
        max_steps: Maximum number of steps for the episode.

    Returns:
        A tuple (trajectory, total_reward):
        - trajectory: A list of (state, action, reward, next_state, terminated) tuples.
        - total_reward: The sum of rewards obtained in the episode.
    """
    trajectory: list[tuple[int, int, float, int, bool]] = []
    current_state, _ = env.reset()
    total_reward: float = 0.0

    for _step in range(max_steps):
        action: int = policy[current_state]
        next_state, reward, terminated, truncated, _info = env.step(action)
        done = terminated or truncated
        trajectory.append((current_state, action, float(reward), next_state, done))
        total_reward += float(reward)
        current_state = next_state
        if done:
            break
    return trajectory, total_reward


def evaluate_policy_performance(
    env, policy: np.ndarray, num_episodes: int = 100, max_steps_per_episode: int = 100
) -> list[float]:
    """Evaluates a policy by running it for a number of episodes.

    Args:
        env: The OpenAI Gym environment.
        policy: A 1D NumPy array where P[s] is the action to take in state s.
        num_episodes: The number of episodes to simulate.
        max_steps_per_episode: Maximum steps for each episode.

    Returns:
        A list of total rewards, one for each episode.
    """
    episode_rewards: list[float] = []
    for _episode in range(num_episodes):
        _trajectory, total_reward = run_single_episode(
            env, policy, max_steps=max_steps_per_episode
        )
        episode_rewards.append(total_reward)
    return episode_rewards


def plot_cumulative_rewards_comparison(
    dp_rewards: list[float],
    mcts_rewards: list[float],
    num_episodes: int,
    save_path: str | None = None,
) -> None:
    """Plots the cumulative rewards per episode for DP and MCTS policies.

    Args:
        dp_rewards: List of rewards per episode for the DP policy.
        mcts_rewards: List of rewards per episode for the MCTS policy.
        num_episodes: The total number of episodes evaluated.
        save_path: Optional path to save the plot image.
    """
    plt.figure(figsize=(12, 6))
    episodes = range(1, num_episodes + 1)

    # Calculate cumulative rewards
    cumulative_dp_rewards = np.cumsum(dp_rewards)
    cumulative_mcts_rewards = np.cumsum(mcts_rewards)

    plt.plot(
        episodes,
        cumulative_dp_rewards,
        label="DP Policy Cumulative Rewards",
        marker="o",
        linestyle="-",
    )
    plt.plot(
        episodes,
        cumulative_mcts_rewards,
        label="MCTS Policy Cumulative Rewards",
        marker="x",
        linestyle="--",
    )

    plt.title(f"Cumulative Rewards Comparison over {num_episodes} Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Cumulative rewards plot saved to {save_path}")
    plt.show()


# --- Monte Carlo Tree Search (MCTS) Implementation ---


def _sample_transition(env_p_s_a: list[tuple[float, int, float, bool]]) -> tuple[int, float, bool]:
    """Samples one (next_state, reward, terminated) from env.P[s][a]'s list.

    Args:
        env_p_s_a: The list of (probability, next_state, reward, terminated)
                   tuples for a given state-action pair from env.P.

    Returns:
        A tuple (next_state, reward, terminated) for the sampled transition.
    """
    probabilities: list[float] = [transition_info[0] for transition_info in env_p_s_a]
    outcome_indices: np.ndarray = np.arange(len(env_p_s_a))
    chosen_outcome_index: int = np.random.choice(outcome_indices, p=probabilities)

    _prob, next_state, reward, terminated = env_p_s_a[chosen_outcome_index]
    return next_state, reward, terminated


class MCTSNode:
    """Represents a node in the Monte Carlo Search Tree."""

    def __init__(
        self,
        state: int,
        parent: "MCTSNode | None" = None,
        action_taken: int | None = None,  # Action that led to this node
        is_terminal: bool = False,
        env_actions_n: int | None = None,  # Needed to initialize untried_actions
    ):
        self.state: int = state
        self.parent: MCTSNode | None = parent
        self.action_taken: int | None = action_taken  # Action that led from parent to this node
        self.children: list[MCTSNode] = []

        self.visits: int = 0
        self.value: float = 0.0  # Sum of rewards from rollouts through this node

        self.is_terminal_state: bool = is_terminal
        self.untried_actions: list[int] | None = None
        if not self.is_terminal_state and env_actions_n is not None:
            self.untried_actions = list(range(env_actions_n))
            random.shuffle(self.untried_actions)  # Explore in random order

    def is_fully_expanded(self) -> bool:
        """Checks if all possible actions from this node have been explored (led to a child)."""
        return self.untried_actions is not None and not self.untried_actions

    def add_child(
        self, action: int, child_state: int, is_terminal: bool, env_actions_n: int
    ) -> "MCTSNode":
        """Adds a new child to this node."""
        child_node = MCTSNode(
            state=child_state,
            parent=self,
            action_taken=action,
            is_terminal=is_terminal,
            env_actions_n=env_actions_n,
        )
        self.children.append(child_node)
        if self.untried_actions and action in self.untried_actions:
            self.untried_actions.remove(action)
        return child_node

    def update(self, reward: float) -> None:
        """Updates the node's visit count and accumulated value."""
        self.visits += 1
        self.value += reward

    def ucb1_score(self, exploration_constant: float = 1.414) -> float:
        """Calculates the UCB1 score for this node.
        Used by the parent to select which child to traverse.
        """
        if self.visits == 0:
            return float("inf")  # Ensure unvisited nodes are selected first

        # Q-value: Average reward obtained from this node
        average_reward: float = self.value / self.visits

        # Exploration term
        # Parent's visits are needed. If no parent (root node), this term is not directly applicable
        # for its own selection, but for its children's selection from its perspective.
        if self.parent is None:  # Should not happen if called on a child
            return average_reward

        exploration_term: float = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return average_reward + exploration_term

    def __repr__(self) -> str:
        return f"MCTSNode(s={self.state}, v={self.value:.2f}, n={self.visits}, childs={len(self.children)})"


def _simulate_rollout(
    start_state: int,
    env,  # Full environment access for P and action_space
    gamma: float,
    max_depth: int,
    terminal_states: set[int],
) -> float:
    """Simulates a random trajectory (rollout) from a start state.

    Args:
        start_state: The state to begin the rollout from.
        env: The OpenAI Gym environment (used for env.unwrapped.P and env.action_space.n).
        gamma: The discount factor.
        max_depth: Maximum number of steps in this rollout.
        terminal_states: A set of state indices that are terminal.

    Returns:
        The discounted sum of rewards obtained during the rollout.
    """
    current_state: int = start_state
    total_discounted_reward: float = 0.0

    for depth in range(max_depth):
        if current_state in terminal_states:
            # If we start in or reach a pre-identified terminal state (G or H for FrozenLake)
            # env.P might give a reward for the transition *into* G.
            # For FrozenLake, G has reward 1 on transition, H has 0.
            # If the *start_state* itself is terminal, no actions are taken, reward is inherent to state or 0.
            # Let's assume the reward is obtained when transitioning *into* the terminal state,
            # so if current_state is already terminal, no further rewards accumulate from this state.
            break

        # Choose a random action
        random_action: int = env.action_space.sample()

        # Sample the transition using our helper
        # env.unwrapped.P[current_state][random_action] will give the list of possible outcomes
        possible_outcomes = env.unwrapped.P[current_state][random_action]
        next_state, reward, terminated = _sample_transition(possible_outcomes)

        total_discounted_reward += (gamma**depth) * reward
        current_state = next_state

        if terminated:  # This terminated is from the environment dynamics
            break

    return total_discounted_reward


def mcts_search(
    root_state: int,
    env,  # Full environment access
    num_iterations: int,
    gamma: float,
    exploration_constant: float,
    max_rollout_depth: int,
    terminal_states: set[int],
) -> tuple[int, float, int]:
    """Performs MCTS search from a root state to find the best action.

    Args:
        root_state: The state from which to start the search.
        env: The OpenAI Gym environment.
        num_iterations: Number of MCTS iterations to run.
        gamma: Discount factor.
        exploration_constant: The C constant for UCB1.
        max_rollout_depth: Maximum depth for simulation rollouts.
        terminal_states: A set of state indices that are terminal.

    Returns:
        A tuple (best_action, root_value, root_visits):
        - best_action: The best action found for the root_state.
        - root_value: The value of the root node after the search.
        - root_visits: The number of visits to the root node after the search.
    """
    num_actions = env.action_space.n
    root_node = MCTSNode(
        state=root_state, env_actions_n=num_actions, is_terminal=(root_state in terminal_states)
    )

    if root_node.is_terminal_state:
        # If the root itself is terminal, there are no actions to choose meaningfully.
        # Depending on how MCTS is used, might return a default action or handle upstream.
        # For FrozenLake, if we are in G or H, we don't act.
        # Let's return an arbitrary action (e.g., 0) or raise an error/handle this scenario.
        # For now, we assume MCTS is called on non-terminal states for action selection.
        print(f"Warning: MCTS called on terminal state {root_state}. No valid action search.")
        return 0, 0.0, 0  # Or some other default/error indication

    for _ in range(num_iterations):
        current_node: MCTSNode = root_node

        # 1. Selection: Traverse the tree using UCB1 until a leaf or unexpanded node is found
        while (
            not current_node.is_terminal_state
            and current_node.is_fully_expanded()
            and current_node.children
        ):
            current_node = max(
                current_node.children, key=lambda node: node.ucb1_score(exploration_constant)
            )

        # 2. Expansion: If the selected node is not terminal and not fully expanded, expand it
        expanded_this_iteration = False
        if not current_node.is_terminal_state and current_node.untried_actions:
            action_to_expand: int = current_node.untried_actions.pop(0)  # Get an untried action

            # Simulate taking this action to get the next state
            possible_outcomes = env.unwrapped.P[current_node.state][action_to_expand]
            next_s, _reward_at_expansion, terminated_at_expansion = _sample_transition(
                possible_outcomes
            )

            # Note: The reward from expansion step is not directly used in MCTS value usually.
            # The value comes from the rollout.

            is_child_terminal = terminated_at_expansion or (next_s in terminal_states)
            child_node: MCTSNode = current_node.add_child(
                action_to_expand, next_s, is_child_terminal, num_actions
            )
            node_for_simulation: MCTSNode = child_node
            expanded_this_iteration = True
        else:
            # If it's terminal or fully expanded (but might be a leaf if it has no children yet tried)
            node_for_simulation = current_node

        # 3. Simulation (Rollout)
        # The reward from the simulation is from the perspective of the node_for_simulation's state
        simulated_reward: float = _simulate_rollout(
            node_for_simulation.state, env, gamma, max_rollout_depth, terminal_states
        )

        # 4. Backpropagation: Update visit counts and values up the tree
        temp_node: MCTSNode | None = node_for_simulation
        while temp_node is not None:
            # The reward needs to be discounted back to the parent from the child if a step was taken.
            # However, standard MCTS often backpropagates the direct simulation outcome.
            # For simplicity, let's backpropagate the simulated_reward directly.
            # More advanced: temp_node.update(gamma * previous_node_sim_reward + direct_reward_to_temp_node)
            temp_node.update(simulated_reward)
            temp_node = temp_node.parent

    # After all iterations, choose the best action from the root node
    if not root_node.children:
        # This can happen if num_iterations is too low or root is terminal-like
        # Or if all actions from root lead immediately to terminal states with no reward.
        print(
            f"Warning: MCTS root node {root_state} has no children after {num_iterations} iterations. Returning random action."
        )
        return env.action_space.sample(), 0.0, 0

    # Choose action leading to child with most visits (robust) or highest average value
    best_child: MCTSNode = max(root_node.children, key=lambda node: node.visits)
    # best_child: MCTSNode = max(root_node.children, key=lambda node: node.value / node.visits if node.visits > 0 else -float('inf'))
    return best_child.action_taken, root_node.value, root_node.visits


# --- End of MCTS Implementation ---


if __name__ == "__main__":
    # Create the FrozenLake-v1 environment
    # is_slippery=True by default, which makes it stochastic
    # render_mode=None as we are not rendering graphically here
    env_name: str = "FrozenLake-v1"
    environment = gym.make(env_name, render_mode=None)

    # For FrozenLake, env.desc provides the map.
    # For more general environments, shape might need to be inferred or passed.
    # `env.unwrapped` is used to access attributes of the original environment
    # if it's wrapped by other environment classes.
    env_desc_np: np.ndarray | None = None
    current_grid_shape: tuple[int, int]

    if hasattr(environment.unwrapped, "desc") and environment.unwrapped.desc is not None:
        env_desc_np = environment.unwrapped.desc
        current_grid_shape = env_desc_np.shape
    elif hasattr(environment.unwrapped, "nrow") and hasattr(environment.unwrapped, "ncol"):
        current_grid_shape = (environment.unwrapped.nrow, environment.unwrapped.ncol)
        # We would need to reconstruct a desc_np if we want to print S,F,H,G chars
        # For now, policy printing will just show arrows if desc is missing.
    else:
        # Fallback: try to infer a square grid if possible
        num_total_states = environment.observation_space.n
        side = int(np.sqrt(num_total_states))
        if side * side == num_total_states:
            current_grid_shape = (side, side)
        else:
            current_grid_shape = (1, num_total_states)  # Print as a single row
        print(
            f"Warning: Environment description (desc) not found or shape not standard. Displaying as {current_grid_shape} grid."
        )

    print(f"--- MDP Solution for {env_name} using Dynamic Programming (Value Iteration) ---")
    print(f"Number of states: {environment.observation_space.n}")
    print(f"Number of actions: {environment.action_space.n}")
    print("Action mapping: 0: Left, 1: Down, 2: Right, 3: Up")

    if env_desc_np is not None:
        print("\nEnvironment Layout:")
        for r_idx in range(env_desc_np.shape[0]):
            print("".join([cell.decode("utf-8") for cell in env_desc_np[r_idx]]))

    # Perform Value Iteration
    gamma_discount: float = 0.99
    convergence_theta: float = 1e-9
    opt_value_func, opt_policy = value_iteration(
        environment, gamma=gamma_discount, theta=convergence_theta
    )

    # Display results
    if env_desc_np is not None:
        print_policy_grid(opt_policy, env_desc_np, current_grid_shape)
    else:
        # Fallback if env.desc is not available for rich policy printing
        print("\nOptimal Policy (action indices):")
        print(opt_policy.reshape(current_grid_shape))

    print_value_function_grid(opt_value_func, current_grid_shape)
    # Add heatmap plotting for DP value function
    dp_heatmap_path = "dp_optimal_value_function_heatmap.png"
    if env_desc_np is not None:
        plot_value_function_heatmap(
            opt_value_func,
            current_grid_shape,
            "Optimal Value Function (DP)",
            env_desc_np,
            save_path=dp_heatmap_path,
        )
    else:
        plot_value_function_heatmap(
            opt_value_func,
            current_grid_shape,
            "Optimal Value Function (DP)",
            save_path=dp_heatmap_path,
        )

    # --- MCTS Execution ---
    print(f"\n\n--- MDP Solution for {env_name} using Monte Carlo Tree Search ---")

    # 1. Determine terminal states (G and H)
    terminal_states_indices = set()
    if env_desc_np is not None:
        for r_idx in range(current_grid_shape[0]):
            for c_idx in range(current_grid_shape[1]):
                char_state = env_desc_np[r_idx, c_idx].decode("utf-8")
                if char_state == "G" or char_state == "H":
                    terminal_states_indices.add(r_idx * current_grid_shape[1] + c_idx)
    else:
        # Fallback for standard 4x4 FrozenLake if description is not available
        # This is less robust and assumes the standard map layout.
        print(
            "Warning: env.desc not found. Attempting to use default terminal states for 4x4 FrozenLake for MCTS."
        )
        if environment.observation_space.n == 16:  # Standard 4x4 FrozenLake
            terminal_states_indices.add(5)  # Hole
            terminal_states_indices.add(7)  # Hole
            terminal_states_indices.add(11)  # Hole
            terminal_states_indices.add(12)  # Hole
            terminal_states_indices.add(15)  # Goal
        else:
            print(
                "Warning: Could not determine terminal states for MCTS. MCTS might behave unexpectedly."
            )

    # 2. MCTS Parameters
    # Note: MCTS can be computationally intensive.
    # Lower iterations for quicker testing, higher for potentially better policy.
    mcts_num_iterations: int = 2000  # Number of MCTS iterations per state decision
    mcts_exploration_constant: float = 1.414  # Typically sqrt(2)
    mcts_max_rollout_depth: int = (
        environment.observation_space.n * 2
    )  # Heuristic for rollout depth

    print(
        f"MCTS Parameters: iterations/state={mcts_num_iterations}, C_p={mcts_exploration_constant}, rollout_depth={mcts_max_rollout_depth}"
    )
    print(f"Identified terminal states for MCTS: {terminal_states_indices}")

    # 3. Generate MCTS Policy
    mcts_policy: np.ndarray = np.zeros(environment.observation_space.n, dtype=int)
    mcts_value_function_approx: np.ndarray = np.zeros(
        environment.observation_space.n, dtype=float
    )  # For storing MCTS state values
    print(
        "\nBuilding MCTS policy and approximating MCTS value function (this might take a moment for each state)..."
    )

    for s in range(environment.observation_space.n):
        if s in terminal_states_indices:
            mcts_policy[s] = 0  # Action for terminal states is arbitrary
            mcts_value_function_approx[s] = 0.0  # Value of terminal hole states is 0
            # For FrozenLake, reward for G is on transition, so V(G) itself can be 0 after reaching.
            # If state s is G (the goal), its value might be 1 if we consider the immediate reward.
            # However, V(s) from DP is future discounted rewards. For consistency in heatmap V(G)=0.
            # Let's find if s is the Goal state (G) to assign specific value if needed.
            if env_desc_np is not None:
                r, c = s // current_grid_shape[1], s % current_grid_shape[1]
                if env_desc_np[r, c].decode("utf-8") == "G":
                    mcts_value_function_approx[s] = (
                        0.0  # Consistent with DP V(G)=0 as it's terminal
                    )
                    # or 1.0 if we define V(G) as its immediate reward. Let's use 0 for now.

            print(
                f"  State {s}: Terminal, skipping MCTS search. Value set to {mcts_value_function_approx[s]}."
            )
            continue

        print(f"  Running MCTS for state {s}...")
        # We need the root node from mcts_search to get its value
        # Modifying mcts_search to optionally return the root node or just its value/visits might be cleaner.
        # For now, let's re-create the root node here to inspect after search (less ideal, but avoids changing mcts_search signature now)
        # A better way would be for mcts_search to return (best_action, root_value, root_visits)

        num_actions_at_s = environment.action_space.n
        # Create a temporary root node to pass to mcts_search, then analyze it.
        # This means mcts_search is robust and doesn't need to return the node itself.
        # The actual mcts_search will build its own tree internally starting from root_state.
        # We want the resulting value *of that root_state's node* after the search.
        # This requires mcts_search to return more than just the action.
        # Let's modify mcts_search to return (best_action, root_value, root_visits)

        best_action_for_s, root_val, root_vis = mcts_search(
            root_state=s,
            env=environment,
            num_iterations=mcts_num_iterations,
            gamma=gamma_discount,
            exploration_constant=mcts_exploration_constant,
            max_rollout_depth=mcts_max_rollout_depth,
            terminal_states=terminal_states_indices,
        )
        mcts_policy[s] = best_action_for_s
        if root_vis > 0:
            mcts_value_function_approx[s] = root_val / root_vis
        else:
            mcts_value_function_approx[s] = (
                0.0  # Should not happen for non-terminal if iterations > 0
            )

    print("MCTS policy and value approximation built.")

    # 4. Print MCTS Policy
    if env_desc_np is not None:
        # Modify the title for the print_policy_grid function or create a new one if needed
        # For now, let's just print a header and reuse.
        print("\n--- MCTS Derived Policy Grid ---")
        print_policy_grid(mcts_policy, env_desc_np, current_grid_shape)
    else:
        print("\n--- MCTS Derived Policy (action indices) ---")
        print(mcts_policy.reshape(current_grid_shape))

    # Add heatmap plotting for MCTS approximate value function
    mcts_heatmap_path = "mcts_approx_value_function_heatmap.png"
    print_value_function_grid(
        mcts_value_function_approx, current_grid_shape
    )  # Print MCTS values too
    if env_desc_np is not None:
        plot_value_function_heatmap(
            mcts_value_function_approx,
            current_grid_shape,
            f"MCTS Approx. Value Function ({mcts_num_iterations} iter/state)",
            env_desc_np,
            save_path=mcts_heatmap_path,
        )
    else:
        plot_value_function_heatmap(
            mcts_value_function_approx,
            current_grid_shape,
            f"MCTS Approx. Value Function ({mcts_num_iterations} iter/state)",
            save_path=mcts_heatmap_path,
        )

    environment.close()

    # --- Evaluation Step ---
    print("\n\n--- Policy Evaluation Step ---")
    num_evaluation_episodes: int = 100  # Number of episodes for evaluation
    max_steps_per_eval_episode: int = 100  # Max steps per episode during evaluation

    # 1. Example Run for DP Policy
    print("\n--- Example Run: DP Optimal Policy ---")
    dp_trajectory, dp_total_reward = run_single_episode(
        environment, opt_policy, max_steps=max_steps_per_eval_episode
    )
    print(f"DP Policy - Total Reward for one episode: {dp_total_reward}")
    print("DP Policy - Trajectory (State, Action, Reward, Next State, Terminated):")
    for i, step_data in enumerate(dp_trajectory):
        print(
            f"  Step {i + 1}: S={step_data[0]}, A={step_data[1]}, R={step_data[2]}, S'={step_data[3]}, Done={step_data[4]}"
        )

    # 2. Example Run for MCTS Policy
    print("\n--- Example Run: MCTS Derived Policy ---")
    mcts_trajectory, mcts_total_reward = run_single_episode(
        environment, mcts_policy, max_steps=max_steps_per_eval_episode
    )
    print(f"MCTS Policy - Total Reward for one episode: {mcts_total_reward}")
    print("MCTS Policy - Trajectory (State, Action, Reward, Next State, Terminated):")
    for i, step_data in enumerate(mcts_trajectory):
        print(
            f"  Step {i + 1}: S={step_data[0]}, A={step_data[1]}, R={step_data[2]}, S'={step_data[3]}, Done={step_data[4]}"
        )

    # 3. Cumulative Reward Evaluation and Plotting
    print(f"\n--- Evaluating Cumulative Rewards over {num_evaluation_episodes} episodes ---")
    # Re-open environment if it was closed, or use a new instance for evaluation
    # For simplicity, let's assume 'environment' is still usable or re-initialize one
    # If env was closed, it must be re-initialized before evaluate_policy_performance
    # For this script structure, environment.close() is at the very end, so it's fine for now.
    # However, good practice might be to pass env_name and re-make it in eval functions
    # or ensure it's open.

    # For FrozenLake, we need to ensure the environment is reset for each policy evaluation run
    # The evaluate_policy_performance and run_single_episode functions handle env.reset() internally.

    dp_episode_rewards = evaluate_policy_performance(
        environment, opt_policy, num_evaluation_episodes, max_steps_per_eval_episode
    )
    mcts_episode_rewards = evaluate_policy_performance(
        environment, mcts_policy, num_evaluation_episodes, max_steps_per_eval_episode
    )

    print(
        f"\nDP Policy - Average reward over {num_evaluation_episodes} episodes: {np.mean(dp_episode_rewards):.2f}"
    )
    print(
        f"MCTS Policy - Average reward over {num_evaluation_episodes} episodes: {np.mean(mcts_episode_rewards):.2f}"
    )

    cumulative_rewards_plot_path = "evaluation_cumulative_rewards.png"
    plot_cumulative_rewards_comparison(
        dp_episode_rewards,
        mcts_episode_rewards,
        num_evaluation_episodes,
        save_path=cumulative_rewards_plot_path,
    )

    # --- Video Recording Step for DP Policy ---
    print("\n\n--- Generating Sample Video for DP Policy ---")
    dp_video_folder = "videos/frozen_lake_dp_policy"
    print(f"Attempting to save DP video to {dp_video_folder}...")
    try:
        dp_video_env = gym.make(env_name, render_mode="rgb_array")
        # dp_video_env.reset() # Initial reset, RecordVideo's reset on wrapped env is often the trigger

        dp_recorded_env = RecordVideo(
            dp_video_env,
            video_folder=dp_video_folder,
            episode_trigger=lambda episode_id: episode_id == 0,
            name_prefix=f"{env_name}-dp-policy",
        )

        print(f"Running one episode with DP policy to record video...")
        current_state, _ = (
            dp_recorded_env.reset()
        )  # This reset triggers video recording for episode 0

        done = False
        dp_video_episode_reward = 0.0
        for _step in range(max_steps_per_eval_episode):
            action = opt_policy[current_state]
            next_state, reward, terminated, truncated, _ = dp_recorded_env.step(action)
            done = terminated or truncated
            dp_video_episode_reward += float(reward)
            current_state = next_state
            if done:
                break

        print(f"Episode for DP video finished. Total reward: {dp_video_episode_reward}")
        dp_recorded_env.close()
        print(f"DP video recording finished. Check the folder: ./{dp_video_folder}")

    except Exception as e:
        print(f"Error during DP video recording: {e}")
        print("Please ensure 'ffmpeg' is installed and accessible in your system's PATH.")
        print(
            "You can typically install it using: sudo apt-get install ffmpeg (Linux) or brew install ffmpeg (macOS)."
        )

    # --- Video Recording Step for MCTS Policy ---
    print("\n\n--- Generating Sample Video for MCTS Policy ---")
    mcts_video_folder = "videos/frozen_lake_mcts_policy"
    print(f"Attempting to save MCTS video to {mcts_video_folder}...")

    try:
        mcts_video_env = gym.make(env_name, render_mode="rgb_array")
        # mcts_video_env.reset() # Initial reset

        mcts_recorded_env = RecordVideo(
            mcts_video_env,
            video_folder=mcts_video_folder,
            episode_trigger=lambda episode_id: episode_id == 0,  # Record only the first episode
            name_prefix=f"{env_name}-mcts-policy",
        )

        print(f"Running one episode with MCTS policy to record video...")
        current_state, _ = mcts_recorded_env.reset()  # This reset triggers video recording

        done = False
        mcts_video_episode_reward = 0.0
        for _step in range(max_steps_per_eval_episode):  # Use existing max_steps
            action = mcts_policy[current_state]  # Use MCTS policy
            next_state, reward, terminated, truncated, _ = mcts_recorded_env.step(action)
            done = terminated or truncated
            mcts_video_episode_reward += float(reward)
            current_state = next_state
            if done:
                break

        print(f"Episode for MCTS video finished. Total reward: {mcts_video_episode_reward}")

        mcts_recorded_env.close()  # Finalizes and saves the MCTS video
        print(f"MCTS video recording finished. Check the folder: ./{mcts_video_folder}")

    except Exception as e:
        print(f"Error during MCTS video recording: {e}")
        print("Please ensure 'ffmpeg' is installed and accessible in your system's PATH.")
        # print("You can typically install it using: sudo apt-get install ffmpeg (Linux) or brew install ffmpeg (macOS).") # Redundant message from DP block

    # Final close of the original environment after all operations
    environment.close()
