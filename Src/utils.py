import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
from Agents.SarsaLambdaAgent import SarsaLambdaAgent
from Agents.MonteCarloAgent import MonteCarloAgent
import gymnasium as gym


def plot_rewards(rewards):
    """
    Displays the evolution of the cumulative reward per episode.

    Args:
        rewards (list): List of cumulative rewards for each episode.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Reward per episode", color='b', alpha=0.7)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Reward Evolution During Training")
    plt.legend()
    plt.grid()
    plt.show()

def moving_average(data, window_size=50):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def plot_q_values(Q):
    """ 
    Plot the Q-values for each action in a heatmap.
    """

    # Extract the x, y values and actions
    x_vals = [state[0] for state, action in Q.keys()]  # x distance to the pipe
    y_vals = [state[1] for state, action in Q.keys()]  # y distance to the pipe
    actions = [action for state, action in Q.keys()]  # actions (0 = no flap, 1 = flap)

    # Create a grid of unique x and y values
    x_unique = sorted(set(x_vals), reverse=True)
    y_unique = sorted(set(y_vals))

    # Create a matrix for storing Q-values (rows correspond to y, columns to x)
    q_matrix_flap = np.full((len(y_unique), len(x_unique)), np.nan)  # 'flap' action (action = 1)
    q_matrix_no_flap = np.full((len(y_unique), len(x_unique)), np.nan)  # 'no flap' action (action = 0)

    # Map the states to their corresponding Q-values
    for (state, action), q_value in Q.items():
        x_idx = x_unique.index(state[0])  # Correct indexing in sorted unique list
        y_idx = y_unique.index(state[1])

        if action == 0:
            q_matrix_no_flap[y_idx, x_idx] = q_value  # y first, then x
        elif action == 1:
            q_matrix_flap[y_idx, x_idx] = q_value  # y first, then x

    # Plot the heatmaps for both actions
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Heatmap for 'No Flap' action (action = 0)
    sns.heatmap(q_matrix_no_flap, ax=axes[0], cmap="coolwarm", cbar_kws={'label': 'Q-value'},
                xticklabels=x_unique, yticklabels=y_unique)
    axes[0].set_title('Q-values for No Flap (Action = 0)')
    axes[0].set_xlabel('Horizontal Distance to Pipe (x)')
    axes[0].set_ylabel('Vertical Distance to Pipe (y)')

    # Heatmap for 'Flap' action (action = 1)
    sns.heatmap(q_matrix_flap, ax=axes[1], cmap="coolwarm", cbar_kws={'label': 'Q-value'},
                xticklabels=x_unique, yticklabels=y_unique)
    axes[1].set_title('Q-values for Flap (Action = 1)')
    axes[1].set_xlabel('Horizontal Distance to Pipe (x)')
    axes[1].set_ylabel('Vertical Distance to Pipe (y)')

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_w_values(W):
    """
    Plot the W-values for each action in a heatmap, ensuring that x is on the x-axis 
    and y is on the y-axis.
    
    Parameters:
    - W: numpy array of shape (14,22,2) containing W-values
    """

    # Extract the W-values for each action
    w_matrix_no_flap = W[:, :, 0]  # Action 0 (No Flap)
    w_matrix_flap = W[:, :, 1]  # Action 1 (Flap)

    # Operations 
    w_matrix_no_flap = np.flip(w_matrix_no_flap.T, axis=1)
    w_matrix_no_flap = np.vstack((w_matrix_no_flap[11:], w_matrix_no_flap[:11]))
    
    w_matrix_flap = np.flip(w_matrix_flap.T, axis=1)
    w_matrix_flap = np.vstack((w_matrix_flap[11:], w_matrix_flap[:11]))

    # Define x and y labels
    x_labels = np.arange(W.shape[0])[::-1]  # 14 unique y-values (vertical distance)
    y_labels = np.arange(-11,11)  # 22 unique x-values (horizontal distance)


    # Plot the heatmaps for both actions
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Heatmap for 'No Flap' action (action = 0)
    sns.heatmap(w_matrix_no_flap, ax=axes[0], cmap="coolwarm", cbar_kws={'label': 'W-value'},
               xticklabels=x_labels, yticklabels=y_labels)
    axes[0].set_title('W-values for No Flap (Action = 0)')
    axes[0].set_xlabel('Horizontal Distance to Pipe (x)')
    axes[0].set_ylabel('Vertical Distance to Pipe (y)')

    # Heatmap for 'Flap' action (action = 1)
    sns.heatmap(w_matrix_flap, ax=axes[1], cmap="coolwarm", cbar_kws={'label': 'W-value'},
                xticklabels=x_labels, yticklabels=y_labels)
    axes[1].set_title('W-values for Flap (Action = 1)')
    axes[1].set_xlabel('Horizontal Distance to Pipe (x)')
    axes[1].set_ylabel('Vertical Distance to Pipe (y)')

    # Show the plot
    plt.tight_layout()
    plt.show()


def run_sarsa_agent(num_episodes):
    """ Runs a single instance of the agent and returns cumulative rewards per episode. """
    env = gym.make('TextFlappyBird-v0', height=15, width=20, pipe_gap=4)

    # Get the size of the action space
    action_size = env.action_space.n
    state_shape = (14, 22)  # Tuple observation space: Discrete(14), Discrete(22)

    # Initialize the Sarsa(λ) agent with the necessary parameters
    agent = SarsaLambdaAgent(state_size=state_shape, action_size=action_size, alpha=0.2, gamma=0.96, lambd=0.9, epsilon=1.0)

    reward_history = []

    epsilon_min = 0.01  # Minimum exploration
    epsilon_decay = 0.995  # Gradual reduction

    for episode in range(num_episodes):
        obs, _ = env.reset()
        trajectory = []  # Stores (state, action, reward)
        done = False
        total_reward = 0

        # Initialize the action using the selection method
        action = agent.select_action(obs)

        while not done:
            next_obs, reward, done, _, info = env.step(action)
            next_action = agent.select_action(next_obs)

            # Add the transition to the trajectory
            trajectory.append((obs, action, reward))

            # Update the agent using the transition
            agent.update_policy(obs, action, reward, next_obs, next_action, done)

            obs = next_obs
            action = next_action
            total_reward += reward

        reward_history.append(total_reward)

        # Gradually reduce exploration
        agent.epsilon = max(epsilon_min, agent.epsilon * epsilon_decay)

    smoothed_rewards = moving_average(reward_history, window_size=50)

    return(smoothed_rewards)

def run_monte_carlo_agent(num_episodes):
    """ Runs a single instance of the agent and returns cumulative rewards per episode. """
    env = gym.make('TextFlappyBird-v0', height=15, width=20, pipe_gap=4)

    # Initialize the Monte Carlo agent with the necessary parameters
    agent = MonteCarloAgent(env, epsilon=1.0, gamma=0.96, alpha=0.2)
    reward_history = []

    epsilon_min = 0.01  # Minimum exploration
    epsilon_decay = 0.995  # Gradual reduction

    for episode in range(num_episodes):
        obs, _ = env.reset()
        trajectory = []  # Stores (state, action, reward)
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(obs)  # Uses the agent's policy
            next_obs, reward, done, _, info = env.step(action)
            trajectory.append((obs, action, reward))  # Store the transition
            obs = next_obs
            total_reward += reward

        reward_history.append(total_reward)

        agent.epsilon = max(epsilon_min, agent.epsilon * epsilon_decay)
        agent.update_policy(trajectory)  # Monte Carlo update after the episode

    smoothed_rewards = moving_average(reward_history, window_size=50)

    return(smoothed_rewards)


def plot_confidence_interval(num_episodes = 2000, num_runs = 10, agent_type:str=""):

    if agent_type == "sarsa":
        # Store all reward curves
        reward_curves = np.array([run_sarsa_agent(num_episodes) for _ in tqdm(range(num_runs))])
    elif agent_type == "monte_carlo":
        # Store all reward curves
        reward_curves = np.array([run_monte_carlo_agent(num_episodes) for _ in tqdm(range(num_runs))])

    # Compute mean and confidence interval (95%)
    mean_rewards = np.mean(reward_curves, axis=0)
    std_rewards = np.std(reward_curves, axis=0)
    confidence = 1.96 * std_rewards / np.sqrt(num_runs)  # 95% confidence interval

    # Plot results
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=np.arange(num_episodes-49), y=mean_rewards, label="Average Cumulative Reward")
    plt.fill_between(np.arange(num_episodes-49), mean_rewards - confidence, mean_rewards + confidence,
                    color="blue", alpha=0.2, label="95% Confidence Interval")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    plt.title("Average Learning Curve with Confidence Interval")
    plt.legend()
    plt.show()

