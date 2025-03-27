import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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
    Plot the Q-values for each action in a heatmap, ensuring that x is on the x-axis 
    and y is on the y-axis by correctly indexing the matrix.
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