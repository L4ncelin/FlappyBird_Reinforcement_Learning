import numpy as np

class MonteCarloAgent:
    def __init__(self, env, gamma=0.9, epsilon=0.1, alpha=0.1):
        self.env = env
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration (ε-greedy)
        self.alpha = alpha  # Learning rate
        self.Q = {}  # Q-values for state-action pairs
        self.policy = {}  # Optimized policy

    def select_action(self, state):
        """ Selects an action using ε-greedy policy """
        if np.random.rand() < self.epsilon:  # Exploration
            return self.env.action_space.sample()

        # Exploitation: pick the action with the highest Q-value for this state
        if state not in self.policy:  # If state not seen before, pick a random action
            return self.env.action_space.sample()
        return max(self.policy[state], key=self.policy[state].get)

    def update_policy(self, trajectory):
        """ Update Q-values and policy after an episode using Monte Carlo """
        G = 0
        visited = set()  # To avoid updating the same state-action pair more than once

        for t in reversed(range(len(trajectory))):
            state, action, reward = trajectory[t]
            G = self.gamma * G + reward  # Compute the return G_t

            if (state, action) not in visited:
                visited.add((state, action))

                # Update Q-value using the simple Monte Carlo rule
                if (state, action) not in self.Q:
                    self.Q[(state, action)] = 0  # Initialize if not present

                self.Q[(state, action)] += self.alpha * (G - self.Q[(state, action)])

                # Update the policy for this state
                if state not in self.policy:
                    self.policy[state] = {a: 0 for a in range(self.env.action_space.n)}

                # After updating Q-values, choose the best action based on the Q-values
                best_action = max(range(self.env.action_space.n), key=lambda a: self.Q.get((state, a), 0))
                for a in range(self.env.action_space.n):
                      # Epsilon greedy policy update
                      self.policy[state][a] = (
                      (1 - self.epsilon + self.epsilon / self.env.action_space.n) if a == best_action
                      else (self.epsilon / self.env.action_space.n))