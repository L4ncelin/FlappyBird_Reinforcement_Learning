import numpy as np
import random

class SarsaLambdaAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.99, lambd=0.9, epsilon=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.lambd = lambd
        self.epsilon = epsilon
        self.w = np.zeros((*state_size, action_size))  # Linear function approximation weights
        self.z = np.zeros((*state_size, action_size))  # Eligibility traces
    
    def select_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.action_size))
        else:
            # state_idx = (state[0], state[1] + 11) 
            return np.argmax(self.w[state])
    
    def update_policy(self, state, action, reward, next_state, next_action, done):
        # state_idx = (state[0], state[1] + 11)
        # next_state_idx= (next_state[0], next_state[1] + 11)

        delta = reward + (self.gamma * self.w[next_state, next_action] * (1 - done)) - self.w[state, action]
        self.z[state, action] += 1  # Accumulating traces
        
        self.w += self.alpha * delta * self.z
        self.z *= self.gamma * self.lambd  # Decay traces
        
        if done:
            self.z.fill(0)  # Reset traces at episode end