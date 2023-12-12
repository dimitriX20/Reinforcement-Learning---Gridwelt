import numpy as np
import random
import plotly.graph_objs as go

class GridWorld:
    def __init__(self, width, height, obstacles, special_states):
        self.width = width
        self.height = height
        self.obstacles = obstacles
        self.special_states = special_states
        self.state_space = [(x, y) for x in range(width) for y in range(height) if (x, y) not in obstacles]
        self.action_space = ['up', 'down', 'left', 'right']
        self.q_values = np.zeros((width, height, len(self.action_space)))

    def is_valid_state(self, state):
        return state in self.state_space

    def move(self, state, action):
        x, y = state
        prob = random.uniform(0, 1)
        if prob < 0.1:
            action = self.action_space[(self.action_space.index(action) - 1) % len(self.action_space)]
        elif prob > 0.9:
            action = self.action_space[(self.action_space.index(action) + 1) % len(self.action_space)]

        if action == 'up':
            y -= 1
        elif action == 'down':
            y += 1
        elif action == 'left':
            x -= 1
        elif action == 'right':
            x += 1

        new_state = (x, y)
        return new_state if self.is_valid_state(new_state) else state

    def get_reward(self, state):
        return self.special_states.get(state, -0.04)
    
    # Konvertiert den Zustand in eine numerische Repräsentation für das neuronale Netz
    def get_state_representation(self, state):
        state_representation = np.zeros((self.width * self.height,))
        state_representation[state[0] * self.height + state[1]] = 1
        return state_representation

class QLearningAgent:
    def __init__(self, grid_world, learning_rate, discount_factor, epsilon):
        self.grid_world = grid_world
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.total_rewards = []
        self.total_steps = []

    def run_episode(self):
        state = random.choice(self.grid_world.state_space)
        total_reward = 0
        steps = 0
        done = False

        while not done:
            if random.uniform(0, 1) < self.epsilon:
                action = random.choice(self.grid_world.action_space)
            else:
                action = self.grid_world.action_space[np.argmax(self.grid_world.q_values[state])]

            next_state = self.grid_world.move(state, action)
            reward = self.grid_world.get_reward(next_state)
            done = next_state in self.grid_world.special_states or next_state in self.grid_world.obstacles

            total_reward += reward
            steps += 1

            # Q-Learning Update
            self.update_q_value(state, action, reward, next_state)

            state = next_state

        self.total_rewards.append(total_reward)
        self.total_steps.append(steps)

    def update_q_value(self, state, action, reward, next_state):
        old_value = self.grid_world.q_values[state + (self.grid_world.action_space.index(action),)]
        next_max = np.max(self.grid_world.q_values[next_state])
        new_value = old_value + self.learning_rate * (reward + self.discount_factor * next_max - old_value)
        self.grid_world.q_values[state + (self.grid_world.action_space.index(action),)] = new_value

    def train(self, episodes):
        for _ in range(episodes):
            self.run_episode()

    def derive_policy(self):
        policy = {}
        for state in self.grid_world.state_space:
            policy[state] = self.grid_world.action_space[np.argmax(self.grid_world.q_values[state])]
        return policy