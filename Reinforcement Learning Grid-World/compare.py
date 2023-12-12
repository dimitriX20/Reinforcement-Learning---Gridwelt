from grid import GridWorld
from grid import QLearningAgent
import plotly.graph_objs as go
from policygradient import PolicyGradient
import random

def display_policy_matrix(policy, grid_world, width, height):
    action_symbols = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→', 'special': '*', 'obstacle': '#'}
    matrix = [[' ' for _ in range(width)] for _ in range(height)]

    for y in range(height):
        for x in range(width):
            state = (x, y)
            if state in grid_world.special_states:
                matrix[y][x] = action_symbols['special']
            elif state in grid_world.obstacles:
                matrix[y][x] = action_symbols['obstacle']
            else:
                action = policy.get(state)
                matrix[y][x] = action_symbols.get(action, ' ')

    return matrix

# initialisiere grid world und trainiere den Agenten
obstacles = [(1, 1)]
special_states = {(3, 0): 1, (3, 1): -1}
grid_world = GridWorld(4, 3, obstacles, special_states)

# Trainieren des Policy Gradient Agenten
def train_policy_gradient_agent(agent, grid_world, episodes):
    total_rewards = [] 
    for episode in range(episodes):
        state = random.choice(grid_world.state_space)
        state_representation = grid_world.get_state_representation(state)
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state_representation)
            next_state = grid_world.move(state, grid_world.action_space[action])
            reward = grid_world.get_reward(next_state)
            done = next_state in grid_world.special_states or next_state in grid_world.obstacles
            next_state_representation = grid_world.get_state_representation(next_state)

            agent.store_transition(state_representation, action, reward)
            state = next_state
            state_representation = next_state_representation
            total_reward += reward

        agent.learn()
        total_rewards.append(total_reward)
        print("Episode:", episode, "Total Reward:", total_reward)
    return total_rewards


N = 100

agent = QLearningAgent(grid_world, learning_rate=0.1, discount_factor=0.9, epsilon=0.1)
agent.train(episodes=N)
q_learning_rewards = agent.total_rewards

pg_agent = PolicyGradient(grid_world.width * grid_world.height, len(grid_world.action_space))
policy_gradient_rewards = train_policy_gradient_agent(pg_agent, grid_world, N)

# vergleiche policy gradient mit q-learning 
trace1 = go.Scatter(y=q_learning_rewards, mode='lines', name='Q-Learning Rewards')
trace2 = go.Scatter(y=policy_gradient_rewards, mode='lines', name='Policy Gradient Rewards')
layout = go.Layout(title='Reward Comparison', xaxis={'title': 'Episode'}, yaxis={'title': 'Total Reward'})
fig = go.Figure(data=[trace1, trace2], layout=layout)
fig.show()

# Ableiten und Ausgeben der Policy vom q-learning
policy = agent.derive_policy()
policy_matrix = display_policy_matrix(policy, grid_world, grid_world.width, grid_world.height)

for row in policy_matrix:
    print(' '.join(row))

#print("Durchschnittliche Belohnung:", sum(total_rewards) / episodes)
#print("Durchschnittliche Schritte:", sum(total_steps) / episodes)