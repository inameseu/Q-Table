import gym
import numpy as np
import random

def select_action(state, explore_rate):
    if random.random() < explore_rate:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_table[state])
    return action

'''
def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(1, 1.0 - math.log10((t+1)/25)))

def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((t+1)/25)))
'''

env = gym.make("FrozenLake-v0")

num_states = env.observation_space.n
num_actions = env.action_space.n

q_table = np.zeros([num_states, num_actions])

learning_rate = 0.5
discount_factor = 0.9
explore_rate = 0.05

num_episodes = 2000
max_time = 200

for i in range(num_episodes):
    obv = env.reset()
    state_0 = obv

    for t in range(max_time):
        env.render()
        
        action = select_action(state_0, explore_rate)
        obv, reward, done, _ = env.step(action)


