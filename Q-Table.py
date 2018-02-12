import gym 
from gym import wrappers
import numpy as np
import random
import math

env = gym.make("FrozenLake-v0")

num_actions = env.action_space.n
state = env.observation_space.n
max_t = 250  # Max time

q_table = np.zeros([state, num_actions])

# Learning parameters
min_learning_rate = 0.8
discount_factor = 0.95
min_explore_rate = 0.05
num_episodes = 1000

print (q_table)


# Some helper funcions
def modify_reward(reward, done):
    if done and reward == 0:
        return -100.0
    elif done:
        return 100.0
    else:
        return 1.0

def select_action(state, explore_rate):
	if random.random() < explore_rate:
		action = env.action_space.sample()
	else:
		action = np.argmax(q_table[state])
	return action

# Main program
rList = []

for episode in range(num_episodes):

	obv = env.reset()

	state_0 = obv

	done = False

	rAll = 0

	for  t in range(max_t):
		env.render()
		
		action = select_action(state_0, min_explore_rate)

		obv, reward, done, _ = env.step(action)

		reward_mod = modify_reward(reward,done)

		state = obv

		reward_to_action = reward_mod

		q_table[state_0,action] += min_learning_rate * (reward_mod + discount_factor * np.argmax(q_table[state]))

		rAll += reward

		state_0 = state

		if done:
			break

	rList.append(rAll)
print(q_table)
print ("Score over time: " +  str(sum(rList)/num_episodes))
