import numpy as np
import gym

from memory import Memory
from dq_network import DQNetwork

from utils import stack_frames

env = gym.make("Assault-v0")

# hyperparams
state_size = [110, 84, 4]
action_size = env.action_space.n

learning_rate = 0.0001
gamma = 0.9

total_episodes = 10000
max_steps = 50000
batch_size = 128

explore_start = 1.0
explore_stop = 0.01
# decay_rate = 0.00001
decay_rate = 0.0001

pretrain_length = batch_size
memory_size = 100000

tau = 250
tau_learning_rate = 0.01

memory = Memory(max_size = memory_size)

for i in range(pretrain_length):
    if i == 0:
        state = env.reset()
        stacked_state, stacked_frames = stack_frames(state, is_new_episode = True)

    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)

    next_stacked_state, stacked_frames = stack_frames(next_state, stack = stacked_frames)

    if done:
        memory.add((
            stacked_state,
            action,
            reward,
            next_stacked_state,
            done
        ))

        state = env.reset()
        stacked_state, stacked_frames = stack_frames(state, is_new_episode = True)
        
    else:
        memory.add((
            stacked_state,
            action,
            reward,
            next_stacked_state,
            done
        ))

        stacked_state = next_stacked_state