import numpy as np
import gym

from memory import Memory
from dq_network import DQNetwork

from utils import stack_frames

# environment initialization
env = gym.make("Assault-v0")

# hyperparams
state_size = [110, 84, 4]
action_size = env.action_space.n

learning_rate = 0.00025
gamma = 0.9

total_episodes = 100
max_steps = 50000
batch_size = 64

explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.0001

pretrain_length = batch_size
memory_size = 1000000

tau = 100

## experience replay
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

action_network = DQNetwork(state_size, action_size, learning_rate, gamma)
target_network = DQNetwork(state_size, action_size, learning_rate, gamma)

def predict_action(explore_start, explore_stop, decay_rate, decay_step, state):
    exp_exp_tradeoff = np.random.rand()

    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

    if (explore_probability > exp_exp_tradeoff):
        action = env.action_space.sample()
    else:
        # Get action from Q-network (exploitation)
        action = action_network.policy_action(state)

    return action, explore_probability

## agent training
decay_step = 0

for episode in range(total_episodes):
    step = 0
    episode_rewards = []
    loss = 0

    state = env.reset()

    stacked_state, stacked_frames = stack_frames(state, is_new_episode=True)

    while step < max_steps:
        step += 1
        decay_step += 1

        action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, stacked_state)

        next_state, reward, done, _ = env.step(action)

        episode_rewards.append(reward)

        if done:
            next_state = np.zeros(state.shape)
            next_stacked_state, stacked_frames = stack_frames(next_state, stack=stacked_frames)

            memory.add((
                stacked_state,
                action,
                reward,
                next_stacked_state,
                done
            ))
            
            break
        else:
            next_stacked_state, stacked_frames = stack_frames(next_state, stack=stacked_frames)

            memory.add((
                stacked_state,
                action,
                reward,
                next_stacked_state,
                done
            ))

            stacked_state = next_stacked_state
        
        batch = memory.sample(batch_size)
        states_mb = np.array([each[0] for each in batch], ndmin=3)
        actions_mb = np.array([each[1] for each in batch])
        rewards_mb = np.array([each[2] for each in batch]) 
        next_states_mb = np.array([each[3] for each in batch], ndmin=3)
        dones_mb = np.array([each[4] for each in batch])

        if step % tau == 0:
            target_network.clone_weights(action_network)

        fix_target_q_values = target_network.calculate_target_q_values(rewards_mb, next_states_mb, dones_mb)

        loss = action_network.mb_targeted_update(states_mb, actions_mb, fix_target_q_values)

    print(f"Episode: {episode} Total reward: {np.sum(episode_rewards)} Explore P: {explore_probability:.4f} Training Loss {loss:.4f}")

#agent evaluation
for episode in range(2):
    total_rewards = 0
    state = env.reset()
    state, stacked_frames = stack_frames(stacked_frames, state, True)

    while True:
        action = action_network.policy_action(state)
        next_state, reward, done, _ = env.step(action)
        total_rewards += reward
        if done:
            print("Score", total_rewards)
            break
        next_state, stack_frames = stack_frames(stacked_frames, next_state, False)
        state = next_state

env.close()