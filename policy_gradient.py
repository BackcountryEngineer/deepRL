import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
import gym

from utils import discount_and_normalize_rewards

env = gym.make("CartPole-v1")

env.unwrapped
env.seed(1)

## ENVIRONMENT Hyperparameters
state_size = 4
action_size = env.action_space.n

## TRAINING Hyperparameters
max_episodes = 300
learning_rate = 0.01
gamma = 0.95 # Discount rate

pga = tf.keras.Sequential()
pga.add(tf.keras.layers.Dense(20, input_shape=(state_size,), activation="relu", name="d1"))
pga.add(tf.keras.layers.Dense(12, activation="relu", name="d2"))
pga.add(tf.keras.layers.Dense(action_size, activation="softmax", name="output"))

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

def policy_action(state):
    action_probs = pga(np.array([state]), training=False)
    dist = tfp.distributions.Categorical(probs=action_probs)
    action = dist.sample()
    return int(action.numpy()[0])

def loss(action_probs, actions, discounted_rewards):
    scce = tf.keras.losses.SparseCategoricalCrossentropy()
    neg_log_prob = scce(actions, action_probs)
    return tf.math.reduce_mean(neg_log_prob * discounted_rewards)

def update_policy(states, rewards, actions):
    discounted_rewards = discount_and_normalize_rewards(rewards, gamma)

    with tf.GradientTape() as tape:
        action_probs = pga(np.array(states), training=True)
        loss_val = loss(action_probs, actions, discounted_rewards)
    grads = tape.gradient(loss_val, pga.trainable_variables)
    optimizer.apply_gradients(zip(grads, pga.trainable_variables))
    return loss_val

allRewards = []
total_rewards = 0
maximumRewardRecorded = 0
episode = 0

for episode in range(max_episodes):
    episode_states, episode_actions, episode_rewards = [],[],[]
        
    episode_rewards_sum = 0

    # Launch the game
    state = env.reset()
    
    env.render()
        
    while True:
        action = policy_action(state)

        # Perform a
        new_state, reward, done, _ = env.step(action)

        # Store s, a, r
        episode_states.append(state)
        episode_actions.append(action)
        episode_rewards.append(reward)

        if done:
            # Calculate sum reward
            episode_rewards_sum = np.sum(episode_rewards)
            
            allRewards.append(episode_rewards_sum)
            
            total_rewards = np.sum(allRewards)
            
            # Mean reward
            mean_reward = np.divide(total_rewards, episode+1)
            
            
            maximumRewardRecorded = np.amax(allRewards)
            
            update_policy(episode_states, episode_rewards, episode_actions)

            print("==========================================")
            print("Episode: ", episode)
            print("Reward: ", episode_rewards_sum)
            print("Mean Reward", mean_reward)
            print("Max reward so far: ", maximumRewardRecorded)
            
            break

        state = new_state

