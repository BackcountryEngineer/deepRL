import tensorflow as tf
import numpy as np


class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, gamma):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma

        conv1 = tf.keras.layers.Conv2D(
            32, 
            (8,8),
            strides = (4,4),
            activation = "relu",
            input_shape = state_size,
            name = "input_conv"
        )

        conv2 = tf.keras.layers.Conv2D(
            64, 
            (4,4),
            strides = (2,2),
            activation = "relu",
            name = "conv2"
        )

        conv3 = tf.keras.layers.Conv2D(
            64, 
            (3,3),
            strides = (2,2),
            activation = "relu",
            name = "conv3"
        )

        flatten = tf.keras.layers.Flatten()

        dense = tf.keras.layers.Dense(
            512,
            activation = "relu",
            name = "dense1"
        )

        output = tf.keras.layers.Dense(
            action_size,
            name = "output_dense"
        )

        self.network = tf.keras.Sequential(
            [
                conv1,
                conv2,
                conv3,
                flatten,
                dense,
                output
            ]
        )

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def loss(self, q_values, target_q_values):
        return tf.math.reduce_mean(tf.math.squared_difference(target_q_values, q_values))

    def calculate_q_values(self, states, actions):
        action_q_values = self.network(states)
        actions_one_hot = tf.one_hot(actions, self.action_size, 1.0, 0.0)
        return tf.math.reduce_sum(action_q_values * actions_one_hot, axis=-1)

    def calculate_target_q_values(self, rewards, next_states, dones):
        next_q_values = self.network(next_states)
        return rewards + self.gamma * np.max(next_q_values, axis=1) * np.logical_not(dones)

    def next_q_values(self, next_states, dones):
        next_q_values = self.network(next_states)
        return np.max(next_q_values, axis=1) * np.logical_not(dones)

    def mb_update(self, states, actions, rewards, next_states, dones):
        # Minibatch MSE => (1/batch_size) * (R + gamma * Q(s',a') - Q(s,a))^2
        with tf.GradientTape() as tape:
            q_values = self.calculate_q_values(states, actions)
            target_q_values = self.calculate_target_q_values(rewards, next_states, dones)

            loss_value = self.loss(q_values, target_q_values)

            grads = tape.gradient(loss_value, self.network.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))
        
        return loss_value

    def mb_targeted_update(self, states, actions, target_q_values):
        # Minibatch MSE => (1/batch_size) * (R + gamma * Q(s',a') - Target_Q_Values)^2
        with tf.GradientTape() as tape:
            q_values = self.calculate_q_values(states, actions)

            loss_value = self.loss(q_values, target_q_values)

            grads = tape.gradient(loss_value, self.network.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))
        
        return loss_value

    def stochastic_update(self, state, action, reward, next_state, done):
        with tf.GradientTape() as tape:
            action_q_values = self.network(state)
            next_q_values = self.network(next_state)

            target_q = reward + self.gamma * np.max(next_q_values) * np.logical_not(done)

            loss_value = self.loss(action_q_values[0, action], target_q)

            grads = tape.gradient(loss_value, self.network.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))

        return loss_value

    def policy_action(self, state):
        action_q_values = self.network(state.reshape((1, *self.state_size)))

        return np.argmax(action_q_values)

    def save(self, path):
        self.network.save_weights(path)

    def load(self, path):
        self.network.load_weights(path)

    def clone_weights(self, src_net, rl=None):
        self.network.set_weights(src_net.get_weights())
        #if not rl:
        #   self.network.set_weights(src_net.get_weights())
        #else:
        #   self.network.set_weights(np.multiply(rl, self.network.get_weights()) + np.multiply((1 - rl), src_net.get_weights()))

    def get_weights(self):
        return self.network.get_weights()
            
            

            
