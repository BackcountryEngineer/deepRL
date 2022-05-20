from math import gamma
import tensorflow as tf
import numpy as np


class DDDQNetwork:
    def __init__(self, state_size, action_size, learning_rate, gamma):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma

        inputs = tf.keras.Input(shape=(*state_size,))

        conv1 = tf.keras.layers.Conv2D(
            32, 
            (8,8),
            strides = (4,4),
            activation = "relu",
            input_shape = state_size,
            name = "input_conv"
        )(inputs)

        conv2 = tf.keras.layers.Conv2D(
            64, 
            (4,4),
            strides = (2,2),
            activation = "relu",
            name = "conv2"
        )(conv1)

        self.conv3 = tf.keras.layers.Conv2D(
            64, 
            (3,3),
            strides = (2,2),
            activation = "relu",
            name = "conv3"
        )(conv2)

        flatten = tf.keras.layers.Flatten()(self.conv3)

        self.value_dense = tf.keras.layers.Dense(
            512,
            activation = "relu",
            name = "value_dense"
        )(flatten)

        self.value_ouputs = tf.keras.layers.Dense(
            action_size,
            activation = "relu",
            name = "value_output"
        )(self.value_dense)

        self.value_network = tf.keras.Model(inputs=inputs, outputs=self.value_ouputs)

        self.advantage_dense = tf.keras.layers.Dense(
            512,
            activation = "relu",
            name = "advantage_dense"
        )(flatten)

        self.advantage_outputs = tf.keras.layers.Dense(
            action_size,
            activation = "relu",
            name = "advantage_output"
        )(self.advantage_dense)

        self.advantage_network = tf.keras.Model(inputs=inputs, outputs=self.advantage_outputs)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def action_q_values(self, states):
        values = self.value_network(states)
        advantages = self.advantage_network(states)
        return values + (advantages - tf.math.reduce_mean(advantages, axis=1, keepdims=True))

    def q_values(self, states, actions):
        action_q_values = self.action_q_values(states)
        actions_one_hot = tf.one_hot(actions, self.action_size, 1.0, 0.0)

        return tf.math.reduce_sum(tf.math.multiply(action_q_values, actions_one_hot), axis=-1)

    def next_q_values(self, rewards, next_states, dones):
        values = self.value_network(next_states)
        advantages = self.advantage_network(next_states)
        action_q_values = values + (advantages - tf.math.reduce_mean(advantages, axis=1, keepdims=True))

        return rewards + self.gamma * np.max(action_q_values, axis=1) * np.logical_not(dones)

    def loss(self, q_values, target_q_values):
        # Minibatch MSE => (1/batch_size) * (R + gamma * Q(s',a') - Target_Q_Values)^2
        return tf.math.reduce_mean(tf.math.squared_difference(target_q_values, q_values))

    def mb_update(self, states, actions, rewards, next_states, dones):
        with tf.GradientTape() as tape:
            qvals = self.q_values(states, actions)

            with tape.stop_recording():
                next_qvals = self.next_q_values(rewards, next_states, dones)

            loss_val = self.loss(qvals, next_qvals)

        all_trainable_weights = self.advantage_network.trainable_variables + self.value_network.trainable_variables
        
        grads = tape.gradient(loss_val, all_trainable_weights)
        self.optimizer.apply_gradients(zip(grads, all_trainable_weights))

        return loss_val

    def mb_targeted_update(self, states, actions, target_q_values):
        with tf.GradientTape() as tape:
            q_values = self.q_values(states, actions)

            loss_val = self.loss(q_values, target_q_values)

        all_trainable_weights = self.advantage_network.trainable_variables + self.value_network.trainable_variables
        
        grads = tape.gradient(loss_val, all_trainable_weights)
        self.optimizer.apply_gradients(zip(grads, all_trainable_weights))

        return loss_val

    def policy_action(self, state):
        action_q_values = self.action_q_values(state.reshape((1, *self.state_size)))

        return np.argmax(action_q_values)

    def save(self, path):
        self.value_network.save_weights(path + "/value")
        self.advantage_network.save_weights(path + "/advatage")

    def load(self, path):
        self.value_network.load_weights(path + "/value")
        self.advantage_network.load_weights(path + "/advatage")

    def clone_weights(self, src_net):
        value_weights, advantage_weights = src_net.get_weights()
        self.value_network.set_weights(value_weights)
        self.advantage_network.set_weights(advantage_weights)

    def get_weights(self):
        return self.value_network.get_weights(), self.advantage_network.get_weights()
