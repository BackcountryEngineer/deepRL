import tensorflow as tf
import numpy as np


class DDDQNetwork:
    def __init__(self, state_size, action_size, learning_rate, gamma):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma

        inputs = tf.keras.Input(shape=(state_size,))

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

        conv3 = tf.keras.layers.Conv2D(
            64, 
            (3,3),
            strides = (2,2),
            activation = "relu",
            name = "conv3"
        )(conv2)

        flatten = tf.keras.layers.Flatten()(conv3)

        value_dense = tf.keras.layers.Dense(
            512,
            activation = "relu",
            name = "value_dense"
        )(flatten)

        self.value_network = tf.keras.layers.Dense(
            action_size,
            activation = "relu",
            name = "value_output"
        )(value_dense)

        advantage_dense = tf.keras.layers.Dense(
            512,
            activation = "relu",
            name = "advantage_dense"
        )(flatten)

        self.advantage_network = tf.keras.layers.Dense(
            action_size,
            activation = "relu",
            name = "advantage_output"
        )(advantage_dense)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def q_values(self, states, actions):
        values = self.value_network(states)
        advantages = self.advantage_network(states)
        action_q_values = values + (advantages - tf.math.reduce_mean(advantages))
        actions_one_hot = tf.one_hot(actions, self.action_size, 1.0, 0.0)

        return tf.math.reduce_sum(tf.math.multiply(action_q_values, actions_one_hot), axis=-1)

    def next_q_values(self, next_states, dones)

    def loss(self, q_values, target_q_values):
        return tf.math.reduce_mean(tf.math.squared_difference(target_q_values, q_values))

    def mb_update(self, states, actions, target_q_values):
        qvals = self.q_values(states)
