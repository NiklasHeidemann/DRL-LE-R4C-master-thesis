import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Concatenate


def create_policy_network(learning_rate, state_dim, action_dim, number_of_big_layers: int = 3):
    inputs = keras.Input(shape=state_dim)
    x = inputs
    for _ in range(number_of_big_layers):
        x = Dense(256, activation=tf.nn.relu)(x)
    mu = Dense(action_dim, activation=None)(x)
    sigma = Dense(action_dim, activation=tf.nn.softplus)(x)
    model = keras.Model(inputs=inputs, outputs=(mu, sigma))
    model.compile(optimizer=Adam(learning_rate=learning_rate))
    return model


def create_q_network(learning_rate, state_dim, action_dim, number_of_big_layers: int = 3):
    inputs_s = keras.Input(shape=state_dim)
    inputs_a = keras.Input(shape=action_dim)
    x = Concatenate()([inputs_s, inputs_a])
    for _ in range(number_of_big_layers):
        x = Dense(256, activation=tf.nn.relu)(x)
    out = Dense(1, activation=None)(x)
    model = keras.Model(inputs=(inputs_s, inputs_a), outputs=out)
    model.compile(optimizer=Adam(learning_rate=learning_rate))
    return model
