import tensorflow as tf
import tensorflow.keras as keras
from keras.layers import LSTM, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Concatenate

from params import LAYER_SIZE, TIME_STEPS, LSTM_SIZE, NUMBER_COMMUNICATION_CHANNELS, SIZE_VOCABULARY, ACTIONS


def create_policy_network(learning_rate, state_dim, action_dim, number_of_big_layers: int = 3, recurrent: bool=False):
    inputs = keras.Input(shape=(TIME_STEPS, state_dim))
    x = inputs
    if recurrent:
        x = LSTM(LSTM_SIZE, activation=tf.nn.relu)(x)
    else:
        x = Reshape([state_dim])(x)
    for _ in range(number_of_big_layers-recurrent):
        x = Dense(LAYER_SIZE, activation=tf.nn.relu)(x)
    #mu = Dense(action_dim, activation=None)(x)
    #sigma = Dense(action_dim, activation=tf.nn.softplus)(x)
    action_output = Dense(len(ACTIONS), activation=tf.nn.softmax)(x)
    com_channel_outputs = [Dense(SIZE_VOCABULARY + 1, activation = tf.nn.softmax)(x) for _ in range(NUMBER_COMMUNICATION_CHANNELS)]
    model = keras.Model(inputs=inputs, outputs=[action_output] + com_channel_outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate))
    return model


def create_q_network_old(learning_rate, state_dim, action_dim, agent_num: int, number_of_big_layers: int = 3, recurrent: bool=False):
    inputs_s = keras.Input(shape=(TIME_STEPS, state_dim * agent_num))
    inputs_a = keras.Input(shape= action_dim*agent_num)
    x = inputs_s
    if recurrent:
        x = LSTM(LSTM_SIZE)(x) # QUESTION lstm initialisation?
    else:
        x = Reshape([state_dim*agent_num])(x)
    x = Concatenate()([x, inputs_a])
    for _ in range(number_of_big_layers-recurrent):
        x = Dense(LAYER_SIZE, activation=tf.nn.relu)(x)
    out = Dense(agent_num, activation=None)(x)
    model = keras.Model(inputs=(inputs_s, inputs_a), outputs=out)
    model.compile(optimizer=Adam(learning_rate=learning_rate))
    return model

def create_q_network(learning_rate, state_dim, action_dim, agent_num: int, number_of_big_layers: int = 3, recurrent: bool=False):
    inputs = keras.Input(shape=(TIME_STEPS, state_dim * agent_num))
    x = inputs
    if recurrent:
        x = LSTM(LSTM_SIZE)(x) # QUESTION lstm initialisation?
    else:
        x = Reshape([state_dim*agent_num])(x)
    for _ in range(number_of_big_layers-recurrent):
        x = Dense(LAYER_SIZE, activation=tf.nn.relu)(x)
    out = Dense(agent_num*action_dim, activation=None)(x)
    model = keras.Model(inputs=inputs, outputs=out)
    model.compile(optimizer=Adam(learning_rate=learning_rate))
    return model


# todo parallelisierung
#todo lstm
# todo arne