import tensorflow as tf
import tensorflow.keras as keras
from keras.layers import LSTM, Reshape
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from domain import ACTIONS


def create_policy_network(learning_rate, state_dim, layer_size:int, lstm_size: int, time_steps: int, size_vocabulary: int, output_activation: str,
                          number_communication_channels, number_of_big_layers: int, recurrent: bool=False):
    inputs = keras.Input(shape=(time_steps, state_dim))
    x = inputs
    if recurrent:
        x = LSTM(lstm_size, activation=tf.nn.relu)(x)
    else:
        x = Reshape([state_dim])(x)
    for _ in range(number_of_big_layers-recurrent):
        x = Dense(layer_size, activation=tf.nn.relu)(x)
    output_activation_function = tf.nn.softmax if output_activation == "softmax" else (tf.nn.log_softmax if output_activation == "log_softmax" else None)
    action_output = Dense(len(ACTIONS), activation=output_activation_function)(x)
    com_channel_outputs = [Dense(size_vocabulary + 1, activation = output_activation_function)(x) for _ in range(number_communication_channels)]
    model = keras.Model(inputs=inputs, outputs=[action_output] + com_channel_outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate,clipnorm=1.0,clipvalue=0.5))
    return model



def create_critic_network(learning_rate, state_dim, output_dim, layer_size:int, lstm_size: int, time_steps: int, agent_num: int, number_of_big_layers: int, recurrent: bool):
    inputs = keras.Input(shape=(time_steps, state_dim * agent_num))
    x = inputs
    if recurrent:
        x = LSTM(lstm_size)(x)
    else:
        x = Reshape([state_dim*agent_num])(x)
    for _ in range(number_of_big_layers-recurrent):
        x = Dense(layer_size, activation=tf.nn.relu,kernel_regularizer=tf.keras.regularizers.l2(0.01) )(x)
    out = Dense(output_dim, activation=None)(x)
    model = keras.Model(inputs=inputs, outputs=out)
    model.compile(optimizer=Adam(learning_rate=learning_rate,clipnorm=1.0,clipvalue=0.5))
    return model