from collections import defaultdict
import datetime
from collections import defaultdict
from threading import Thread
from typing import List, Tuple, Dict

import numpy as np
import tensorflow as tf
from tensorflow import math as tfm
from tensorflow_probability import distributions as tfd

from SAC.ExperienceReplayBuffer import RecurrentExperienceReplayBuffer
from environment.render import render_permanently
from loss_logger import LossLogger, CRITIC_LOSS, ACTOR_LOSS, LOG_PROBS, RETURNS, Q_VALUES, OTHER_ACTOR_LOSS, \
    MAX_Q_VALUES
from params import BATCH_SIZE, LEARNING_RATE, TIME_STEPS, TRAININGS_PER_TRAINING, ALPHA, ACTIONS, GAMMA
from plots import plot_multiple


class SACAgent:
    """
    Soft Actor-Critic (SAC) Agent
    Based on pseudocode of OpenAI Spinning Up (2022) (https://spinningup.openai.com/en/latest/algorithms/sac.html)
    and the Paper of Haarnoja et al. (2018) (https://arxiv.org/abs/1801.01290)

    :param environment: The environment to learn from (conforming to the gymnasium specifics https://gymnasium.farama.org/)
    :param state_dim: Dimensions of the state space
    :param action_dim: Dimensions of the action space
    :param actor_network_generator: a generator function for the actor network
        signature: learning_rate, state_dim, action_dim -> tensorflow Model
    :param critic_network_generator: a generator function for the critic networks (Q-networks)
        signature: learning_rate, state_dim, action_dim -> tensorflow Model
    :param learning_rate=0.0003: Learning rate for adam optimizer. # todo testing by kl-divergence
        The same learning rate will be used for all networks (Q-Values, Actor)
    :param gamma=0.99: discount factor
    :param tau=0.005:  Polyak averaging coefficient (between 0 and 1)
    :param alpha=0.2: Entropy regularization coefficient (between 0 and 1).
        Controlling the exploration/exploitation trade off.
        (inverse of reward scale in the original SAC paper)
    :param batch_size=256: Minibatch size for each gradient update
    :param max_replay_buffer_size=1000000: Size of the replay buffer
    :param model_path: path to the location the model is saved and loaded from
    """
# todo cnn
    def __init__(self, environment, self_play: bool, agent_ids: List[str], state_dim, action_dim,  from_save: bool,
                 actor_network_generator, critic_network_generator, max_replay_buffer_size:int, recurrent: bool,
                 learning_rate=LEARNING_RATE, gamma=GAMMA, tau=0.005, reward_scale=1, alpha=ALPHA,
                 batch_size:int=BATCH_SIZE,  model_path="model/"):
        self._environment = environment
        self._self_play = self_play
        self._action_dim = action_dim
        self._gamma = gamma
        self._tau = tau
        self._reward_scale = reward_scale
        self._alpha = alpha
        self._batch_size = batch_size
        self._mse = tf.keras.losses.MeanSquaredError()
        self._model_path = model_path
        self._reply_buffer = RecurrentExperienceReplayBuffer(state_dim, action_dim, agent_number=len(agent_ids), max_size=max_replay_buffer_size,batch_size= batch_size)
        self._agent_ids = agent_ids
        if self_play:
            self._actor = actor_network_generator(learning_rate, recurrent=recurrent)
        else:
            self._actors = {agent_id: actor_network_generator(learning_rate, recurrent = recurrent) for agent_id in agent_ids}
        self._critic_1 = critic_network_generator(learning_rate=learning_rate, agent_num=len(agent_ids), recurrent = recurrent)
        self._critic_2 = critic_network_generator(learning_rate=learning_rate, agent_num=len(agent_ids), recurrent = recurrent)
        self._critic_1_t = critic_network_generator(learning_rate=learning_rate, agent_num=len(agent_ids), recurrent = recurrent)
        self._critic_2_t = critic_network_generator(learning_rate=learning_rate, agent_num=len(agent_ids), recurrent = recurrent)
        if from_save:
            self.load_models(name="")
        else:
            self._wight_init()

    def _get_max_q_value(self,states):
        reshaped_states = tf.reshape(np.array(list(states.values())),shape=(1,TIME_STEPS,self._environment.stats.observation_dimension*len(self._agent_ids)))
        q = (tfm.minimum(self._critic_1(reshaped_states), self._critic_2(reshaped_states)))
        q_by_agent = tf.reshape(q, shape=(len(self._agent_ids), len(ACTIONS)))
        max_q_values = np.max(q_by_agent, axis=1)
        max_q_actions = np.argmax(q_by_agent, axis=1)
        return {agent_id: (max_q_value, max_q_action) for agent_id, max_q_value, max_q_action in zip(states.keys(), max_q_values, max_q_actions)}
    def save_models(self, name):
        if self._self_play:
            self._actor.save_weights(f"{self._model_path}actor{name}")
        else:
            for id, actor in self._actors.items():
                actor.save_weights(f"{self._model_path}actor_{id}_{name}")
        self._critic_1.save_weights(f"{self._model_path}critic_1{name}")
        self._critic_2.save_weights(f"{self._model_path}critic_2{name}")
        self._critic_1_t.save_weights(f"{self._model_path}critic_1_t{name}")
        self._critic_2_t.save_weights(f"{self._model_path}critic_2_t{name}")

    def load_models(self, name):
        if self._self_play:
            self._actor.load_weights(f"{self._model_path}actor{name}")
        else:
            for id, actor in self._actors.items():
                actor.load_weights(f"{self._model_path}actor_{id}_{name}")
        self._critic_1.load_weights(f"{self._model_path}critic_1{name}")
        self._critic_2.load_weights(f"{self._model_path}critic_2{name}")
        self._critic_1_t.load_weights(f"{self._model_path}critic_1_t{name}")
        self._critic_2_t.load_weights(f"{self._model_path}critic_2_t{name}")

    def _wight_init(self):
        self._critic_1.set_weights(self._critic_1_t.weights)
        self._critic_2.set_weights(self._critic_2_t.weights)

    def update_target_weights(self):
        self._weight_update(self._critic_1_t, self._critic_1)
        self._weight_update(self._critic_2_t, self._critic_2)

    def _weight_update(self, target_network, network):
        new_wights = []
        for w_t, w in zip(target_network.weights, network.weights):
            new_wights.append((1 - self._tau) * w_t + self._tau * w)
        target_network.set_weights(new_wights)
    def train(self, epochs, environment_steps_before_training=1, training_steps_per_update=1,
              max_environment_steps_per_epoch=None, pre_sampling_steps=1024, save_models=False):
        """
        trains the SAC Agent
        :param epochs: Number of epochs to train.
            One epoch is finished if the agents are done ore the maximum steps are reached
        :param environment_steps_before_training=1: Number of steps the agent takes in the environment
            before the training cycle starts (the networks are updated after each step in the environment)
        :param training_steps_per_update=1: Number of times the networks are update per update cykle
        :param max_environment_steps_per_epoch=None: Maximal number of staps taken in the environment in an epoch
        :param pre_sampling_steps=1024: Number of exploration steps sampled to the replay buffer before training starts
        :param save_models=False: Determines if the models are saved per epoch
        """
        self._pre_sample(pre_sampling_steps=pre_sampling_steps)
        print("start training!")
        self._loss_logger = LossLogger()
        self._loss_logger.add_lists([CRITIC_LOSS, ACTOR_LOSS, LOG_PROBS, Q_VALUES, MAX_Q_VALUES], smoothed=100)
        self._loss_logger.add_lists([RETURNS], smoothed=1000)
        return_dict = defaultdict(float)
        observation_dict = self._environment.reset()
        action_probs = defaultdict(lambda: np.zeros(shape=(1,len(ACTIONS))))
        done_dict = {"" :False}
        ret = 0
        epoch = 0
        steps = 0
        j = 0
        render_save = []
        last_render_as_list = []
        thread = Thread(target=render_permanently, args=[last_render_as_list])
        thread.start()
        while True:
            i = 0
            while i < environment_steps_before_training or self._reply_buffer.size() < self._batch_size:
                if epoch%10==0:
                    render_save.append((self._environment.render(),action_probs, self._get_max_q_value(states=observation_dict)))
                if False not in done_dict.values() or (max_environment_steps_per_epoch is not None and j >= max_environment_steps_per_epoch):
                    if epoch%10==0:
                        print("epoch:", epoch, "steps:", steps, "actor-loss: {:.2f}".format(self._loss_logger.last(ACTOR_LOSS)), "critic-loss: {:.2f}".format(self._loss_logger.last(CRITIC_LOSS)), "return: {:.2f}".format(self._loss_logger.last(RETURNS)), "avg return: {:.2f}".format(self._loss_logger.avg_last(RETURNS, 10)))
                        last_render_as_list.append(render_save)
                        render_save = []
                        if len(last_render_as_list)>1:
                            last_render_as_list.pop(0)
                    observation_dict = self._environment.reset()
                    self._loss_logger.add_value(identifier=RETURNS,value=ret)
                    if save_models:
                        self.save_models(f"SAC_{epoch}_{steps}_{ret}_{datetime.datetime.now()}")
                    ret = 0
                    return_dict = defaultdict(float)
                    epoch += 1
                    if epoch % 1000 == 0:
                        self.test(n_samples=20, verbose_samples=0)
                    if epoch%1000==0 or epoch>=epochs:
                        self.save_models(name="")
                    if epoch%1000==0 or epoch >=epochs:
                        thread = Thread(target=plot_multiple,args=[self._loss_logger.all_smoothed()])
                        thread.start()
                    if epoch >= epochs:
                        print("training finished!")
                        return
                (actions_dict, new_observation_dict, reward_dict, done_dict), action_probs = self.act_stochastic(observation_dict)
                self._reply_buffer.add_transition(state=observation_dict, action=actions_dict, reward=reward_dict, state_=new_observation_dict, done=done_dict)
                observation_dict = new_observation_dict
                steps += 1
                ret += sum(reward_dict.values())
                return_dict = {key: return_dict[key]+reward_dict[key] for key in reward_dict.keys()}
                i += 1
                j += 1
            for _ in range(training_steps_per_update):
                self.learn()

    def learn(self)->None:
        for _ in range(TRAININGS_PER_TRAINING):
            states, actions, rewards, states_prime, dones = self._reply_buffer.sample_batch()
            critic_loss, log_probs = self.train_step_critic(
                states=tf.reshape(tensor=states,shape=self.extended_shape),
                actions=tf.reshape(tensor=actions,shape=(self._batch_size, len(self._agent_ids)*self._environment.stats.action_dimension)),
                rewards=tf.reshape(tensor=rewards, shape=(self._batch_size, len(self._agent_ids))),
                states_prime=tf.reshape(tensor=states_prime,shape=self.extended_shape),
                dones=tf.reshape(tensor=dones, shape=(self._batch_size, len(self._agent_ids))),
                )
            self.update_target_weights()
            actor_loss, q_values, max_q_values = self.train_step_actor(states)
            self._loss_logger.add_aggregatable_values({CRITIC_LOSS: critic_loss, LOG_PROBS:log_probs, ACTOR_LOSS: actor_loss, Q_VALUES: q_values, MAX_Q_VALUES: max_q_values})
        self._loss_logger.avg_aggregatables([CRITIC_LOSS, LOG_PROBS, ACTOR_LOSS, Q_VALUES, MAX_Q_VALUES])


    @tf.function
    def train_step_critic(self, states, actions, rewards, states_prime, dones):
        _, action_probs, log_probs = self.sample_actions_prime_and_log_probs_from_policy(states=states_prime)
        flattened_states_prime = tf.reshape(states_prime,shape=self.agent_flattened_shape)
        flattened_states = tf.reshape(states,shape=self.agent_flattened_shape)
        q1 = tf.reshape(self._critic_1_t(flattened_states_prime),shape=action_probs.shape)
        q2 = tf.reshape(self._critic_2_t(flattened_states_prime),shape=action_probs.shape)
        q_r = tfm.minimum(q1, q2) - self._alpha * log_probs
        q_r_mean = tf.math.reduce_sum(action_probs * q_r, axis=2)
        targets =  self._reward_scale * tf.reshape(rewards,q_r_mean.shape) + self._gamma * (1 - tf.reshape(dones,q_r_mean.shape)) * q_r_mean
        loss_1 = self._critic_update(self._critic_1, flattened_states, actions, targets)
        loss_2 = self._critic_update(self._critic_2, flattened_states, actions, targets)
        return tf.add(loss_1, loss_2), log_probs

    def _critic_update(self, critic, states, actions, targets):
        with tf.GradientTape() as tape:
            q = tf.reduce_sum(tf.reshape(critic(states)*actions,shape=(BATCH_SIZE,len(self._agent_ids), len(ACTIONS))),axis=2)
            loss = 0.5 * self._mse(targets, q)
        gradients = tape.gradient(loss, critic.trainable_variables)
        critic.optimizer.apply_gradients(zip(gradients, critic.trainable_variables))
        return loss

    @tf.function
    def train_step_actor(self, states)->Tuple[float,float, float]:
        losses = []
        q_values = []
        max_q_values = []
        reshaped_states = tf.reshape(states, shape=self.agent_flattened_shape)
        for index, id in enumerate(self._agent_ids):
            actor = self._get_actor(agent_id=id)
            with tf.GradientTape() as tape:
                _, actions_probs, log_probs = self.sample_actions_prime_and_log_probs_from_policy(states = states)
                q1 = self._critic_1(reshaped_states)[:,index * len(ACTIONS):(index + 1) * len(ACTIONS)]
                q2 = self._critic_2(reshaped_states)[:,index*len(ACTIONS):(index+1)*len(ACTIONS)]
                entropy_part = actions_probs[:,index]*self._alpha * log_probs[:,index]
                q_part = actions_probs[:,index]*tfm.minimum(q1, q2)
                sum_part = tfm.reduce_sum(entropy_part - q_part, axis=1)
                losses.append(tfm.reduce_mean(sum_part))
                q_values.append(tfm.reduce_mean(tfm.minimum(q1, q2)))
                max_q_values.append(tf.reduce_max(tfm.minimum(q1,q2)))
            # equal to loss = -tfm.reduce_mean(tfm.minimum(q1, q2) - self._alpha * log_probs)
            gradients = tape.gradient(losses[-1], actor.trainable_variables)
            actor.optimizer.apply_gradients(zip(gradients, actor.trainable_variables))
        return tf.reduce_mean(losses), tf.reduce_mean(q_values), tf.reduce_mean(max_q_values)

    @tf.function
    def sample_actions_prime_and_log_probs_from_policy(self, states):
        states_prime_dict = {agent_id: states[:, index,:,:] for index, agent_id in enumerate(self._agent_ids)}
        actions_prime, action_probs, log_probs = [
            tf.reshape(list_,shape=(BATCH_SIZE,len(self._agent_ids), len(ACTIONS))) for list_ in
            #tf.concat(values=list_,axis=1) for list_ in
            zip(*[self.sample_actions_from_policy(state=states_prime_dict[agent_id], agent_id=agent_id) for agent_id in
                  self._agent_ids])
        ]
        return actions_prime, action_probs, log_probs

    @tf.function
    def sample_actions_from_policy(self, state, agent_id: str):
        #mu, sigma = self._get_actor(agent_id=agent_id)(state)
        #distribution = tfd.Normal(mu, sigma)
        #actions = distribution.sample()
        #log_probs = distribution.log_prob(actions)
        #actions = tfm.tanh(actions)
        #log_probs -= tfm.log(1 - tfm.pow(actions, 2) + 1e-6)  # + 1e-6 because log undefined for 0
        #log_probs = tfm.reduce_sum(log_probs, axis=-1, keepdims=True)
        probabilities = self._get_actor(agent_id=agent_id)(state)
        log_probs = tf.math.log(probabilities)
        actions = tf.random.categorical(logits=log_probs,num_samples=len(self._agent_ids))[:,self._agent_ids.index(agent_id)] # todo
        #assert float("nan") not in actions.numpy()
        #assert float("nan") not in log_probs.numpy()
        #assert float("nan") not in probabilities.numpy()
        return tf.one_hot(actions,depth=len(ACTIONS)), probabilities, log_probs

    def act_deterministic(self, state):
        actions_prime = {}
        probabilities = {}
        for agent_id in self._agent_ids:
            probabilities[agent_id] = self._get_actor(agent_id=agent_id)(tf.convert_to_tensor([state[agent_id]], dtype=tf.float32))[0]
            actions_prime[agent_id] = tf.reshape(tf.one_hot(tf.argmax(probabilities[agent_id]),depth=len(ACTIONS)), shape=(1,len(ACTIONS)))
        return self._act(actions_prime), probabilities

    def act_stochastic(self, state) -> Tuple[Tuple, Dict[str, np.ndarray]]:
        actions_prime = {agent_id:
                             (self.sample_actions_from_policy(tf.convert_to_tensor([state[agent_id]], dtype=tf.float32), agent_id=agent_id)
                              )
                         for index, agent_id in enumerate(self._agent_ids)}
        return self._act({key: action[0] for key, action in actions_prime.items()}), {key: action[1] for key, action in actions_prime.items()}

    def _act(self, all_actions):
        communications = {agent_id: action[0][len(ACTIONS):] for agent_id, action in all_actions.items()}
        selected_communications = {agent_id: self._select_communication(communication=com) for agent_id, com in communications.items()}
        selected_actions = {agent_id: ACTIONS[np.argmax(action[0][:len(ACTIONS)])] for agent_id, action in all_actions.items()}
        actions_dict = {agent_id: (selected_actions[agent_id], selected_communications[agent_id]) for agent_id in all_actions.keys()}
        observation_prime, reward, done, truncated, _ = self._environment.step(actions_dict)
        return all_actions, observation_prime, reward, {agent_id: done[agent_id] or truncated[agent_id] for agent_id in self._agent_ids}

    def _select_communication(self, communication: np.ndarray)->np.ndarray:
        number_channels = self._environment.stats.number_communication_channels
        vocab_size =self._environment.stats.size_vocabulary
        base_array = np.zeros(shape=(number_channels*vocab_size))
        for base_index in range(0, len(base_array), vocab_size+1):
            if communication[base_index]>0:
                index = np.argmax(communication[base_index+1:base_index+2+vocab_size])
                base_array[base_index+index] = 1
        return base_array


    def _pre_sample(self, pre_sampling_steps: int):
        print(f"Random exploration for {pre_sampling_steps} steps!")
        observation_dict = self._environment.reset()
        ret = 0
        for _ in range(pre_sampling_steps):
            (actions_dict, new_observation_dict, reward_dict, done_dict), _ = self.act_stochastic(observation_dict)
            ret += sum(reward_dict.values())
            self._reply_buffer.add_transition(state=observation_dict, action=actions_dict, reward=reward_dict, state_=new_observation_dict, done=done_dict)
            if False not in done_dict.values():
                ret = 0
                observation_dict = self._environment.reset()
            else:
                observation_dict = new_observation_dict
    def test(self, n_samples: int, verbose_samples: int):
        returns = []
        render_save = []
        for index in range(n_samples):
            observation_dict = self._environment.reset()
            action_probs = defaultdict(lambda: np.zeros(shape=(1,len(ACTIONS))))
            return_ = 0
            while True:
                if index< verbose_samples:
                    render_save.append((self._environment.render(),action_probs))
                (actions_dict, new_observation_dict, reward_dict, done_dict), action_probs = self.act_deterministic(observation_dict)
                if index<verbose_samples:
                    print(actions_dict)
                return_ += sum(reward_dict.values())
                if False not in done_dict.values():
                    returns.append(return_)
                    if index< verbose_samples:
                        render_save.append(self._environment.render())
                    print(f"Finished test episode {index} with a return of {return_} after {self._environment.stats.time_step} time steps.")
                    break
                observation_dict = new_observation_dict
            #if index == 0:
            #    thread = Thread(target=render_episode,args=[render_save])
            #    thread.start()
        print(f"The average return is {np.mean(returns)}")

    @property
    def extended_shape(self):
        return (self._batch_size, len(self._agent_ids), TIME_STEPS, self._environment.stats.observation_dimension)
    @property
    def agent_flattened_shape(self):
        return (self._batch_size, TIME_STEPS, self._environment.stats.observation_dimension*len(self._agent_ids))
    def _get_actor(self, agent_id: str) -> tf.keras.Model:
        return self._actor if self._self_play else self._actors[agent_id]