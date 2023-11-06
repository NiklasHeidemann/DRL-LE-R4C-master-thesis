from threading import Thread
from typing import List, Dict, Optional

import numpy as np
import tensorflow as tf
from typing_extensions import override

from domain import RenderSaveExtended
from environment.envbatcher import EnvBatcher
from environment.render import render_permanently
from utils.loss_logger import CRITIC_LOSS, ACTOR_LOSS, LOG_PROBS, RETURNS, Q_VALUES, MAX_Q_VALUES, ALPHA_VALUES, \
    ENTROPY, COM_ENTROPY, N_AGENT_RETURNS, TEST_RETURNS, TEST_SOCIAL_RETURNS
from utils.timer import MultiTimer
from training.ExperienceReplayBuffer import STATE_KEY, ACTION_KEY, REWARD_KEY
from training.SAC.ExperienceReplayBuffer import SACExperienceReplayBuffer, STATE_PRIME_KEY, DONE_KEY
from training.SAC.SACagent import SACAgent
from training.Trainer import Trainer


class SACTrainer(Trainer):
    def __init__(self, environment, agent_ids: List[str], state_dim, action_dim, from_save: bool,
                 actor_network_generator, critic_network_generator, max_replay_buffer_size: int, gamma: float, mov_alpha: float, com_alpha: float, env_parallel: int, seed: int, run_name: str,
                 social_reward_weight: float,plotting:bool,
                 social_influence_sample_size: int,epsilon:Optional[float],
                 batch_size: int, target_entropy: float, tau: float, batches_per_epoch: int,
                 environment_steps_per_epoch: int, pre_sampling_steps: int, model_path="model/"):
        super().__init__()
        self._env_batcher = EnvBatcher(env=environment, batch_size=env_parallel)
        replay_buffer = SACExperienceReplayBuffer(state_dims=state_dim, action_dims=action_dim,
                                                  agent_number=len(agent_ids),
                                                  max_size=max_replay_buffer_size, batch_size=batch_size,
                                                  time_steps=environment.stats.recurrency)
        agent = SACAgent(environment=environment, env_batcher=self._env_batcher, seed=seed,
                         agent_ids=agent_ids,
                         actor_network_generator=actor_network_generator,
                         social_influence_sample_size=social_influence_sample_size,
                         social_reward_weight=social_reward_weight,epsilon=epsilon,
                         critic_network_generator=critic_network_generator, gamma=gamma, tau=tau, mov_alpha=mov_alpha, com_alpha=com_alpha,
                         batch_size=batch_size, model_path=model_path, target_entropy=target_entropy)
        metrics = {
            100: [CRITIC_LOSS, ACTOR_LOSS, LOG_PROBS, Q_VALUES, MAX_Q_VALUES, ENTROPY, COM_ENTROPY],
            10: [ALPHA_VALUES],
            3: [TEST_RETURNS, TEST_SOCIAL_RETURNS],
            1000: [RETURNS, N_AGENT_RETURNS(2), N_AGENT_RETURNS(3)]
        }
        self._init(environment=environment, agent=agent, plotting=plotting,replay_buffer=replay_buffer,
                   run_name=run_name, from_save=from_save, metrics=metrics)
        self._batch_size = batch_size
        self._agent_ids = agent_ids
        self._batches_per_epoch = batches_per_epoch
        self._environment_steps_per_epoch = environment_steps_per_epoch
        self._pre_sampling_steps = pre_sampling_steps

    def _env_step(self, observation_array, return_array: np.ndarray):
        (actions_array, new_observation_array, reward_array, done), action_probs, _ = self._agent.act_batched(
            batched_state=observation_array, deterministic=False, env_batcher=self._env_batcher, include_social=True)
        self._replay_buffer.add_transition_batch({
            STATE_KEY: observation_array, ACTION_KEY: actions_array, REWARD_KEY: reward_array,
            STATE_PRIME_KEY: new_observation_array, DONE_KEY: np.expand_dims(done,1)
        }
        )
        observation_array = new_observation_array
        return_array += np.sum(reward_array, axis=1)
        return observation_array, return_array, done, action_probs[0]

    def _extend_render_save(self, render_save: List[RenderSaveExtended], action_probs, observation_array: np.ndarray):
        render_save.append((self._env_batcher.render(index=0), action_probs,
                            self._agent._get_max_q_value(states=observation_array)))

    def _reset_env(self, epoch_array: List[int], steps_total: int, render_save: List[RenderSaveExtended], render: bool,
                   return_array: np.ndarray, done_mask: List[bool], observation_array: np.ndarray):
        if done_mask[0]:
            print("epoch:", epoch_array[0], "steps:", steps_total,
                  "actor-loss: {:.2f}".format(self._loss_logger.last(ACTOR_LOSS)),
                  "critic-loss: {:.2f}".format(self._loss_logger.last(CRITIC_LOSS)),
                  "return: {:.2f}".format(self._loss_logger.last(RETURNS)),
                  "avg return: {:.2f}".format(self._loss_logger.avg_last(RETURNS, 40)))
            if render:
                self._last_render_as_list.append(render_save)
                render_save = []
                if len(self._last_render_as_list) > 1:
                    self._last_render_as_list.pop(0)
        self._loss_logger.add_value_list(identifier=RETURNS, values=return_array[done_mask])
        self._loss_logger.add_value_list(
            values=return_array[tf.math.logical_and(done_mask, (self._env_batcher.env_types == 2))],
            identifier=N_AGENT_RETURNS(2))
        self._loss_logger.add_value_list(
            values=return_array[tf.math.logical_and(done_mask, (self._env_batcher.env_types == 3))],
            identifier=N_AGENT_RETURNS(3))
        return_array[done_mask] = 0
        observation_array = self._env_batcher.reset(observation_array=observation_array, mask=done_mask)
        epoch_array[done_mask] += self._env_batcher.size
        if render and done_mask[0]:
            self._extend_render_save(render_save=render_save, action_probs=np.zeros(
                shape=(len(self._agent_ids), self._environment.stats.action_dimension)),
                                     observation_array=observation_array[0])
        if epoch_array[0] % (self._env_batcher.size * 100) == 0 and done_mask[0]:
            self._every_few_epochs(render=render)
        return observation_array, epoch_array, return_array

    @override
    def train(self, num_epochs: int, render: bool):
        self._pre_sample(pre_sampling_steps=self._pre_sampling_steps)
        if render:
            self._last_render_as_list = []
            thread = Thread(target=render_permanently, args=[self._last_render_as_list])
            thread.start()
        print("start training!")

        observation_array = self._env_batcher.reset_all()
        # default values for first iteration:
        done = np.array([False] * self._env_batcher.size)
        action_probs_0: Dict[str, np.ndarray] = np.zeros(
            shape=(len(self._agent_ids), self._environment.stats.action_dimension))
        return_array = np.zeros(shape=(self._env_batcher.size))

        render_save = []
        epoch_array = np.array(list(range(self._env_batcher.size))) + self.epoch
        steps_total = 0
        while True:
            steps = 0
            while steps < self._environment_steps_per_epoch:
                if render:
                    self._extend_render_save(render_save=render_save, action_probs=action_probs_0,
                                             observation_array=observation_array[0])
                observation_array, epoch_array, ret = self._reset_env(epoch_array=epoch_array, steps_total=steps_total,
                                                                      render_save=render_save, render=render,
                                                                      return_array=return_array, done_mask=done,
                                                                      observation_array=observation_array)
                self.epoch = min(epoch_array)
                if self.epoch >= num_epochs:
                    print("training finished!")
                    return
                observation_array, ret, done, action_probs_0 = self._env_step(
                    observation_array=observation_array, return_array=return_array
                )
                steps += self._env_batcher.size
                steps_total += self._env_batcher.size

            for _ in range(self._batches_per_epoch):
                self.learn()
        return self._loss_logger
    def learn(self) -> None:
        inputs = self._replay_buffer.sample_batch()
        _, actor_metrics = self._agent.train_step_actor(inputs)
        target_shapes = {STATE_KEY: self.extended_shape, ACTION_KEY: (
        self._batch_size, len(self._agent_ids) * self._environment.stats.action_dimension),
                         REWARD_KEY: (self._batch_size, len(self._agent_ids)), STATE_PRIME_KEY: self.extended_shape,
                         DONE_KEY: (self._batch_size)}
        critic_metrics = self._agent.train_step_critic(
            {key: tf.reshape(input, shape=target_shapes[key]) for key, input in inputs.items()}
        )
        # self._agent.train_step_temperature(states)
        self._agent.update_target_weights()
        self._loss_logger.add_aggregatable_values(critic_metrics)
        self._loss_logger.add_aggregatable_values(actor_metrics)
        self._loss_logger.avg_aggregatables(list(critic_metrics.keys()) + list(actor_metrics.keys()))


    def _pre_sample(self, pre_sampling_steps: int):
        print(f"Random exploration for {pre_sampling_steps} steps!")
        observation_array, _ = self._environment.reset()
        ret = 0
        for _ in range(pre_sampling_steps):
            (actions_array, new_observation_array, reward_array, done), _ = self._agent.act(state=observation_array,
                                                                                            deterministic=False,
                                                                                            env=self._environment)
            ret += sum(reward_array)
            inputs = {
                STATE_KEY: [observation_array], ACTION_KEY: [actions_array], REWARD_KEY: [reward_array],
                STATE_PRIME_KEY: [new_observation_array], DONE_KEY: [done]
            }
            self._replay_buffer.add_transition_batch(
                inputs={key: np.expand_dims(array, 0) for key, array in inputs.items()})
            if done:
                ret = 0
                observation_array, _ = self._environment.reset()
            else:
                observation_array = new_observation_array

    @property
    def extended_shape(self):
        return (self._batch_size, len(self._agent_ids), self._environment.stats.recurrency,
                self._environment.stats.observation_dimension)
