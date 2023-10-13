import numpy as np
import tensorflow as tf
class ActionSampler:
    
    def __init__(self, actor_uses_log_probs: bool, generators, actor, agent_ids):
        self._actor = actor
        self._agent_ids = agent_ids
        self._generators = generators
        self._actor_uses_log_probs = actor_uses_log_probs

    def batched_compute_actions_one_hot_and_probs_or_log_probs(self, state, deterministic):
            actions_one_hot, probs_or_log_probs = self._batched_compute_actions_one_hot_and_probs_or_log_probs(state, deterministic)
            if state.shape[0] == 1:
                actions_one_hot, probs_or_log_probs = tf.expand_dims(actions_one_hot, axis=1), probs_or_log_probs
            return np.moveaxis(np.array(actions_one_hot), 0, 1), np.moveaxis(np.array(probs_or_log_probs), 0, 1)

    @tf.function
    def _batched_compute_actions_one_hot_and_probs_or_log_probs(self, state, deterministic):
            actions_one_hot, log_probs, probs = list(
                zip(*[self.sample_actions(deterministic=deterministic, generator_index=index,
                                          states=tf.convert_to_tensor(state[:, :, index, :]),
                                          actor=self._actor) for index in range(len(self._agent_ids))]))
            if self._actor_uses_log_probs:
                return actions_one_hot, log_probs
            else:
                return actions_one_hot, probs

    @tf.function
    def sample_actions(self, states, actor, generator_index:int = 0, deterministic=False):
        output_groups = actor(states) # first one for actions, rest for communication channels
        output_groups = output_groups if type(output_groups) == list else [output_groups]
        if self._actor_uses_log_probs:
            log_prob_groups = output_groups
        else:
            log_prob_groups = [tf.math.log(probabilities) for probabilities in output_groups]
        if deterministic:
            action_groups = [tf.argmax(probabilities, axis=1) for probabilities in log_prob_groups]
        else:
            action_groups = [tf.random.stateless_categorical(logits=log_probs, num_samples=1, seed=self._generators[generator_index].make_seeds(2)[0])[:,0] for log_probs in log_prob_groups]
        one_hot_action_groups = [tf.one_hot(actions, depth=probabilities.shape[-1]) for actions, probabilities in zip(action_groups,log_prob_groups)]
        probability_groups = None if self._actor_uses_log_probs else  tf.concat(output_groups,axis=-1)
        return tf.squeeze(tf.concat(one_hot_action_groups,axis=-1)), tf.concat(log_prob_groups,axis=-1), probability_groups
