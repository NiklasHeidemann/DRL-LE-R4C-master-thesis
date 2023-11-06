from random import choices

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.model_selection import train_test_split

from environment.env import CoopGridWorld, DEFAULT_COMMUNCIATIONS
from environment.envbatcher import EnvBatcher
from training.Agent import Agent
from training.SAC.SACagent import SACAgent

SEED=12932
NUMBER_SAMPLES = 100
MAX_ATTEMPTS = 400
BATCH_SIZE = 40
TEST_SIZE = 0.2
class TrainPredictGoal:

    def __init__(self, environment: CoopGridWorld):
        self._environment_batcher = EnvBatcher(env=environment, batch_size=BATCH_SIZE)
    def __call__(self, agent: Agent)->float:
        X, y = self.sample_batched(environment_batcher=self._environment_batcher, agent=agent)
        if len(y) < 10:
            print("Not enough samples")
            return 0
        Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, test_size=TEST_SIZE)
        classifier = RandomForestClassifier(random_state=SEED)
        classifier.fit(Xtrain, ytrain)
        random_predictions = choices(population=classifier.classes_, k=len(ytest))
        pred = classifier.predict(Xtest)
        print("Balanced accuracy:", balanced_accuracy_score(ytest, pred), "; Accuracy:", accuracy_score(ytest, pred))
        print("Random: Balanced accuracy:", balanced_accuracy_score(ytest, random_predictions), "; Accuracy:", accuracy_score(ytest, random_predictions))
        train_pred = classifier.predict(Xtrain)
        print("Trainset: Balanced accuracy:", balanced_accuracy_score(ytrain, train_pred), "; Accuracy:", accuracy_score(ytrain, train_pred))
        return balanced_accuracy_score(ytest, pred)

    def sample_batched(self, environment_batcher: EnvBatcher, agent: Agent):
        communications = []
        type_of_returns = []
        finished_samples = 0
        attempts = 0
        observation_arrays = environment_batcher.reset_all()
        while finished_samples < NUMBER_SAMPLES and attempts < MAX_ATTEMPTS:
            (_, new_observation_arrays, reward_arrays, done_mask),_, _ = agent.act_batched(
                observation_arrays, deterministic=True, env_batcher=environment_batcher, include_social=False)
            usable_mask = np.bitwise_and(done_mask, np.sum(reward_arrays, axis=1)>0)
            envs = environment_batcher.get_envs(mask=usable_mask)
            batch_all_communications = [env._communications for env in envs]
            batch_number_dummy_communications = [max(0, 5-len(all_communications)) for all_communications in batch_all_communications]
            batch_trimmed_communications = [[{agent_id: DEFAULT_COMMUNCIATIONS(env.stats.size_vocabulary, env.stats.number_communication_channels) for agent_id in agent._agent_ids}]*number_dummy_communications + all_communications[-(min(5,len(all_communications))):]
                                            for env, number_dummy_communications, all_communications in zip(envs, batch_number_dummy_communications, batch_all_communications)]

            communications.extend([np.concatenate([np.concatenate(list(dict_.values())) for dict_ in trimmed_communications]) for trimmed_communications in batch_trimmed_communications])
            batch_colors = [[np.where(env._grid[position] != 0)[0][0] for agent_id, position in env._agent_positions.items() if len(np.where(env._grid[position] != 0)[0])>0]for env in envs]
            batch_most_freq_color = [max(set(colors), key=colors.count) for colors in batch_colors]
            type_of_returns.extend(batch_most_freq_color)
            finished_samples += sum(usable_mask)
            attempts += np.sum(done_mask)
            observation_arrays = environment_batcher.reset(mask=done_mask, observation_array=new_observation_arrays)
        return communications, type_of_returns


