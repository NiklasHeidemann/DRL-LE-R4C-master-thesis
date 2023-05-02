from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

from SAC.SACagent import SACAgent
from environment.env import CoopGridWorld, DEFAULT_COMMUNCIATIONS
import numpy as np

from params import NEG_REWARD, SEED

NUMBER_SAMPLES = 1000
TEST_SIZE = 0.2
class TrainPredictGoal:

    def __call__(self, environment: CoopGridWorld, agent: SACAgent):
        X, y = self.sample(environment=environment, agent=agent)
        Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, test_size=TEST_SIZE)
        classifier = RandomForestClassifier(random_state=SEED)
        classifier.fit(Xtrain, ytrain)
        print("Balanced accuracy:", balanced_accuracy_score(ytest, classifier.predict(Xtest)))
    def sample(self, environment: CoopGridWorld, agent: SACAgent):
        communications = []
        type_of_returns = []
        for index in range(NUMBER_SAMPLES):
            observation_dict = environment.reset()
            while True:
                    (actions_dict, new_observation_dict, reward_dict, done_dict), _ = agent.act(
                        observation_dict, deterministic=True, env=environment)
                    observation_dict = new_observation_dict
                    if False not in done_dict.values():
                        if not sum(list(reward_dict.values()))> NEG_REWARD:
                            break
                        all_communications = environment._communications
                        number_dummy_communications = max(0, 5-len(all_communications))
                        trimmed_communications = [{agent_id: DEFAULT_COMMUNCIATIONS for agent_id in agent._agent_ids}]*number_dummy_communications + all_communications[-(min(5,len(all_communications))):]
                        communications.append(np.concatenate([np.concatenate(list(dict_.values())) for dict_ in trimmed_communications]))
                        colors = {np.where(environment._grid[position] != 0)[0][0] for agent_id, position in environment._agent_positions.items()}
                        assert len(colors)==1
                        type_of_returns.append(list(colors)[0])
                        break
        return communications, type_of_returns

