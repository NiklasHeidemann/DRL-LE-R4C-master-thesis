from random import random, sample, choices

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.model_selection import train_test_split

from SAC.SACagent import SACAgent
from environment.env import CoopGridWorld, DEFAULT_COMMUNCIATIONS
import numpy as np

from params import NEG_REWARD, SEED

NUMBER_SAMPLES = 100
TEST_SIZE = 0.2
class TrainPredictGoal:

    def __call__(self, environment: CoopGridWorld, agent: SACAgent)->float:
        X, y = self.sample(environment=environment, agent=agent)
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
    def sample(self, environment: CoopGridWorld, agent: SACAgent):
        communications = []
        type_of_returns = []
        for index in range(NUMBER_SAMPLES):
            observation_array, _ = environment.reset()
            while True:
                    (_, new_observation_array, reward_array, done),_ = agent.act(
                        observation_array, deterministic=True, env=environment)
                    observation_array = new_observation_array
                    if done:
                        if not sum(reward_array)> NEG_REWARD:
                            break
                        all_communications = environment._communications
                        number_dummy_communications = max(0, 5-len(all_communications))
                        trimmed_communications = [{agent_id: DEFAULT_COMMUNCIATIONS(environment.stats.size_vocabulary, environment.stats.number_communication_channels) for agent_id in agent._agent_ids}]*number_dummy_communications + all_communications[-(min(5,len(all_communications))):]
                        communications.append(np.concatenate([np.concatenate(list(dict_.values())) for dict_ in trimmed_communications]))
                        colors = [np.where(environment._grid[position] != 0)[0][0] for agent_id, position in environment._agent_positions.items() if len(np.where(environment._grid[position] != 0)[0])>0]
                        most_freq_color = max(set(colors), key=colors.count)
                        type_of_returns.append(most_freq_color)
                        break
        return communications, type_of_returns

