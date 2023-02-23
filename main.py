from functools import partial

from SAC.GenericMLPs1D import create_policy_network, create_q_network
from SAC.SoftActorCriticAgent import SACAgent, multiplicative_scaling
from agent import ManhattanWalker
from grid_world import DemoGridWorld
MAX_TIMESTEPS = 20

number_of_big_layers = 2

env = DemoGridWorld(possible_agents=["0_0"])
env.seed(0)
env.reset()

agents = {agent_id: ManhattanWalker() for agent_id in env.agents[1:]}
sac_agent = SACAgent(environment=env, agent_id=env.agents[0], state_dim=(env.observation_space_dim,), action_dim=env.action_space_shape[0],
                  action_scaling=partial(multiplicative_scaling, factors=1),
                  actor_network_generator=partial(create_policy_network, state_dim=env.observation_space_dim,
                                                  action_dim=env.action_space_shape[0],number_of_big_layers=number_of_big_layers),
                  critic_network_generator=partial(create_q_network, state_dim=env.observation_space_dim,
                                                   action_dim=env.action_space_shape[0],number_of_big_layers=number_of_big_layers))
sac_agent.train(epochs=1000, pre_sampling_steps=1000, environment_steps_before_training=1)
agents[env.agents[0]] = sac_agent
returns = {agent_id: 0 for agent_id in agents.keys()}

obs = env.reset()
for time_step in range(MAX_TIMESTEPS):
    print("Step", time_step)
    actions = {agent_id: agents[agent_id].pick_action(personal_obs) for agent_id,personal_obs in obs.items()}
    obs,rewards, _, _, _ = env.step(actions=actions)
    print(env.state())
    returns = {agent_id: returns[agent_id]+rewards[agent_id] for agent_id in agents.keys()}
print (returns)