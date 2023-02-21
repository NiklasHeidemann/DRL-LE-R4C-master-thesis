from agent import ManhattanWalker
from grid_world import DemoGridWorld

MAX_TIMESTEPS = 20

env = DemoGridWorld()
env.seed(0)
obs = env.reset()
agents = {agent_id: ManhattanWalker() for agent_id in env.agents}
returns = {agent_id: 0 for agent_id in agents.keys()}

for time_step in range(MAX_TIMESTEPS):
    print("Step", time_step)
    actions = {agent_id: agents[agent_id].pick_action(personal_obs) for agent_id,personal_obs in obs.items()}
    obs,rewards, _, _, _ = env.step(actions=actions)
    print(env.state())
    returns = {agent_id: returns[agent_id]+rewards[agent_id] for agent_id in agents.keys()}
print (returns)