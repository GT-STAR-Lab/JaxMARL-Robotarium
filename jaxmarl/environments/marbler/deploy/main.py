"""
Executed deployment script.
"""

import numpy as jnp
import os
import yaml
import json
import importlib
import imageio

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d
        self.__json__ = json.dumps(d, indent=4)

if __name__ == "__main__":
    module_dir = os.path.dirname(__file__)
    if module_dir.split("/")[-1] != "deploy":
        module_dir = ""

    config_path = os.path.join(module_dir, 'config.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    num_agents = config.pop('num_agents')
    max_steps = config.pop('max_steps')
    config = objectview(config)

    scenario_module = importlib.import_module(f"{config.scenario.lower()}")
    scenario = getattr(scenario_module, config.scenario)
    env = scenario(num_agents, max_steps, **config.__dict__)
   
    state = env.initial_state
    for i in range(max_steps):
        goal_pos = state.p_pos[num_agents:, :2]
        actions = jnp.array([[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]])
        dir_to_goal = goal_pos - state.p_pos[:num_agents, :2]
        dir_to_goal = dir_to_goal / jnp.linalg.norm(dir_to_goal, axis=1)[:, None]
        dots = jnp.array([jnp.dot(actions, dir_to_goal[i]) for i in range(num_agents)])
        best_action = jnp.argmax(dots, axis=1)
        actions = {f'agent_{i}': best_action[i] for i in range(num_agents)}

        obs, state, reward, dones, info = env.step_env(None, state, actions)

        env.visualize_robotarium(state)

    
    if config.save_gif:
        imageio.mimsave(f'{config.scenario.lower()}.gif', env.frames, duration=50, loop=0)
