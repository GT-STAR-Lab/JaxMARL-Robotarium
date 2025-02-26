"""
Executed deployment script.
"""

import torch
import numpy as jnp
import os
import yaml
import json
import importlib

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d
        self.__json__ = json.dumps(d, indent=4)

if __name__ == "__main__":
    module_dir = os.path.dirname(__file__)
    if module_dir.split("/")[-1] != "deploy":
        module_dir = ""

    # load config
    config_path = os.path.join(module_dir, 'config.npy')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    num_agents = config.pop('num_agents')
    max_steps = config.pop('max_steps')
    config = objectview(config)

    # set seed
    jnp.random.seed(config.seed)

    # load scenario
    scenario_module = importlib.import_module(f"{config.scenario.lower()}")
    scenario = getattr(scenario_module, config.scenario)
    env = scenario(num_agents, max_steps, **config.__dict__)

    # load actor
    actor_module = importlib.import_module(f"{config.model_file[:-3]}")
    actor_class = getattr(actor_module, config.model_class)
    input_dim = config.input_dim + num_agents if config.preprocess_obs else config.input_dim
    actor = actor_class(input_dim, config.output_dim, config.hidden_dim)
    actor_weights = torch.load(config.model_weights)
    actor.load_state_dict(actor_weights)
   
    state = env.initial_state
    obs = env.get_obs(state)
    hs = jnp.zeros((num_agents, config.hidden_dim))
    one_hot_id = jnp.eye(num_agents)
    for i in range(max_steps):
        # get agent action
        if config.preprocess_obs:
            obs = jnp.hstack([jnp.vstack([obs_i for obs_i in obs.values()]), one_hot_id])
        else:
            obs = jnp.vstack([obs_i for obs_i in obs.values()])
        obs = torch.tensor(obs).to(torch.float32)
        hs = torch.tensor(hs).to(torch.float32)
        qvals, hs = actor(obs, hs)

        actions = {f'agent_{i}': jnp.argmax(qvals[i].detach().numpy()) for i in range(num_agents)}

        obs, state, reward, dones, info = env.step_env(None, state, actions)

        env.visualize_robotarium(state)

    
    if config.save_gif:
        import imageio
        imageio.mimsave(f'{config.scenario.lower()}.gif', env.frames, duration=50, loop=0)
    
    env.robotarium.call_at_scripts_end()
