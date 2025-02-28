import torch
import os
import json
import yaml
import importlib
import numpy as jnp

from navigation import Navigation

from safetensors.flax import load_file

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d
        self.__json__ = json.dumps(d, indent=4)

def flax_to_torch(flax_state_dict, torch_state_dict):
    """
    Convert flax state dict to torch state dict

    Args:
        flax_state_dict: (Dict) dictionary of names and associated parameters
        torch_state_dict: (Dict) dictionary of names and associated parameters

    Returns:
        (Dict) matching torch_state_dict format with parameters from flax_state_dict
    """

    def _bias(param):
        return torch.from_numpy(param)

    def _dense(param):
        return torch.from_numpy(param.T)

    for name, param in flax_state_dict.items():
        param = jnp.array(param)
        # skip all non agent parameters
        if 'agent' not in name:
            continue
        
        if 'Dense' in name:
            if 'kernel' in name:
                torch_state_dict[f'{name.split(",")[-2]}.weight'] = _dense(param)
            if 'bias' in name:
                torch_state_dict[f'{name.split(",")[-2]}.bias'] = _bias(param)

        if 'GRUCell' in name:
            gru_param = name.split(",")[-2]
            N = param.shape[0]
            param_id = 'ih' if 'i' in gru_param else 'hh'
            if 'r' in gru_param:
                if 'kernel' in name:
                    torch_state_dict[f'{name.split(",")[-3]}.weight_{param_id}'][:N,:] = _dense(param)
                if 'bias' in name:
                    torch_state_dict[f'{name.split(",")[-3]}.bias_{param_id}'][:N] = _bias(param)
            if 'z' in gru_param:
                if 'kernel' in name:
                    torch_state_dict[f'{name.split(",")[-3]}.weight_{param_id}'][N:2*N,:] = _dense(param)
                if 'bias' in name:
                    torch_state_dict[f'{name.split(",")[-3]}.bias_{param_id}'][N:2*N] = _bias(param)
            if 'n' in gru_param:
                if 'kernel' in name:
                    torch_state_dict[f'{name.split(",")[-3]}.weight_{param_id}'][2*N:,:] = _dense(param)
                if 'bias' in name:
                    torch_state_dict[f'{name.split(",")[-3]}.bias_{param_id}'][2*N:] = _bias(param)
        
    return torch_state_dict


# load config
module_dir = os.path.dirname(__file__)
config_path = os.path.join(module_dir, 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
config = objectview(config)

# load actor and save as .tiff
actor_module = importlib.import_module(f"{config.model_file[:-3]}")
actor_class = getattr(actor_module, config.model_class)
input_dim = config.input_dim + config.num_agents if config.preprocess_obs else config.input_dim
actor = actor_class(input_dim, config.output_dim, config.hidden_dim)
weights = load_file(config.model_weights)

state_dict = flax_to_torch(weights, actor.state_dict())
actor.load_state_dict(state_dict)

# load scenario
env = Navigation(**config.__dict__)
num_agents = config.num_agents
max_steps = config.max_steps

# rollout
state = env.initial_state
obs = env.get_obs(state)
hs = torch.from_numpy(jnp.zeros((num_agents, config.hidden_dim))).to(torch.float32)
one_hot_id = jnp.eye(num_agents)
for i in range(max_steps):
    # get agent action
    if config.preprocess_obs:
        obs = jnp.hstack([jnp.vstack([obs_i for obs_i in obs.values()]), one_hot_id])
    else:
        obs = jnp.vstack([obs_i for obs_i in obs.values()])
    obs = torch.from_numpy(obs).to(torch.float32)
    qvals, hs = actor(obs, hs)

    actions = {f'agent_{i}': jnp.argmax(qvals[i].detach().numpy()) for i in range(num_agents)}

    obs, state, reward, dones, info = env.step_env(None, state, actions)

    env.visualize_robotarium(state)


if config.save_gif:
    import imageio
    imageio.mimsave(f'{config.scenario.lower()}.gif', env.frames, duration=50, loop=0)

env.robotarium.call_at_scripts_end()