"""
Generates a folder with all files necessary for Robotarium deployment.
"""
import argparse
import torch
import os
import json
import yaml
import importlib
import numpy as np
import shutil

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
        param = np.array(param)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='experiment', help='folder to save deployment files')
    args = parser.parse_args()

    module_dir = os.path.dirname(__file__)
    config_path = os.path.join(module_dir, 'config.yaml')

    # get experiment output dir
    output_dir = os.path.join(module_dir, 'robotarium_submissions', args.name)
    os.makedirs(output_dir, exist_ok=True)
    
    # load config
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
    torch.save(actor.state_dict(), os.path.join(output_dir, 'agent.tiff'))

    # update config and save as .npy
    config_output_path = os.path.join(output_dir, 'config.npy')
    shutil.copy(config_path, config_output_path)
    with open(config_output_path, 'r') as file:
        data = file.read()
    data = data.replace(config.model_weights, 'agent.tiff')
    data = data.replace('"save_gif": True', '"save_gif": False')
    with open(config_output_path, 'w') as file:
            file.write(data)
    
    # copy scenario and constants files
    scenario_py = f'{config.scenario.lower()}.py'
    scenario_path = os.path.join(
        "/".join(module_dir.split("/")[:-1]),
        'scenarios',
        scenario_py
    )
    scenario_output_path = os.path.join(output_dir, scenario_py)
    shutil.copy(scenario_path, scenario_output_path)

    constants_path = os.path.join(
        "/".join(module_dir.split("/")[:-1]),
        'constants.py'
    )
    constants_output_path = os.path.join(output_dir, 'constants.py')
    shutil.copy(constants_path, constants_output_path)

    # copy model, robotarium_env, and main files
    shutil.copy(os.path.join(module_dir, config.model_file), os.path.join(output_dir, config.model_file))
    shutil.copy(os.path.join(module_dir, 'robotarium_env.py'), os.path.join(output_dir, 'robotarium_env.py'))
    shutil.copy(os.path.join(module_dir, 'main.py'), os.path.join(output_dir, 'main.py'))
    