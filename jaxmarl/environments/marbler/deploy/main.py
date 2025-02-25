"""
Executed deployment script.
"""

import argparse
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
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='', help='path to deployment files')
    args = parser.parse_args()

    config_path = os.path.join(module_dir, args.path, 'config.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    num_agents = config.pop('num_agents')
    max_steps = config.pop('max_steps')
    config = objectview(config)

    if module_dir == '':
        scenario_module = importlib.import_module(f"{config.scenario.lower()}")
    else:
        scenario_module = importlib.import_module(f"jaxmarl.environments.marbler.scenarios.{config.scenario.lower()}")
    scenario = getattr(scenario_module, config.scenario)
    env = scenario(num_agents, max_steps, **config.__dict__)
