import os
import copy
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Any

import chex
import optax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from gymnax.wrappers.purerl import LogWrapper
import hydra
from omegaconf import OmegaConf
import flashbax as fbx
import wandb

from jaxmarl import make
from jaxmarl.environments.smax import map_name_to_scenario
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.wrappers.baselines import (
    SMAXLogWrapper,
    MPELogWrapper,
    LogWrapper,
    CTRolloutManager,
)
from jaxmarl.wrappers.baselines import load_params


class ScannedRNN(nn.Module):

    @partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        hidden_size = rnn_state.shape[-1]
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(hidden_size, *resets.shape),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(hidden_size)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(hidden_size, *batch_size):
        # Use a dummy key since the default state init fn is just zeros.
        return nn.GRUCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (*batch_size, hidden_size)
        )


class QNetwork(nn.Module):
    action_dim: int
    hidden_size: int = 512
    num_layers: int = 4
    norm_input: bool = False
    norm_type: str = "layer_norm"
    dueling: bool = False

    @nn.compact
    def __call__(self, hidden, x, dones, train: bool = False):
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            normalize = lambda x: nn.BatchNorm(use_running_average=not train)(x)
        else:
            normalize = lambda x: x

        # if self.norm_input:
        #     x = nn.BatchNorm(use_running_average=not train)(x)
        # else:
        #     # dummy normalize input in any case for global compatibility
        #     x_dummy = nn.BatchNorm(use_running_average=not train)(x)

        for l in range(self.num_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = normalize(x)
            x = nn.relu(x)

        rnn_in = (x, dones)
        hidden, x = ScannedRNN()(hidden, rnn_in)

        if self.dueling:
            adv = nn.Dense(self.action_dim)(x)
            val = nn.Dense(1)(x)
            q_vals = val + adv - jnp.mean(adv, axis=-1, keepdims=True)
        else:
            q_vals = nn.Dense(self.action_dim)(x)

        return hidden, q_vals


def eval(config, env):
    def batchify(x: dict):
        return jnp.stack([x[agent] for agent in env.agents], axis=0)

    def unbatchify(x: jnp.ndarray):
        return {agent: x[i] for i, agent in enumerate(env.agents)}
    
    def get_greedy_actions(q_vals, valid_actions):
        unavail_actions = 1 - valid_actions
        q_vals = q_vals - (unavail_actions * 1e10)
        return jnp.argmax(q_vals, axis=-1)

    def run(rng, params):

        # INIT ENV
        original_seed = rng[0]
        rng, _rng = jax.random.split(rng)
        test_env = CTRolloutManager(
            env, batch_size=config["TEST_NUM_ENVS"], preprocess_obs=config.get("PREPROCESS_OBS", False)
        )  # batched env for testing (has different batch size)

        network = QNetwork(
            action_dim=test_env.max_action_space,
            hidden_size=config["HIDDEN_SIZE"],
            num_layers=config["NUM_LAYERS"],
            norm_type=config["NORM_TYPE"],
            norm_input=config.get("NORM_INPUT", False),
            dueling=config.get("DUELING", False),
        )

        def get_greedy_metrics(rng, params):
            """Help function to test greedy policy during training"""
            if not config.get("TEST_DURING_TRAINING", True):
                return None

            def _greedy_env_step(step_state, unused):
                env_state, last_obs, last_dones, hstate, rng = step_state
                rng, key_s = jax.random.split(rng)
                _obs = batchify(last_obs)[:, np.newaxis]
                _dones = batchify(last_dones)[:, np.newaxis]
                hstate, q_vals = jax.vmap(
                    partial(network.apply), in_axes=(None, 0, 0, 0, None)
                )(
                    {
                        "params": params,
                    },
                    hstate,
                    _obs,
                    _dones,
                    False,
                )
                q_vals = q_vals.squeeze(axis=1)
                valid_actions = test_env.get_valid_actions(env_state.env_state)
                actions = get_greedy_actions(q_vals, batchify(valid_actions))
                actions = unbatchify(actions)
                obs, env_state, rewards, dones, infos = test_env.batch_step(
                    key_s, env_state, actions
                )
                step_state = (env_state, obs, dones, hstate, rng)
                return step_state, (rewards, dones, infos)

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = test_env.batch_reset(_rng)
            init_dones = {
                agent: jnp.zeros((config["TEST_NUM_ENVS"]), dtype=bool)
                for agent in env.agents + ["__all__"]
            }
            rng, _rng = jax.random.split(rng)
            hstate = ScannedRNN.initialize_carry(
                config["HIDDEN_SIZE"], len(env.agents), config["TEST_NUM_ENVS"]
            )  # (n_agents*n_envs, hs_size)
            step_state = (
                env_state,
                init_obs,
                init_dones,
                hstate,
                _rng,
            )
            step_state, (rewards, dones, infos) = jax.lax.scan(
                _greedy_env_step, step_state, None, config["TEST_NUM_STEPS"]
            )
            metrics = jax.tree.map(
                lambda x: jnp.where(
                    infos["returned_episode"],
                    x,
                    jnp.nan,
                ),
                infos,
            )
            return metrics

        rng, _rng = jax.random.split(rng)
        test_metrics = get_greedy_metrics(_rng, params)

        return test_metrics

    return run


def env_from_config(config):
    env_name = config["ENV_NAME"]
    # smax init neeeds a scenario
    if "smax" in env_name.lower():
        config["ENV_KWARGS"]["scenario"] = map_name_to_scenario(config["MAP_NAME"])
        env_name = f"{config['ENV_NAME']}_{config['MAP_NAME']}"
        env = make(config["ENV_NAME"], **config["ENV_KWARGS"])
        env = SMAXLogWrapper(env)
    # overcooked needs a layout
    elif "overcooked" in env_name.lower():
        env_name = f"{config['ENV_NAME']}_{config['ENV_KWARGS']['layout']}"
        config["ENV_KWARGS"]["layout"] = overcooked_layouts[
            config["ENV_KWARGS"]["layout"]
        ]
        env = make(config["ENV_NAME"], **config["ENV_KWARGS"])
        env = LogWrapper(env)
    elif "mpe" in env_name.lower():
        env = make(config["ENV_NAME"], **config["ENV_KWARGS"])
        env = MPELogWrapper(env)
    else:
        env = make(config["ENV_NAME"], **config["ENV_KWARGS"])
        env = LogWrapper(env)
    return env, env_name


def single_run(config):

    config = {**config, **config["alg"]}  # merge the alg config with the main config
    config["ENV_KWARGS"]["eval"] = True
    # print("Config:\n", OmegaConf.to_yaml(config))

    alg_name = config.get("ALG_NAME", "pqn_rnn")
    env, env_name = env_from_config(copy.deepcopy(config))

    # wandb.init(
    #     entity=config["ENTITY"],
    #     project=config["PROJECT"],
    #     tags=[
    #         alg_name.upper(),
    #         env_name.upper(),
    #         f"jax_{jax.__version__}",
    #         "eval-qmix"
    #     ],
    #     name=f"{alg_name}_{env_name}",
    #     config=config,
    #     mode=config["WANDB_MODE"],
    # )

    rng = jax.random.PRNGKey(config["SEED"])
    params = []
    for i, model_weights in enumerate(os.listdir(config["MODELS_PATH"])):
        path = os.path.join(config["MODELS_PATH"], model_weights)
        if os.path.isdir(path):
            continue
        params.append(load_params(path))
    
    all_outs = []
    for i in range(len(params)):
        run = jax.jit(eval(config, env))
        outs = jax.block_until_ready(run(rng, params[i]))
        all_outs.append(outs)
    
    # Merge the outputs: this will stack outputs for each metric across runs
    merged_outs = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *all_outs)

    # Compute mean and std for each metric
    mean_outs = jax.tree_util.tree_map(lambda x: jnp.nanmean(x), merged_outs)
    std_outs = jax.tree_util.tree_map(lambda x: jnp.nanstd(x), merged_outs)

    # Print the results
    print(f"\n=== {env_name} ===")
    def print_metrics(metrics_dict, stat_name):
        print(f"\n--- {stat_name.upper()} ---")
        for key, val in metrics_dict.items():
            if isinstance(val, dict):
                print(f"{key}:")
                for subkey, subval in val.items():
                    print(f"  {subkey}: {subval}")
            else:
                print(f"{key}: {val}")

    print_metrics(mean_outs, "mean")
    print_metrics(std_outs, "std")



@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    # print("Config:\n", OmegaConf.to_yaml(config))
    single_run(config)


if __name__ == "__main__":
    main()
