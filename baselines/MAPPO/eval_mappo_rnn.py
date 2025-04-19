import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Tuple, Union, Dict
import chex

from flax.training.train_state import TrainState
import distrax
import hydra
from omegaconf import DictConfig, OmegaConf
from functools import partial
import jaxmarl
from jaxmarl.wrappers.baselines import MPELogWrapper, JaxMARLWrapper, LogWrapper
from jaxmarl.environments.multi_agent_env import MultiAgentEnv, State

import wandb
import functools
import matplotlib.pyplot as plt
import os
import copy

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


class MPEWorldStateWrapper(JaxMARLWrapper):
    
    @partial(jax.jit, static_argnums=0)
    def reset(self,
              key):
        obs, env_state = self._env.reset(key)
        obs["world_state"] = self.world_state(obs)
        return obs, env_state
    
    @partial(jax.jit, static_argnums=0)
    def step(self,
             key,
             state,
             action):
        obs, env_state, reward, done, info = self._env.step(
            key, state, action
        )
        obs["world_state"] = self.world_state(obs)
        return obs, env_state, reward, done, info

    @partial(jax.jit, static_argnums=0)
    def world_state(self, obs):
        """ 
        For each agent: [agent obs, all other agent obs]
        """
        
        @partial(jax.vmap, in_axes=(0, None))
        def _roll_obs(aidx, all_obs):
            robs = jnp.roll(all_obs, -aidx, axis=0)
            robs = robs.flatten()
            return robs
            
        all_obs = jnp.array([obs[agent] for agent in self._env.agents]).flatten()
        all_obs = jnp.expand_dims(all_obs, axis=0).repeat(self._env.num_agents, axis=0)
        return all_obs
    
    def world_state_size(self):
        spaces = [self._env.observation_space(agent) for agent in self._env.agents]
        return sum([space.shape[-1] for space in spaces])

class ScannedRNN(nn.Module):
    @functools.partial(
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
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(*rnn_state.shape),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            self.config["HIDDEN_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(self.config["HIDDEN_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        action_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=action_logits)

        return hidden, pi


def eval(config, test_env):

    def batchify(x: dict, agent_list, num_actors):
        x = jnp.stack([x[a] for a in agent_list])
        return x.reshape((num_actors, -1))


    def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
        x = x.reshape((num_actors, num_envs, -1))
        return {a: x[i] for i, a in enumerate(agent_list)}

    def run(rng, params):
        env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
        config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
        config["NUM_TEST_ACTORS"] = env.num_agents * config["NUM_TEST_ENVS"]
        config["NUM_UPDATES"] = (
            config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
        )
        config["MINIBATCH_SIZE"] = (
            config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
        )
        config["CLIP_EPS"] = config["CLIP_EPS"] / env.num_agents if config["SCALE_CLIP_EPS"] else config["CLIP_EPS"]

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        # ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["HIDDEN_SIZE"])
        # cr_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["HIDDEN_SIZE"])

        actor_network = ActorRNN(env.action_space(env.agents[0]).n, config=config)

        # greedy test
        def _get_greedy_metrics(rng, actor_params):
            """
            Tests greedy policy in test env (which may have different teams).
            """
            # define a step in test_env, then lax.scan over it to rollout the greedy policy in the env, gather viz_env_states
            def _greedy_env_step(step_state, unused):
                actor_params, env_state, last_obs, last_done, ac_hstate, rng = step_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                obs_batch = batchify(last_obs, env.agents, config["NUM_TEST_ACTORS"])
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                )
                ac_hstate, pi = actor_network.apply(actor_params, ac_hstate, ac_in)
                # here, instead of sampling from distribution, take mode
                action = pi.mode()
                env_act = unbatchify(
                    action, env.agents, config["NUM_TEST_ENVS"], env.num_agents
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_TEST_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    test_env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)

                info = jax.tree_map(lambda x: x.reshape((config["NUM_TEST_ACTORS"])), info)

                done_batch = batchify(done, env.agents, config["NUM_TEST_ACTORS"]).squeeze()
                reward_batch = batchify(reward, env.agents, config["NUM_TEST_ACTORS"]).squeeze()

                step_state = (actor_params, env_state, obsv, done_batch, ac_hstate, rng)
                return step_state, (reward_batch, done_batch, info, env_state.env_state, obs_batch, ac_hstate)

            # reset test env
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config["NUM_TEST_ENVS"])
            init_obsv, env_state = jax.vmap(test_env.reset, in_axes=(0,))(reset_rng)
            init_dones = jnp.zeros((config["NUM_TEST_ACTORS"]), dtype=bool)
            ac_hstate = ScannedRNN.initialize_carry(config["NUM_TEST_ACTORS"], config["HIDDEN_SIZE"])
            rng, _rng = jax.random.split(rng)

            step_state = (actor_params, env_state, init_obsv, init_dones, ac_hstate, _rng)
            step_state, (rewards, dones, infos, viz_env_states, obs, hstate) = jax.lax.scan(
                _greedy_env_step, step_state, None, config["NUM_STEPS"]
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
        test_metrics = _get_greedy_metrics(_rng, params)

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
        env = MPEWorldStateWrapper(env)
        env = LogWrapper(env)
    return env, env_name


def single_run(config):

    config = {**config, **config["alg"]}  # merge the alg config with the main config
    config["ENV_KWARGS"]["eval"] = True
    # print("Config:\n", OmegaConf.to_yaml(config))

    alg_name = config.get("ALG_NAME", "mappo")
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
