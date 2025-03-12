import os
import json
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import yaml
import flax.linen as nn


from jaxmarl.environments.marbler.scenarios.predator_capture_prey import PredatorCapturePrey
from jaxmarl.environments.marbler.robotarium_env import State
from jaxmarl.wrappers.baselines import load_params
from flax.linen.initializers import constant, orthogonal
from functools import partial

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
        hidden_size = ins.shape[-1]
        rnn_state = jnp.where(
            resets[:, jnp.newaxis],
            self.initialize_carry(hidden_size, *ins.shape[:-1]),
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

class RNNQNetwork(nn.Module):
    # homogenous agent for parameters sharing, assumes all agents have same obs and action dim
    action_dim: int
    hidden_dim: int
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, hidden, obs, dones):
        embedding = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        q_vals = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(embedding)

        return hidden, q_vals

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d
        self.__json__ = json.dumps(d, indent=4)

# load config
module_dir = os.path.dirname(__file__)
config_path = os.path.join(module_dir, 'config_jax.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
config = objectview(config)

# load actor
params = load_params(config.model_weights)['agent']
network = RNNQNetwork(
    action_dim=config.output_dim,
    hidden_dim=config.hidden_dim,
)

# load scenario
env = PredatorCapturePrey(**config.__dict__)
num_agents = config.num_agents
max_steps = config.max_steps

# rollout
_, state = env.reset(jax.random.PRNGKey(0))
initial_state = state.replace(
    p_pos=jnp.array(config.initial_conditions),
    done=jnp.full((num_agents), False),
    step=0,
)
initial_obs = env.get_obs(initial_state)

def batchify(x: dict):
        return jnp.stack([x[agent] for agent in env.agents], axis=0)

def unbatchify(x: jnp.ndarray):
    return {agent: x[i] for i, agent in enumerate(env.agents)}

def _greedy_env_step(step_state, unused):
    params, env_state, obs, last_dones, hstate = step_state
    _obs = batchify(obs)[:, jnp.newaxis]
    _dones = batchify(last_dones)[:, jnp.newaxis]
    hstate, q_vals = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
        params,
        hstate,
        _obs,
        _dones,
    )
    q_vals = q_vals.squeeze(axis=1)
    actions = jnp.argmax(q_vals, axis=-1)
    obs, state, reward, dones, info = env.step_env(None, env_state, unbatchify(actions))

    step_state = (params, state, obs, last_dones, hstate)

    return step_state, state

hstate = ScannedRNN.initialize_carry(
    config.hidden_dim, len(env.agents), 1
)  # (n_agents*n_envs, hs_size)
init_dones = {
    agent: jnp.zeros((1), dtype=bool)
    for agent in env.agents + ["__all__"]
}
step_state = (
    params,
    initial_state,
    initial_obs,
    init_dones,
    hstate,
)
step_state, batch = jax.lax.scan(
    _greedy_env_step, step_state, None, config.max_steps
)
# hack to add extra dims
render_batch = State()
fields = {}
for attr in batch.__dict__.keys():
    if getattr(batch, attr) is None:
        continue
    fields[f'{attr}'] = getattr(batch, attr)[None, :, None, ...]
render_batch = render_batch.replace(**fields)
frames = env.render(render_batch, seed_index=0, env_index=0)
frames[0].save(
    'pcp.gif',
    save_all=True,
    append_images=frames[1:],
    duration=100,
    loop=0
)