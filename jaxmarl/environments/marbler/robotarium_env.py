"""
Base environment for robotarium simulator
"""

import jax
import jax.numpy as jnp
import chex
from flax import struct
from typing import Tuple, Optional, Dict

from jaxmarl.environments.marbler.constants import *
from rps_jax.robotarium import *
from rps_jax.robotarium_abc import *
from rps_jax.utilities.controllers import *
from rps_jax.utilities.barrier_certificates2 import *
from rps_jax.utilities.misc import *

@struct.dataclass
class State:
    p_pos: chex.Array
    done: chex.Array
    step: int

class Controller:
    def __init__(
        self,
        controller,
        barrier_fn,
        **kwargs
    ):
        """
        Initialize wrapper class for handling calling controllers and barrier functions.

        Args:
            controller: (str) name of controller, supported controllers defined in constants.py
            barrier_fn: (str) name of barrier fn, supported barrier functions defined in constants.py 
        """
        if controller not in CONTROLLERS:
            raise ValueError(f'{controller} not in supported controllers, {CONTROLLERS}')
        elif controller == 'si_position':
            controller = create_si_position_controller(**kwargs.get('controller_args', {}))
        elif controller == 'clf_uni_position':
            controller = create_clf_unicycle_position_controller(**kwargs.get('controller_args', {}))
        elif controller == 'clf_uni_pose':
            controller = create_clf_unicycle_pose_controller(**kwargs.get('controller_args', {}))

        
        if barrier_fn not in BARRIERS:
            raise ValueError(f'{controller} not in supported controllers, {CONTROLLERS}')
        elif barrier_fn == 'robust_barriers':
            barrier_fn = create_robust_barriers(**kwargs.get('barrier_args', {}))

        self.controller = controller
        self.barrier_fn = barrier_fn
    
    def get_action(self, x, g):
        """
        Applies controller and barrier function to get action
        
        Args:
            x: (jnp.ndarray) 3xN states (x, y, theta)
            g: (jnp.ndarray) 2xN (x, y) positions or 3xN poses (x, y, theta)
        
        Returns:
            (jnp.ndarray) 2xN unicycle controls (linear velocity, angular velocity)
        """
        dxu = self.controller(x, g)
        dxu_safe = self.barrier_fn(dxu, x, [])

class RobotariumEnv:
    def __init__(
        self,
        num_agents: int,
        max_steps=MAX_STEPS,
        **kwargs
    ) -> None:
        """
        Initialize robotarium environment

        Args:
            num_agents (int): maximum number of agents within the environment, used to set array dimensions
        """
        self.num_agents = num_agents
        self.agents = [f"agent_{i}" for i in range(num_agents)]
        self.observation_spaces = dict()
        self.action_spaces = dict()
        self.max_steps = max_steps

        self.robotarium = Robotarium(**kwargs.get('robotarium', {'number_of_robots': num_agents}))
        self.controller = Controller(**kwargs.get('controller', None)) if 'controller' in kwargs else None

    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        """
        Performs resetting of the environment.
        
        Args:
            key: (chex.PRNGKey)
        
        Returns:
            (Tuple[Dict[str, check.Array], State]) initial observation and environment state
        """

        # randomly generate initial poses for robots
        self.robotarium.poses = generate_initial_conditions(
            key,
            self.num_agents,
            width=ROBOTARIUM_WIDTH,
            height=ROBOTARIUM_HEIGHT,
        )

        # set velocities to 0
        self.robotarium.set_velocities(jnp.arange(self.num_agents), jnp.zeros((2, self.num_agents)))

        state = State(
            p_pos=self.robotarium.get_poses().T,
            done=jnp.full((self.num_agents), False),
            step=0,
        )

        return self.get_obs(state), state

    def step(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Dict[str, chex.Array],
        reset_state: Optional[State] = None,
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Performs step transitions in the environment. Resets the environment if done.
        To control the reset state, pass `reset_state`. Otherwise, the environment will reset randomly."""

        key, key_reset = jax.random.split(key)
        obs_st, states_st, rewards, dones, infos = self.step_env(key, state, actions)

        if reset_state is None:
            obs_re, states_re = self.reset(key_reset)
        else:
            states_re = reset_state
            obs_re = self.get_obs(states_re)

        # Auto-reset environment based on termination
        states = jax.tree.map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), states_re, states_st
        )
        obs = jax.tree.map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), obs_re, obs_st
        )
        return obs, states, rewards, dones, infos

    def step_env(
        self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """
        Environment-specific step transition.
        
        Args:
            key: (chex.PRNGKey)
            state: (State) environment state
            actions: (Dict) agent actions
        
        Returns:
            Tuple(
                (Dict[str, chex.Array]) new observation
                (State) new environment state
                (Dict[str, float]) agent rewards
                (Dict[str, bool]) dones
                (Dict) environment info
            )
        """

        actions = jnp.array([actions[f'agent_{i}'] for i in range(self.num_agents)]).reshape(
            (self.num_agents, -1)
        ) 
        poses = state.p_pos.T
        dxu = actions.T

        # if controller exists, convert actions to control inputs
        if self.controller:
            dxu = self.controller(poses.T, actions.T)   # actions interpreted as goals for controller

        # update pose
        updated_pose = self.robotarium.batch_step(poses, dxu)
        state = state.replace(
            p_pos=updated_pose.T,
        )

        # check for violations
        violations = self.get_violations(state)
        collision = violations['collision'] > 0
        boundary = violations['boundary'] > 0
        done = jnp.full((self.num_agents), state.step >= self.max_steps & boundary & collision)
        state = state.replace(
            done=done,
            step=state.step + 1,
        )

        reward = self.rewards(state)

        obs = self.get_obs(state)

        info = {}

        dones = {a: done[i] for i, a in enumerate(self.agents)}
        dones.update({"__all__": jnp.all(done)})

        return obs, state, reward, dones, info
    
    def rewards(self, state: State) -> Dict[str, float]:
        """
        Assigns rewards, trivially returns 0 here.
        
        Args:
            state: (State) environment state
        
        Returns:
            (Dict[str, float]) agent rewards
        """

        return {agent: 0 for _, agent in enumerate(self.agents)}

    def get_violations(self, state: State) -> Dict[str, float]:
        b = self.robotarium.boundaries
        p = state.p_pos[:self.num_agents, :].T
        N = self.num_agents

        # Check boundary conditions
        x_out_of_bounds = (p[0, :] < b[0]) | (p[0, :] > (b[0] + b[2]))
        y_out_of_bounds = (p[1, :] < b[1]) | (p[1, :] > (b[1] + b[3]))
        boundary_violations = jnp.where(x_out_of_bounds | y_out_of_bounds, 1, 0)
        boundary_violations = jnp.sum(boundary_violations)

        # Pairwise distance computation for collision checking
        distances = jnp.sqrt(jnp.sum((p[:2, :, None] - p[:2, None, :])**2, axis=0))
        
        collision_matrix = distances < self.robotarium.collision_diameter
        collision_violations = (jnp.sum(collision_matrix) - N) // 2 # Subtract N to remove self-collisions, divide by 2 for symmetry

        return {'collision': collision_violations, 'boundary': boundary_violations}

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """
        Applies observation function to state.

        Args:
            state: (State) environment state
        
        Returns:
            (Dict[str, float]) agent observations
        """

        return {a: state.p_pos[i] for i, a in enumerate(self.agents)}

    def observation_space(self, agent: str):
        """Observation space for a given agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        """Action space for a given agent."""
        return self.action_spaces[agent]
