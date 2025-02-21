"""
Base environment for robotarium simulator
"""

import jax
import jax.numpy as jnp
import chex
from flax import struct
from typing import Tuple, Optional, Dict

from jaxmarl.environments.marbler.constants import *
from jaxmarl.environments.spaces import Box, Discrete

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
        controller = None,
        barrier_fn = None,
        **kwargs
    ):
        """
        Initialize wrapper class for handling calling controllers and barrier functions.

        Args:
            controller: (str) name of controller, supported controllers defined in constants.py
            barrier_fn: (str) name of barrier fn, supported barrier functions defined in constants.py 
        """
        if controller is None:
            # if controller is not set, return trivial pass through of actions
            self.controller = lambda x, g: g
            self.barrier_fn = lambda dxu, x, unused: dxu
            return

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
            barrier_fn = create_robust_barriers(safety_radius=SAFETY_RADIUS)

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

        return dxu_safe

class RobotariumEnv:
    def __init__(
        self,
        num_agents: int,
        max_steps=MAX_STEPS,
        action_type=DISCRETE_ACT,
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

        # Initialize robotarium and controller backends
        default_robotarium_args = {'number_of_robots': num_agents, 'show_figure': True, 'sim_in_real_time': True}
        self.robotarium = Robotarium(**kwargs.get('robotarium', default_robotarium_args))
        self.controller = Controller(**kwargs.get('controller', {}))
        self.step_dist = kwargs.get('step_dist', 0.2)
        self.update_frequency = kwargs.get('update_frequency', 10)

        # Action type
        self.action_dim = 5
        if action_type == DISCRETE_ACT:
            self.action_spaces = {i: Discrete(self.action_dim) for i in self.agents}
            self.action_decoder = self._decode_discrete_action
        elif action_type == CONTINUOUS_ACT:
            self.action_spaces = {i: Box(0.0, 1.0, (self.action_dim,)) for i in self.agents}
            self.action_decoder = self._decode_continuous_action
        
        # Observation space
        self.obs_dim = 3
        self.observation_spaces = {
            i: Box(-jnp.inf, jnp.inf, (self.obs_dim,)) for i in self.agents
        }

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

        actions = jnp.array([self.action_decoder(i, actions[f'agent_{i}'], state) for i in range(self.num_agents)]).reshape(
            (self.num_agents, -1)
        ) 
        poses = state.p_pos

        # update pose
        updated_pose = self._robotarium_step(poses, actions)
        state = state.replace(
            p_pos=updated_pose.T,
        )

        # check for violations
        violations = self.get_violations(state)
        collision = violations['collision'] > 0
        boundary = violations['boundary'] > 0
        done = jnp.full((self.num_agents), state.step >= self.max_steps | boundary | collision)
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
        """
        Checks environment for collision and boundary violations.

        Args:
            state: (State) environment state
        
        Returns
            (Dict[str, float]) collision and boundary violations
        """
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

    def _decode_discrete_action(self, a_idx: int, action: int, state: State):
        """
        Decode action index into null, up, down, left, right actions

        Args:
            a_idx (int): agent index
            action: (int) action index
            state: (State) environment state
        
        Returns:
            (chex.Array) desired (x,y) position
        """
        goals = jnp.array([[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]])
        candidate_goals = state.p_pos[a_idx,:2] + (goals[action] * self.step_dist)

        # ensure goals are in bound
        b = jnp.array(self.robotarium.boundaries)
        in_goals = jnp.clip(candidate_goals, b[:2], b[:2] + b[2:])

        return in_goals

    def _decode_continuous_action(self, a_idx: int, action: chex.Array, state: State):
        """
        Trivially returns actions, assumes directly setting v and omega

        Args:
            a_idx: (int) agent index
            action: (chex.Array) action
            state: (State) environment state
        
        Returns:
            (chex.Array) action
        """
        return action
    
    def _robotarium_step(self, poses: jnp.ndarray, goals: jnp.ndarray):
        """
        Wrapper to step robotarium simulator update_frequency times

        Args:
            poses: (jnp.ndarray) 3xN array of robot poses
            actions: (jnp.ndarray) 2xN array of robot actions
            update_frequency: (int) number of times to step robotarium simulator
        
        Returns:
            (jnp.ndarray) final poses after update_frequency steps
        """
        poses = poses.T
        goals = goals.T
        def wrapped_step(poses, unused):
            dxu = self.controller.get_action(poses, goals) 
            updated_pose = self.robotarium.batch_step(poses, dxu)
            return updated_pose, None
        final_pose, _ = jax.lax.scan(wrapped_step, poses, None, self.update_frequency)

        return final_pose

    def render(self, batch, name='env', save_path=None):
        import numpy as np
        import matplotlib.pyplot as plt
        from PIL import Image
        import imageio 

        colors = plt.get_cmap('viridis', self.num_agents)
        x_positions = []
        for i in range(self.num_agents):
            x_positions.append(np.array(batch[:, i, 0]))
        y_positions = []
        for i in range(self.num_agents):
            y_positions.append(np.array(batch[:, i, 1]))

        # Setup plot
        fig, ax = plt.subplots(figsize=(6.4, 4))
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1, 1)
        ax.set_title(f'{name}')

        # Plot full trajectory with transparency (trails)
        for i in range(self.num_agents):
            ax.plot(x_positions[i], y_positions[i], color=colors(i), alpha=0.3)

        # Initialize moving robot markers
        robots = []
        for i in range(self.num_agents):
            robot, = ax.plot([], [], 'o', markersize=10,  color=colors(i), label=f"robot {i}")
            robots.append(robot)

        ax.legend()

        # List to store frames
        frames = []

        # Generate and save each frame
        for frame in range(len(x_positions[0])):
            for i in range(self.num_agents):
                x, y = x_positions[i][frame], y_positions[i][frame]
                robots[i].set_data([x], [y])

            # Save the current frame as an image
            fig.canvas.draw()
            frame_image = np.array(fig.canvas.renderer.buffer_rgba())  # Get image from canvas
            frames.append(Image.fromarray(frame_image))  # Convert to PIL Image

        # Save frames as a GIF
        if save_path:
            frames[0].save(save_path, save_all=True, append_images=frames[1:], duration=20, loop=0)

            print(f"GIF saved at {save_path}")

        return frames