"""
Base robotarium_env implementation compatible with non-Jax Robotarium Python Simulator.
"""

from dataclasses import dataclass
import numpy as jnp
from typing import Tuple, Optional, Dict

from constants import *

from rps.robotarium import *
from rps.robotarium_abc import *
from rps.utilities.controllers import *
from rps.utilities.barrier_certificates2 import *
from rps.utilities.misc import *

@dataclass
class State:
    p_pos: jnp.ndarray
    done: jnp.ndarray
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
            controller = lambda x, g: g
        elif controller not in CONTROLLERS:
            raise ValueError(f'{controller} not in supported controllers, {CONTROLLERS}')
        elif controller == 'si_position':
            controller = create_si_position_controller(**kwargs.get('controller_args', {}))
        elif controller == 'clf_uni_position':
            controller = create_clf_unicycle_position_controller(**kwargs.get('controller_args', {}))
        elif controller == 'clf_uni_pose':
            controller = create_clf_unicycle_pose_controller(**kwargs.get('controller_args', {}))

        if barrier_fn is None:
            barrier_fn = lambda dxu, x, unused: dxu
        elif barrier_fn not in BARRIERS:
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
        robotarium_args = {
            'number_of_robots': num_agents,
            'show_figure': True,
            'sim_in_real_time': True,
            'initial_conditions': kwargs.get('initial_conditions')
        }
        self.robotarium = Robotarium(**robotarium_args)
        self.controller = Controller(kwargs.get('controller', None), kwargs.get('barrier_fn', None))
        self.step_dist = kwargs.get('step_dist', 0.2)
        self.update_frequency = kwargs.get('update_frequency', 10)

        # Action type
        self.action_dim = 5
        if action_type == DISCRETE_ACT:
            self.action_decoder = self._decode_discrete_action
        elif action_type == CONTINUOUS_ACT:
            self.action_decoder = self._decode_continuous_action

    def reset(self) -> Tuple[Dict[str, np.ndarray], State]:
        """
        Performs resetting of the environment.
        
        Args:
            key: (chex.PRNGKey)
        
        Returns:
            (Tuple[Dict[str, check.Array], State]) initial observation and environment state
        """

        raise NotImplementedError

    def step_env(
        self, state: State, actions: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], State, Dict[str, float], Dict[str, bool], Dict]:
        """
        Environment-specific step transition.
        
        Args:
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

        raise NotImplementedError
    
    def rewards(self, state: State) -> Dict[str, float]:
        """
        Assigns rewards, trivially returns 0 here.
        
        Args:
            state: (State) environment state
        
        Returns:
            (Dict[str, float]) agent rewards
        """

        raise NotImplementedError
    
    def get_obs(self, state: State) -> Dict[str, np.ndarray]:
        """
        Applies observation function to state.

        Args:
            state: (State) environment state
        
        Returns:
            (Dict[str, float]) agent observations
        """

        raise NotImplementedError

    def _get_violations(self, state: State) -> Dict[str, float]:
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
        in_goals = jnp.clip(candidate_goals, b[:2] + 0.1, b[:2] + b[2:] - 0.1)

        return in_goals

    def _decode_continuous_action(self, a_idx: int, action: np.ndarray, state: State):
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
            poses: (jnp.ndarray) Nx3 array of robot poses
            actions: (jnp.ndarray) Nx2 array of robot actions
            update_frequency: (int) number of times to step robotarium simulator
        
        Returns:
            (jnp.ndarray) final poses after update_frequency steps
        """
        poses = poses.T
        goals = goals.T
        dxu = self.controller.get_action(poses, goals) 
        for i in range(self.update_frequency):
            dxu = self.controller.get_action(poses, goals)
            self.robotarium.set_velocities(jnp.arange(self.num_agents), dxu)
            self.robotarium.step()
            poses = self.robotarium.get_poses()
        final_pose = poses.T

        return final_pose

    #-----------------------------------------
    # Deployment Specific Functions
    #-----------------------------------------
    def initial_robotarium_state(self, seed: int = 0):
        """
        Sets initial conditions for robotarium

        Args:
            seed: (int) seed for random functions
        
        Returns:
            (State) initial state
        """

        raise NotImplementedError
    
    def render(self):
        """
        Visualization for robotarium
        """

        raise NotImplementedError