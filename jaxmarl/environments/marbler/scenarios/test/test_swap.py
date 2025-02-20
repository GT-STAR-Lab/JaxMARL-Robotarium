import unittest
import jax
import jax.numpy as jnp

from jaxmarl.environments.marbler.scenarios.swap import Swap

from rps_jax.utilities.barrier_certificates2 import create_robust_barriers
from rps_jax.utilities.controllers import create_clf_unicycle_position_controller

VISUALIZE = True

class TestSwap(unittest.TestCase):
    """unit tests for robotarium_env.py"""

    def setUp(self):
        self.num_agents = 3
        self.batch_size = 10
        self.env = Swap(num_agents=self.num_agents, action_type="Continuous", max_steps=1000)
        self.key = jax.random.PRNGKey(0)

    def test_step_collision(self):
        _, state = self.env.reset(self.key)

        # positions that will lead to collision violation
        collision_p_pos = jnp.array([[1, 0, jnp.pi], [0.87, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        state = state.replace(
            p_pos = collision_p_pos
        )
        actions = {str(f'agent_{i}'): jnp.array([1, 0.0]) for i in range(self.num_agents)}
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)
        
        for i in range(self.num_agents):
            self.assertEqual(rewards[f'agent_{i}'], self.env.violation_shaping)
            self.assertTrue(dones[f'agent_{i}'])

    def test_step_boundary(self):
        _, state = self.env.reset(self.key)

        # positions that will lead to boundary violation
        boundary_p_pos = jnp.array([[1.59, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        state = state.replace(
            p_pos = boundary_p_pos
        )
        actions = {str(f'agent_{i}'): jnp.array([1, 0.0]) for i in range(self.num_agents)}
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)
        
        for i in range(self.num_agents):
            self.assertEqual(rewards[f'agent_{i}'], self.env.violation_shaping)
            self.assertTrue(dones[f'agent_{i}'])
    
    def test_step_no_violation(self):
        _, state = self.env.reset(self.key)

        # positions that will lead to boundary violation
        boundary_p_pos = jnp.array([[-1, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        state = state.replace(
            p_pos = boundary_p_pos
        )
        actions = {str(f'agent_{i}'): jnp.array([1, 0.0]) for i in range(self.num_agents)}
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)
        
        for i in range(self.num_agents):
            self.assertFalse(dones[f'agent_{i}'])
    
    def test_batched_rollout(self):
        keys = jax.random.split(self.key, self.batch_size)
        _, state = jax.vmap(self.env.reset, in_axes=0)(keys)
        initial_state = state
        barrier_fn = create_robust_barriers(safety_radius=0.2)
        controller = create_clf_unicycle_position_controller()

        def get_action(state):
            agent_pos = state.p_pos[:self.num_agents, :].T
            goal_pos = state.p_pos[self.num_agents:, :2].T
            dxu = controller(agent_pos, goal_pos)
            dxu_safe = barrier_fn(dxu, agent_pos, [])
            dxu_safe = dxu_safe.T
            return {str(f'agent_{i}'): dxu_safe[i] for i in range(self.num_agents)}
        
        def wrapped_step(poses, unused):
            actions = jax.vmap(get_action, in_axes=(0))(poses)
            new_obs, new_state, rewards, dones, infos = jax.vmap(self.env.step, in_axes=(0, 0, 0))(keys, poses, actions)
            return new_state, new_state

        final_state, batch = jax.lax.scan(wrapped_step, state, None, 250)
        
        # check that the robot moved
        for i in range(self.num_agents):
            self.assertGreater(
                jnp.sqrt(jnp.sum((final_state.p_pos.T[i][0] - initial_state.p_pos.T[i][0])**2)),
                0
            )
        
        if VISUALIZE:
            self.env.render(batch.p_pos[:, 0, ...], name='swap env 0', save_path='jaxmarl/environments/marbler/scenarios/test/swap.gif')
        

if __name__ == '__main__':
    unittest.main()