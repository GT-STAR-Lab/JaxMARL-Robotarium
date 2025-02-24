import unittest
import jax
import jax.numpy as jnp

from jaxmarl.environments.marbler.scenarios.navigation import Navigation

from rps_jax.utilities.barrier_certificates2 import create_robust_barriers
from rps_jax.utilities.controllers import create_clf_unicycle_position_controller

VISUALIZE = True

class TestNavigation(unittest.TestCase):
    """unit tests for navigation.py"""

    def setUp(self):
        self.num_agents = 3
        self.batch_size = 10
        self.env = Navigation(num_agents=self.num_agents, action_type="Continuous", max_steps=250, update_frequency=1)
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
        boundary_p_pos = jnp.array([[-1., 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        state = state.replace(
            p_pos = boundary_p_pos
        )
        actions = {str(f'agent_{i}'): jnp.array([1, 0.0]) for i in range(self.num_agents)}
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)
        
        for i in range(self.num_agents):
            self.assertFalse(dones[f'agent_{i}'])
    
    def test_reward(self):
        _, state = self.env.reset(self.key)
        p_pos = jnp.array([[1, 0, 0], [-1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        state = state.replace(
            p_pos = p_pos
        )
        rewards = self.env.rewards(state)
        self.assertEqual(rewards['agent_0'], 1 * self.env.pos_shaping)
        self.assertEqual(rewards['agent_1'], 1 * self.env.pos_shaping)
        self.assertEqual(rewards['agent_2'], 0)
    
    def test_get_obs(self):
        _, state = self.env.reset(self.key)
        p_pos = jnp.array([[1, 0, 0], [-1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        state = state.replace(
            p_pos = p_pos
        )
        observations = self.env.get_obs(state)
        self.assertEqual(len(observations), self.num_agents)
        for i in range(self.num_agents):
            self.assertTrue(
                jnp.array_equal(
                    observations[str(f'agent_{i}')], 
                    jnp.concatenate((state.p_pos[i], state.p_pos[self.num_agents+i, :2]-state.p_pos[i, :2]))
                )
            )
    
    def test_batched_rollout(self):
        self.env = Navigation(
            num_agents=2,
            action_type="Discrete",
            max_steps=75,
            update_frequency=30,
            controller={
                "controller": "clf_uni_position",
                "barrier_fn": "robust_barriers",
            }
        )
        keys = jax.random.split(self.key, self.batch_size)
        _, state = jax.vmap(self.env.reset, in_axes=0)(keys)
        initial_state = state

        def get_action(state):
            goal_pos = state.p_pos[self.env.num_agents:, :2]
            actions = jnp.array([[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]])
            dir_to_goal = goal_pos - state.p_pos[:self.env.num_agents, :2]
            dir_to_goal = dir_to_goal / jnp.linalg.norm(dir_to_goal, axis=1)[:, None]
            dots = jax.vmap(jnp.dot, in_axes=(None, 0))(actions, dir_to_goal)
            best_action = jnp.argmax(dots, axis=1)
            return {str(f'agent_{i}'): best_action[i] for i in range(self.env.num_agents)}
        
        def wrapped_step(poses, unused):
            actions = jax.vmap(get_action, in_axes=(0))(poses)
            new_obs, new_state, rewards, dones, infos = jax.vmap(self.env.step, in_axes=(0, 0, 0))(keys, poses, actions)
            return new_state, (new_state, rewards)

        final_state, (batch, rewards) = jax.lax.scan(wrapped_step, state, None, 75)

        rewards = jnp.array([rewards[agent] for agent in rewards])
        print(rewards.shape)
        print(jnp.sum(rewards) / 10)
        
        # check that the robot moved
        for i in range(self.num_agents):
            self.assertGreater(
                jnp.sqrt(jnp.sum((final_state.p_pos.T[i][0] - initial_state.p_pos.T[i][0])**2)),
                0
            )
        
        if VISUALIZE:
            self.env.render(batch.p_pos[:, 1, ...], name='swap env 0', save_path='jaxmarl/environments/marbler/scenarios/test/swap.gif')
        

if __name__ == '__main__':
    unittest.main()