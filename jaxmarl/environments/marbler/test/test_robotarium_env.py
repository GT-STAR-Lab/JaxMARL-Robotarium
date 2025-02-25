import unittest
import jax
import jax.numpy as jnp
from jaxmarl.environments.marbler.robotarium_env import RobotariumEnv, State

class TestRobotariumEnv(unittest.TestCase):
    """unit tests for robotarium_env.py"""

    def setUp(self):
        self.num_agents = 3
        self.batch_size = 10
        self.env = RobotariumEnv(num_agents=self.num_agents, action_type="Continuous")
        self.key = jax.random.PRNGKey(0)

    def test_reset(self):
        obs, state = self.env.reset(self.key)
        self.assertEqual(len(obs), self.num_agents)
        self.assertIsInstance(state, State)
        self.assertEqual(state.p_pos.shape, (self.num_agents, 3))
        self.assertFalse(jnp.any(state.done))
        self.assertEqual(state.step, 0)

    def test_step(self):
        _, state = self.env.reset(self.key)
        actions = {str(f'agent_{i}'): jnp.array([0.0, 0.0]) for i in range(self.num_agents)}
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)
        self.assertEqual(len(new_obs), self.num_agents)
        self.assertIsInstance(new_state, State)
        self.assertEqual(new_state.p_pos.shape, (self.num_agents, 3))
        self.assertEqual(len(rewards), self.num_agents)
        self.assertEqual(len(dones), self.num_agents + 1)  # including "__all__"
        self.assertIsInstance(infos, dict)

    def test_rewards(self):
        _, state = self.env.reset(self.key)
        rewards = self.env.rewards(state)
        self.assertEqual(len(rewards), self.num_agents)
        self.assertTrue(all(reward == 0 for reward in rewards.values()))
    
    def test_get_violations(self):
        _, state = self.env.reset(self.key)
        state = state.replace(
            p_pos = jnp.ones_like(state.p_pos)
        )
        violations = self.env.get_violations(state)
        self.assertEqual(violations['collision'], 3)

    def test_get_obs(self):
        _, state = self.env.reset(self.key)
        observations = self.env.get_obs(state)
        self.assertEqual(len(observations), self.num_agents)
        for i in range(self.num_agents):
            self.assertTrue(jnp.array_equal(observations[str(f'agent_{i}')], state.p_pos[i]))

    def test_observation_space(self):
        self.env.observation_spaces = {str(i): "obs_space" for i in range(self.num_agents)}
        for i in range(self.num_agents):
            self.assertEqual(self.env.observation_space(str(i)), "obs_space")

    def test_action_space(self):
        self.env.action_spaces = {str(i): "act_space" for i in range(self.num_agents)}
        for i in range(self.num_agents):
            self.assertEqual(self.env.action_space(str(i)), "act_space")
    
    def test_action_decoder(self):
        _, state = self.env.reset(self.key)

        state = state.replace(
            p_pos = jnp.array([[0.0, 0.9, 0.0], [0.0, -0.9, 0.0], [1.5, 0.0, 0.0]])
        )
        decoded_actions = [self.env._decode_discrete_action(i, i+1, state) for i in range(self.num_agents)]
        for i in range(self.num_agents):
            self.assertTrue(decoded_actions[i][0] >= -1.6 and decoded_actions[i][0] <= 1.6)
            self.assertTrue(decoded_actions[i][1] >= -1 and decoded_actions[i][1] <= 1)
    
    def test_batched_rollout(self):
        keys = jax.random.split(self.key, self.batch_size)
        _, state = jax.vmap(self.env.reset, in_axes=0)(keys)
        initial_state = state

        def get_action(poses):
            return {str(f'agent_{i}'): jnp.array([0.5, 0.0]) for i in range(self.num_agents)}
        
        def wrapped_step(poses, unused):
            actions = jax.vmap(get_action, in_axes=(0))(poses)
            new_obs, new_state, rewards, dones, infos = jax.vmap(self.env.step, in_axes=(0, 0, 0))(keys, poses, actions)
            return new_state, new_state

        final_state, batch = jax.lax.scan(wrapped_step, state, None, 100)
        
        # check that the robot moved
        for i in range(self.num_agents):
            self.assertGreater(
                jnp.sqrt(jnp.sum((final_state.p_pos.T[i][0] - initial_state.p_pos.T[i][0])**2)),
                0
            )

        

if __name__ == '__main__':
    unittest.main()