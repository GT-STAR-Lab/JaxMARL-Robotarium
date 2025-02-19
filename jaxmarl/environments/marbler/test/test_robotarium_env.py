import unittest
import jax
import jax.numpy as jnp
from marbler import RobotariumEnv, State

class TestRobotariumEnv(unittest.TestCase):
    """unit tests for robotarium_env.py"""

    def setUp(self):
        self.num_agents = 3
        self.env = RobotariumEnv(num_agents=self.num_agents)
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
        actions = {str(i): jnp.array([0.0, 0.0]) for i in range(self.num_agents)}
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

    def test_get_obs(self):
        _, state = self.env.reset(self.key)
        observations = self.env.get_obs(state)
        self.assertEqual(len(observations), self.num_agents)
        for i in range(self.num_agents):
            self.assertTrue(jnp.array_equal(observations[str(i)], state.p_pos.T[i]))

    def test_observation_space(self):
        self.env.observation_spaces = {str(i): "obs_space" for i in range(self.num_agents)}
        for i in range(self.num_agents):
            self.assertEqual(self.env.observation_space(str(i)), "obs_space")

    def test_action_space(self):
        self.env.action_spaces = {str(i): "act_space" for i in range(self.num_agents)}
        for i in range(self.num_agents):
            self.assertEqual(self.env.action_space(str(i)), "act_space")

if __name__ == '__main__':
    unittest.main()