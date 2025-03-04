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
    
    def test_get_violations(self):
        state = State(
            p_pos = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            done = jnp.full((self.num_agents,), False),
            step = 0
        )
        violations = self.env._get_violations(state)
        self.assertEqual(violations['collision'], 3)

    def test_observation_space(self):
        self.env.observation_spaces = {str(i): "obs_space" for i in range(self.num_agents)}
        for i in range(self.num_agents):
            self.assertEqual(self.env.observation_space(str(i)), "obs_space")

    def test_action_space(self):
        self.env.action_spaces = {str(i): "act_space" for i in range(self.num_agents)}
        for i in range(self.num_agents):
            self.assertEqual(self.env.action_space(str(i)), "act_space")
    
    def test_discrete_action_decoder(self):
        state = State(
            p_pos = jnp.array([[0.0, 0.9, 0.0], [0.0, -0.9, 0.0], [1.5, 0.0, 0.0]]),
            done = jnp.full((self.num_agents,), False),
            step = 0
        )
        decoded_actions = [self.env._decode_discrete_action(i, i+1, state) for i in range(self.num_agents)]
        for i in range(self.num_agents):
            self.assertTrue(decoded_actions[i][0] >= -1.6 and decoded_actions[i][0] <= 1.6)
            self.assertTrue(decoded_actions[i][1] >= -1 and decoded_actions[i][1] <= 1)
        
        state = State(
            p_pos = jnp.array([[0.0, 1.1, 0.0], [0.0, -1.1, 0.0], [1.7, 0.0, 0.0]]),
            done = jnp.full((self.num_agents,), False),
            step = 0
        )
        decoded_actions = [self.env._decode_discrete_action(i, i+1, state) for i in range(self.num_agents)]
        for i in range(self.num_agents):
            self.assertTrue(decoded_actions[i][0] >= -1.6 and decoded_actions[i][0] <= 1.6)
            self.assertTrue(decoded_actions[i][1] >= -1 and decoded_actions[i][1] <= 1)
    
    def test_continuous_action_decoder(self):
        state = State(
            p_pos = jnp.array([[0.0, 0.9, 0.0], [0.0, -0.9, 0.0], [1.5, 0.0, 0.0]]),
            done = jnp.full((self.num_agents,), False),
            step = 0
        )
        actions = jnp.array([[0, 0], [0.1, 0.1], [0.2, 0.2]])
        decoded_actions = [self.env._decode_continuous_action(i, actions[i], state) for i in range(self.num_agents)]
        for i in range(self.num_agents):
            self.assertTrue(decoded_actions[i][0] == actions[i][0])
            self.assertTrue(decoded_actions[i][1] == actions[i][1])
    
    def test_robotarium_step(self):
        poses = jnp.array([[0., 0, 0]])
        goals = jnp.array([[1., 0]])
        self.env = RobotariumEnv(
            num_agents=self.num_agents,
            action_type="Discrete",
            controller={"controller": "clf_uni_position"},
            update_frequency=30
        )
        final_pose = self.env._robotarium_step(poses, goals)
        self.assertTrue(jnp.linalg.norm(poses - final_pose) > 0.1)
        

if __name__ == '__main__':
    unittest.main()