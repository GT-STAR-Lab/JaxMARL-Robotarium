import unittest
import jax
import jax.numpy as jnp

from jaxmarl.environments.marbler.robotarium_env import State
from jaxmarl.environments.marbler.scenarios.rware import RWARE

VISUALIZE = False

class TestRWARE(unittest.TestCase):
    """unit tests for test_rware.py"""

    def setUp(self):
        self.num_agents = 2
        self.batch_size = 10
        self.env = RWARE(
            num_agents=self.num_agents,
            num_cells=self.num_agents,
            action_type="Discrete",
            max_steps=70,
            update_frequency=1,
            time_shaping=0,
        )
        self.key = jax.random.PRNGKey(0)
    
    def test_reset(self):
        obs, state = self.env.reset(self.key)
        for agent, agent_obs in obs.items():
            self.assertTrue(agent_obs.shape == (self.env.obs_dim,))
        
        self.assertTrue(~jnp.all(state.done))
        self.assertTrue(jnp.array_equal(state.payload, jnp.full((self.num_agents,), -1)))
        self.assertTrue(state.p_pos.shape == (self.num_agents+self.env.num_cells, 3))
        self.assertTrue(state.grid.shape == (self.env.num_cells, 3))
        self.assertTrue(state.request.shape == (self.num_agents,))
        self.assertTrue(jnp.unique(state.request).shape[0] == self.num_agents)
        self.assertTrue(state.step == 0)
    
    def test_step(self):
        # agent 0 picks up shelf 0
        _, state = self.env.reset(self.key)
        p_pos = jnp.array([[0.25, 0.25,  0], [1, 0, 0], [0.25, 0.25,  0], [.25, -0.25, 0]])
        state = state.replace(
            p_pos = p_pos,
        )

        actions = {str(f'agent_{i}'): jnp.array([0]) for i in range(self.num_agents)}
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)

        self.assertTrue(jnp.array_equal(new_state.grid[:, 2], jnp.array([-1, 1])))
        self.assertTrue(jnp.array_equal(new_state.payload, jnp.array([0, -1])))

        # agent 0 picks up shelf 1
        _, state = self.env.reset(self.key)
        p_pos = jnp.array([[0.25, -0.25,  0], [1, 0, 0], [0.25, 0.25,  0], [.25, -0.25, 0]])
        state = state.replace(
            p_pos = p_pos,
        )

        actions = {str(f'agent_{i}'): jnp.array([0]) for i in range(self.num_agents)}
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)

        self.assertTrue(jnp.array_equal(new_state.grid[:, 2], jnp.array([0, -1])))
        self.assertTrue(jnp.array_equal(new_state.payload, jnp.array([1, -1])))

        # both agents attempt to pick up shelf 0
        _, state = self.env.reset(self.key)
        p_pos = jnp.array([[0.25, 0.25,  0], [0.25, 0.25, 0], [0.25, 0.25,  0], [.25, -0.25, 0]])
        state = state.replace(
            p_pos = p_pos,
        )

        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)

        self.assertTrue(jnp.array_equal(new_state.grid[:, 2], jnp.array([-1, 1])))
        self.assertTrue(jnp.array_equal(new_state.payload, jnp.array([0, -1])))

        # both agents attempt to pick up unique shelves
        _, state = self.env.reset(self.key)
        p_pos = jnp.array([[0.25, 0.25,  0], [0.25, -0.25, 0], [0.25, 0.25,  0], [.25, -0.25, 0]])
        state = state.replace(
            p_pos = p_pos,
        )

        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)

        self.assertTrue(jnp.array_equal(new_state.grid[:, 2], jnp.array([-1, -1])))
        self.assertTrue(jnp.array_equal(new_state.payload, jnp.array([0, 1])))

        # agent 0 drops off shelf
        state = new_state
        p_pos = jnp.array([[1.5, 0.25,  0], [0.25, 0.25, 0], [0.25, 0.25,  0], [.25, -0.25, 0]])
        state = state.replace(
            p_pos = p_pos,
        )
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)

        self.assertTrue(jnp.array_equal(new_state.grid[:, 2], jnp.array([-1, -1])))
        self.assertTrue(jnp.array_equal(new_state.request, jnp.array([0, 1]))) # same shelf back in queue

        # agent 0 returns shelf 0 at cell 1
        prev_state = new_state
        state = new_state
        p_pos = jnp.array([[0.25, -0.25,  0], [-1, 0.25, 0], [0.25, 0.25,  0], [.25, -0.25, 0]])
        state = state.replace(
            p_pos = p_pos,
        )
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)

        self.assertTrue(jnp.array_equal(new_state.grid[:, 2], jnp.array([-1, 0])))
        self.assertTrue(jnp.array_equal(new_state.payload, jnp.array([-1, 1])))

        # both agents attempt to return
        state = prev_state
        p_pos = jnp.array([[0.25, -0.25,  0], [0.25, -0.25, 0], [0.25, 0.25,  0], [.25, -0.25, 0]])
        state = state.replace(
            p_pos = p_pos,
        )
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)

        self.assertTrue(jnp.array_equal(new_state.grid[:, 2], jnp.array([-1, 0])))
        self.assertTrue(jnp.array_equal(new_state.payload, jnp.array([-1, 1])))

    def test_reward(self):
        # NOTE: this test is hacky because we unify step_env and reward in rware
        actions = {str(f'agent_{i}'): jnp.array([0]) for i in range(self.num_agents)}
        _, state = self.env.reset(self.key)
        p_pos = jnp.array([[0.25, 0.25,  0], [0.25, -0.25, 0], [0.25, 0.25,  0], [.25, -0.25, 0]])
        state = state.replace(
            p_pos = p_pos,
        )

        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)

        self.assertEqual(rewards['agent_0'], 0)
        self.assertEqual(rewards['agent_1'], 0)

        # agent 0 drops off shelf
        state = new_state
        p_pos = jnp.array([[1.5, 0.25,  0], [0.25, 0.25, 0], [0.25, 0.25,  0], [.25, -0.25, 0]])
        state = state.replace(
            p_pos = p_pos,
        )
        new_obs, new_state, rewards, dones, infos = self.env.step(self.key, state, actions)

        self.assertEqual(rewards['agent_0'], 1)
        self.assertEqual(rewards['agent_1'], 1)
    
    def test_get_obs(self):
        _, state = self.env.reset(self.key)
        p_pos = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        grid = jnp.array([[7, 8, -1], [10, 11, 1]])
        state = state.replace(
            p_pos=p_pos,
            payload=jnp.array([0, -1]),
            grid=grid,
        )

        obs = self.env.get_obs(state)
        
        # agent 0
        expected_obs = jnp.array([1, 2, 3, 4, 5, 6, 0, 7, 8, -1, 10, 11, 1])
        self.assertTrue(
            jnp.array_equal(obs['agent_0'], expected_obs)
        )

        # agent 1
        expected_obs = jnp.array([4, 5, 6, 1, 2, 3, -1, 7, 8, -1, 10, 11, 1])
        self.assertTrue(
            jnp.array_equal(obs['agent_1'], expected_obs)
        )