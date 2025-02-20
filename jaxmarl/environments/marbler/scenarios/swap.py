"""
Simple scenario where robots must swap positions.
"""

from jaxmarl.environments.marbler.robotarium_env import *

class Swap(RobotariumEnv):

    def __init__(self, num_agents, max_steps=50, **kwargs):
        super().__init__(num_agents, max_steps, **kwargs)
        self.name = 'MARBLER_swap'

        self.pos_shaping = kwargs.get('pos_shaping', -0.01)
        self.violation_shaping = kwargs.get('violation_shaping', -10)
        self.goal_radius = kwargs.get('goal_radius', 0.1)

        # Observation space
        self.obs_dim = 5
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

        # set goals to other agent indices
        goals = jnp.vstack([self.robotarium.poses.T[1:], self.robotarium.poses.T[0]])

        # set velocities to 0
        self.robotarium.set_velocities(jnp.arange(self.num_agents), jnp.zeros((2, self.num_agents)))

        state = State(
            p_pos=jnp.vstack([self.robotarium.get_poses().T, goals]),
            done=jnp.full((self.num_agents), False),
            step=0,
        )

        return self.get_obs(state), state

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
        poses = state.p_pos.T[:, :self.num_agents]

        # if controller exists, convert actions to control inputs
        if self.controller:
            dxu = self.controller.get_action(poses, actions.T)   # actions interpreted as goals for controller
        else:
            dxu = actions.T

        # update pose
        updated_pose = self.robotarium.batch_step(poses, dxu)
        state = state.replace(
            p_pos=jnp.vstack([updated_pose.T, state.p_pos[self.num_agents:, :]]),
        )

        # check for violations
        violations = self.get_violations(state)
        collision = violations['collision'] > 0
        boundary = violations['boundary'] > 0
        done = jnp.full((self.num_agents), ((state.step >= self.max_steps) | boundary | collision))
        state = state.replace(
            done=done,
            step=state.step + 1,
        )

        reward = self.rewards(state)

        obs = self.get_obs(state)

        info = {
            'collision': jnp.full((self.num_agents,), violations['collision']),
            'boundary': jnp.full((self.num_agents,), violations['boundary'])
        }

        dones = {a: done[i] for i, a in enumerate(self.agents)}
        dones.update({"__all__": jnp.all(done)})

        return obs, state, reward, dones, info
    
    def rewards(self, state: State) -> Dict[str, float]:
        """
        Assigns rewards, (shaping reward of distance to goal +1 for reaching goal).
        
        Args:
            state: (State) environment state
        
        Returns:
            (Dict[str, float]) agent rewards
        """

        # agent specific shaping reward
        goals = state.p_pos[self.num_agents:, :2]
        agent_pos = state.p_pos[:self.num_agents, :2]
        d_goal = jnp.linalg.norm(agent_pos - goals, axis=1)
        pos_rew = d_goal * self.pos_shaping

        # global penalty for collisions and boundary violation
        violations = self.get_violations(state)
        collisions = violations['collision']
        boundaries = violations['boundary']
        violation_rew = self.violation_shaping * (collisions + boundaries)

        # final reward
        on_goal = d_goal < self.goal_radius
        final_rew = jnp.where(jnp.sum(on_goal) < self.num_agents, 0, 1)

        return {agent: jnp.where(violation_rew == 0, pos_rew[i] + final_rew, violation_rew) for i, agent in enumerate(self.agents)}

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """
        Get observation (pos, vector to goal)

        Args:
            state: (State) environment state
        
        Returns:
            (Dict[str, float]) agent observations
        """

        goals = state.p_pos[self.num_agents:, :2]
        agent_pos = state.p_pos[:self.num_agents, :2]
        to_goal = goals - agent_pos

        return {a: jnp.concatenate([state.p_pos[i], to_goal[i]]) for i, a in enumerate(self.agents)}