"""
Simple scenario where robots must swap positions.
"""

from jaxmarl.environments.marbler.robotarium_env import *

class Navigation(RobotariumEnv):

    def __init__(self, num_agents, max_steps=50, **kwargs):
        super().__init__(num_agents, max_steps, **kwargs)
        self.name = 'MARBLER_navigation'

        self.pos_shaping = kwargs.get('pos_shaping', -1)
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
        poses = generate_initial_conditions(
            key,
            2*self.num_agents,
            width=ROBOTARIUM_WIDTH,
            height=ROBOTARIUM_HEIGHT,
            spacing=0.5
        )
        self.robotarium.poses = poses[:, :self.num_agents]

        # set velocities to 0
        self.robotarium.set_velocities(jnp.arange(self.num_agents), jnp.zeros((2, self.num_agents)))

        state = State(
            p_pos=poses.T,
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
        poses = state.p_pos[:self.num_agents, :]

        # update pose
        updated_pose = self._robotarium_step(poses, actions)
        state = state.replace(
            p_pos=jnp.vstack([updated_pose.T, state.p_pos[self.num_agents:, :]]),
        )

        # check for violations
        violations = self.get_violations(state)
        collision = violations['collision'] > 0
        boundary = violations['boundary'] > 0
        # done = jnp.full((self.num_agents), ((state.step >= self.max_steps-1) | boundary | collision))
        done = jnp.full((self.num_agents), state.step >= self.max_steps)
        state = state.replace(
            done=done,
            step=state.step + 1,
        )

        reward = self.rewards(state)

        obs = self.get_obs(state)

        # check if agents reached goal
        goals = state.p_pos[self.num_agents:, :2]
        agent_pos = state.p_pos[:self.num_agents, :2]
        d_goal = jnp.linalg.norm(agent_pos - goals, axis=1)
        on_goal = d_goal < self.goal_radius

        info = {
            'collision': jnp.full((self.num_agents,), violations['collision']),
            'boundary': jnp.full((self.num_agents,), violations['boundary']),
            'success_rate': jnp.full(
                (self.num_agents,),
                jnp.where(jnp.sum(on_goal) < self.num_agents, 0, 1)
            )
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
        # final_rew = jnp.where(jnp.sum(on_goal) < self.num_agents, 0, 10)

        # return {agent: jnp.where(violation_rew == 0, pos_rew[i] + final_rew, violation_rew) for i, agent in enumerate(self.agents)}
        return {agent: pos_rew[i] for i, agent in enumerate(self.agents)}

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
    
    def render(self, pos, name='env', save_path=None):
        import numpy as np
        import matplotlib.pyplot as plt
        from PIL import Image
        import imageio 

        colors = plt.get_cmap('viridis', self.num_agents)
        x_positions = []
        for i in range(self.num_agents):
            x_positions.append(np.array(pos[:, i, 0]))
        y_positions = []
        for i in range(self.num_agents):
            y_positions.append(np.array(pos[:, i, 1]))

        # Setup plot
        fig, ax = plt.subplots(figsize=(6.4, 4))
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1, 1)
        ax.set_title(f'{name}')

        # Plot full trajectory with transparency (trails)
        for i in range(self.num_agents):
            ax.plot(x_positions[i], y_positions[i], color=colors(i), alpha=0.3)
        
        # Initialize goal markers
        for i in range(self.num_agents):
            idx = i + self.num_agents
            ax.plot(np.array(pos[0, idx, 0]), np.array(pos[0, idx, 1]), 'o', markersize=5, color=colors(i))

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