"""
Robots collaborate to manage shelves in a warehouse.
"""

# wrap import statement in try-except block to allow for correct import during deployment
try:
    from jaxmarl.environments.marbler.robotarium_env import *
except Exception as e:
    from robotarium_env import *

class RWARE(RobotariumEnv):
    def __init__(self, num_agents, max_steps=70, **kwargs):
        self.name = 'MARBLER_rware'
        self.backend = kwargs.get('backend', 'jax')

        self.num_cells = kwargs.get('num_cells', 8) # NOTE: this needs to be even right now
        self.queue_length = self.num_cells * 4 # this is to simulate random requests
        self.pickup_radius = 0.2

        if self.backend == 'jax':
            super().__init__(num_agents, max_steps, **kwargs)
        else:
            self.num_agents = num_agents
            self.initial_state = self.initialize_robotarium_state(kwargs.get("seed", 0))
            kwargs['initial_conditions'] = self.initial_state.p_pos[:self.num_agents, :].T
            super().__init__(num_agents, max_steps, **kwargs)
        
        # Reward shaping
        self.dropoff_shaping = kwargs.get('dropoff_shaping', 1)
        self.violation_shaping = kwargs.get('violation_shaping', 0)

        # Observation space (poses of all agents, payload, poses of all storage locations, requests)
        self.obs_dim = (3*self.num_agents) + 1 + (3*self.num_cells)

        if self.backend == 'jax':
            self.observation_spaces = {
                i: Box(-jnp.inf, jnp.inf, (self.obs_dim,)) for i in self.agents
            }
        
        # Zone dimensions
        self.zone_width = 0.5
        
        # Visualization
        self.robot_markers = []
        self.storage_markers = []
        self.shelf_markers = []
        self.shelf_labels = []
        self.storage_labels = []
        self.request_label = None
        self.dropoff_marker = None
    
    def reset(self, key) -> Tuple[Dict, State]:
        """
        Performs resetting of the environment.

        Args:
            key: (check.PRNGKey)
        
        Returns:
            (Tuple[Dict[str, chex.Array], State]) initial observation and environment state
        """

        # set agents to always start lined up on bottom of env
        bounds = self.robotarium.boundaries # lower left point / width/ height
        x_poses = jnp.linspace(bounds[0] + 0.5, bounds[0] + bounds[2] - 0.5, self.num_agents)
        y_poses = jnp.full((self.num_agents, ), (- bounds[3] // 2) + 0.25)
        rad = jnp.full(self.num_agents, jnp.pi/2)
        agent_poses = jnp.stack((x_poses, y_poses, rad))

        # set poses of shelves to be centered in free space, evenly space along x axis 
        x_poses = jnp.linspace(bounds[0] + 0.5, bounds[0] + bounds[2] - self.zone_width, self.num_cells // 2)
        x_poses = jnp.concatenate((x_poses, x_poses))
        y_poses = jnp.concatenate(
            [jnp.full((self.num_cells // 2,), 0.25), jnp.full((self.num_cells // 2,), -0.25)]
        )
        rad = jnp.zeros((self.num_cells,))
        occupied = jnp.arange(self.num_cells)

        shelf_poses = jnp.stack((x_poses, y_poses, rad))
        shelf_grid = jnp.stack((x_poses, y_poses, occupied))

        poses = jnp.concatenate((agent_poses, shelf_poses), axis=-1)

        # choose unique starting requested shelves
        key, key_r = jax.random.split(key)
        request = jax.random.choice(key_r, jnp.arange(self.num_cells), (self.num_agents,), replace=False)

        state = State(
            p_pos=poses.T,
            done=jnp.full((self.num_agents), False),
            step=0,
            payload=jnp.full((self.num_agents,), -1), # track shelves carried by agents, -1 indicates no shelf carried, # use payload to track shelf weights
            grid=shelf_grid.T, # use grid to track shelf storage locations, 
            request=request,
        )

        return self.get_obs(state), state
    
    def step_env(
        self, key, state: State, actions: Dict
    ) -> Tuple[Dict, State, Dict[str, float], Dict[str, bool], Dict]:
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

        action_idx = jnp.array([actions[f'agent_{i}'] for i in range(self.num_agents)])
        actions = jnp.array([self.action_decoder(i, actions[f'agent_{i}'], state) for i in range(self.num_agents)]).reshape(
            (self.num_agents, -1)
        ) 
        poses = state.p_pos[:self.num_agents, :]

        # update pose
        updated_pose = self._robotarium_step(poses, actions)
        state = state.replace(
            p_pos=jnp.vstack([updated_pose, state.p_pos[self.num_agents:, :]]),
        )

        # check for violations
        violations = self._get_violations(state)

        # get reward
        reward = 0

        #----------------------------------------------------------
        # handle shelf pickup
        #----------------------------------------------------------
        agent_pos = state.p_pos[:self.num_agents, :2]
        shelf_pos = state.grid[:, :2]
        dists = jnp.linalg.norm(agent_pos[:, None] - shelf_pos[None, :], axis=-1)

        pickup_mask = jnp.logical_and(
            dists < self.pickup_radius, jnp.logical_and(                        # in range of shelf
                state.payload == -1, jnp.logical_and(                           # agent isn't carrying shelf
                    (action_idx == 0).flatten(),                                # agent is in pickup mode
                    state.grid[:, 2] >= 0,                                      # shelf is at pickup location
                )
            )
        ) # this is kinda nasty

        # enforce only one agent can pickup one shelf
        constraint = jnp.argmax(pickup_mask, axis=0)
        pickup_mask = jnp.array(
            [[jnp.where(i == constraint[j], pickup_mask[i][j], False) for j in range(self.num_cells)] for i in range(self.num_agents)]
        )

        # find first valid pickup per agent using argmax
        picked_shelves = jnp.argmax(pickup_mask, axis=1)
        valid_pickup = jnp.any(pickup_mask, axis=1)
        payload = jnp.where(valid_pickup, picked_shelves, state.payload)
        updated_grid = jnp.array(
            [jnp.where(jnp.isin(state.grid[i, 2], payload), -1, state.grid[i,2]) for i in range(self.num_cells)]
        )
        grid = jnp.concatenate((state.grid[:, :2], updated_grid.reshape(-1,1)), axis=-1)   # update grid to reflect shelves picked up

        #----------------------------------------------------------
        # handle shelf dropoff
        #----------------------------------------------------------
        bounds = self.robotarium.boundaries
        in_dropoff = jnp.logical_and((agent_pos[:, 0] > (bounds[0] + bounds[2] - self.zone_width)), (state.payload >= 0))
        payload_requested = jnp.isin(state.payload, state.request)

        # get indices of shelves in dropoff zone that are being dropped off
        in_dropoff = jnp.where(in_dropoff, 1, -1)
        payload_requested = jnp.where(payload_requested, 1, -1)
        shelf_indices = (state.payload+1) * in_dropoff * payload_requested # indices of shelves that are being dropped off
        shelf_indices = jnp.where(shelf_indices >=0, shelf_indices-1, -jnp.inf) # shift back

        # mask request to indicate dropped off shelves
        request = jnp.where(shelf_indices == state.request, -jnp.inf, state.request)
        num_dropped_off = jnp.sum(request != state.request)

        # update requests by randomly sampling available shelves
        p = jnp.array([jnp.where(jnp.isin(i, request), 0, 1) for i in range(self.num_cells)])
        key, key_r = jax.random.split(key)
        random_request = jax.random.choice(key_r, jnp.arange(self.num_cells,), (self.num_agents,), p=p)
        request = jnp.where(request < 0, random_request, request).astype(int)
    
        #----------------------------------------------------------
        # handle shelf return
        #----------------------------------------------------------
        return_mask = jnp.logical_and(
            dists < self.pickup_radius, jnp.logical_and(                        # in range of shelf
                state.payload >= 0, jnp.logical_and(                            # agent isn't carrying shelf
                    (action_idx == 0).flatten(), 
                    state.grid[:, 2] < 0,                                       # no shelf at pickup location                                                                                                
                )
            )
        ) # this is kinda nasty

        # enforce only one cell can receive a single shelf
        constraint = jnp.argmax(return_mask, axis=0)
        return_mask = jnp.array(
            [[jnp.where(i == constraint[j], return_mask[i][j], False) for j in range(self.num_cells)] for i in range(self.num_agents)]
        )

        # check shelves being returned and cells they are being returned to
        valid_return = jnp.any(return_mask, axis=1)
        return_shelves = jnp.where(valid_return, state.payload, -1)
        return_cells = jnp.where(valid_return, jnp.argmax(return_mask, axis=1), -1)
        payload = jnp.where(valid_return, -1, payload)

        # mask where each position indicates if it should be updated
        # (shape: [num_agents, num_cells])
        cell_update_mask = (jnp.arange(self.num_cells) == return_cells[:, None]) & valid_return[:, None]

        # values to scatter (shape: [num_agents, num_cells])
        values_to_scatter = return_shelves[:, None] * cell_update_mask

        # updates from all agents (using max to handle overlaps)
        new_values = jnp.max(values_to_scatter, axis=0)

        # updated grid by selecting between new and old values
        updated_grid = jnp.where(
            jnp.any(cell_update_mask, axis=0),  # check if agent wants to update this cell
            new_values,                         # yes, take the new value
            grid[:, 2]                          # no, keep original value
        )

        grid = jnp.concatenate((grid[:, :2], updated_grid.reshape(-1,1)), axis=-1)   # update grid to reflect shelves picked up

        state = state.replace(
            request=request,
            payload=payload,
            grid=grid
        )
        
        # get obs
        obs = self.get_obs(state)

        # get reward
        rew = num_dropped_off * self.dropoff_shaping
        violation_rew = self.violation_shaping * (violations['collision'] + violations['boundary'])
        reward = {agent: jnp.where(violation_rew == 0, rew, violation_rew) for _, agent in enumerate(self.agents)}

        # set dones
        done = jnp.full((self.num_agents), state.step >= self.max_steps)
        state = state.replace(
            done=done,
            step=state.step + 1,
        )

        info = {
            'collision': jnp.full((self.num_agents,), violations['collision']),
            'boundary': jnp.full((self.num_agents,), violations['boundary']),
        }

        dones = {a: done[i] for i, a in enumerate(self.agents)}
        dones.update({"__all__": jnp.all(done)})

        return obs, state, reward, dones, info
    
    def get_obs(self, state: State) -> Dict:
        """
        Get observation (ego_pos, payload, other_pos, storage info)

        Args:
            state: (State) environment state
        
        Returns:
            (Dict[str, float]) agent observations
        """

        def _obs(aidx: int):
            """Helper function to create agent observation"""
            
            def shift_array(arr, i):
                """
                Assuming arr is 2D, moves row i to the front
                """
                i = i % arr.shape[0]
                first_part = arr[i:]
                second_part = arr[:i]
                return jnp.concatenate([first_part, second_part])

            # get ego pose and other agent poses
            agent_pos = state.p_pos[:self.num_agents, :]
            other_pos = shift_array(agent_pos, aidx)
            ego_pos = other_pos[0]
            other_pos = other_pos[1:]

            obs = jnp.concatenate([
                ego_pos.flatten(),  # 3
                other_pos.flatten(),  # num_agents-1, 3
                state.payload[aidx].reshape(-1), # 1
                state.grid.flatten(), # num_cells, 3
            ])

            return obs

        return {a: _obs(i) for i, a in enumerate(self.agents)}

    #-----------------------------------------
    # Visualization Specific Functions (NOT INTENDED TO BE JITTED)
    #-----------------------------------------

    def render_frame(self, state: State):
        """
        Updates visualizer figure to include goal position markers

        Args:
            state: (State) environment state
        """
        
        # reset markers if at first step
        if state.step == 1:
            self.robot_markers = []
            self.storage_markers = []
            self.shelf_markers = []
            self.shelf_labels = []
            self.storage_labels = []
            self.request_label = None
            self.dropoff_marker = None
        
        shelves = state.grid[:, :2]
        agents = state.p_pos[:self.num_agents, :2]

        # add markers for robots
        if not self.robot_markers:
            self.robot_markers = [
                self.visualizer.axes.scatter(
                    jnp.array(agents[i, 0]),
                    jnp.array(agents[i, 1]),
                    marker='o',
                    s=self.determine_marker_size(self.pickup_radius),
                    facecolors='none',
                    edgecolors='black',
                    zorder=-2,
                    linewidth=1
                ) for i in range(self.num_agents)
            ]

        # add markers for shelves        
        if not self.shelf_markers:
            self.shelf_markers = [
                self.visualizer.axes.add_patch(
                    patches.Rectangle(
                        shelves[i]-0.125,
                        0.25,
                        0.25,
                        color='blue',
                        alpha=0.5,
                        zorder=1
                    )
                ) for i in range(self.num_cells)
            ]

            self.shelf_labels.extend(
                [
                    self.visualizer.axes.text(
                        shelves[i,0], shelves[i,1], i,
                        verticalalignment='center', horizontalalignment='center'
                    ) for i in range(self.num_cells)
                ]
            )
        
        # add markers for storage zones
        if not self.storage_markers:
            self.storage_markers = [
                self.visualizer.axes.add_patch(
                    patches.Rectangle(
                        shelves[i]-0.125,
                        0.25,
                        0.25,
                        color='grey',
                        alpha=0.5,
                        zorder=-1
                    )
                ) for i in range(self.num_cells)
            ]

            self.storage_labels.extend(
                [
                    self.visualizer.axes.text(
                        shelves[i,0], shelves[i,1], i,
                        verticalalignment='center', horizontalalignment='center'
                    ) for i in range(self.num_cells)
                ]
            )
        
        # add label for current reqests
        if not self.request_label:
            self.request_label = self.visualizer.axes.text(
                -1.5, 0.8, f"Request: {state.request}",
                verticalalignment='center', horizontalalignment='left'
            )
        
        # add marker for dropoff zone
        self.dropoff_marker = self.visualizer.axes.add_patch(
            patches.Rectangle([1.5-self.zone_width, -1], self.zone_width, 2, color='green', zorder=-2)
        )

        # update robot marker positions
        for i in range(self.num_agents):
            self.robot_markers[i].set_offsets(agents[i])
        
        # update shelf markers
        for i in range(self.num_agents):
            if state.payload[i] >= 0:
                idx = state.payload[i]
                self.shelf_markers[idx].set_facecolor("yellow")
                self.shelf_markers[idx].set_x(agents[i, 0]-0.125)
                self.shelf_markers[idx].set_y(agents[i, 1]-0.125)
                self.shelf_labels[idx].set_position(agents[i])
        for i in range(self.num_cells):
            if state.grid[i, 2] >= 0:
                idx = int(state.grid[i, 2])
                self.shelf_markers[idx].set_facecolor("blue")
                self.shelf_markers[idx].set_x(state.grid[i, 0]-0.125)
                self.shelf_markers[idx].set_y(state.grid[i, 1]-0.125)
                self.shelf_labels[idx].set_position(state.grid[i, :2])
