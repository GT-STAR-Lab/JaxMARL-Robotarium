import jax 
import jax.numpy as jnp
import chex
from typing import Tuple, Dict
from functools import partial
from smax.environments.mpe.simple import SimpleMPE, State
from gymnax.environments.spaces import Box
from smax.environments.mpe.default_params import *


class SimpleTagMPE(SimpleMPE):

    def __init__(self,
                 num_good_agents=1,
                 num_adversaries=3,
                 num_obs=2,
                 action_type=DISCRETE_ACT,):
        
        dim_c = 2 # NOTE follows code rather than docs

        num_agents = num_good_agents + num_adversaries
        num_landmarks = num_obs 

        self.num_good_agents, self.num_adversaries = num_good_agents, num_adversaries

        self.adversaries = ["adversary_{}".format(i) for i in range(num_adversaries)]
        self.good_agents = ["agent_{}".format(i) for i in range(num_good_agents)]
        agents = self.adversaries + self.good_agents

        landmarks = ["landmark {}".format(i) for i in range(num_obs)]

        # Action and observation spaces
        #action_spaces = {i: Box(0.0, 1.0, (5,)) for i in agents}

        observation_spaces = {i: Box(-jnp.inf, jnp.inf, (16,)) for i in self.adversaries }
        observation_spaces.update({i: Box(-jnp.inf, jnp.inf, (14,)) for i in self.good_agents})


        colour = [ADVERSARY_COLOUR] * num_adversaries + [AGENT_COLOUR] * num_good_agents + [OBS_COLOUR] * num_obs 
            
        # Parameters
        rad = jnp.concatenate([jnp.full((self.num_adversaries), ADVERSARY_RADIUS),
                            jnp.full((self.num_good_agents), AGENT_RADIUS),
                            jnp.full((num_landmarks), LANDMARK_RADIUS)])
        accel = jnp.concatenate([jnp.full((self.num_adversaries), ADVERSARY_ACCEL),
                                jnp.full((self.num_good_agents), 4.0)])
        max_speed = jnp.concatenate([jnp.full((self.num_adversaries), 1.0),
                                jnp.full((self.num_good_agents), 1.3),
                                jnp.full((num_landmarks), 0.0)])
        
        super().__init__(num_agents=num_agents, 
                         agents=agents,
                         num_landmarks=num_landmarks,
                         landmarks=landmarks,
                         action_type=action_type,
                         observation_spaces=observation_spaces,
                         dim_c=dim_c,
                         colour=colour,
                         rad=rad,
                         accel=accel,
                         max_speed=max_speed,
                         )
    

    def get_obs(self, state: State):

        @partial(jax.vmap, in_axes=(0))
        def _common_stats(aidx):
            """ Values needed in all observations """
            
            landmark_pos = state.p_pos[self.num_agents:] - state.p_pos[aidx]  # Landmark positions in agent reference frame

            # Zero out unseen agents with other_mask
            other_pos = (state.p_pos[:self.num_agents] - state.p_pos[aidx]) 
            other_vel = state.p_vel[:self.num_agents] 
            
            # use jnp.roll to remove ego agent from other_pos and other_vel arrays
            other_pos = jnp.roll(other_pos, shift=self.num_agents-aidx-1, axis=0)[:self.num_agents-1]
            other_vel = jnp.roll(other_vel, shift=self.num_agents-aidx-1, axis=0)[:self.num_agents-1]
            
            other_pos = jnp.roll(other_pos, shift=aidx, axis=0)
            other_vel = jnp.roll(other_vel, shift=aidx, axis=0)
            
            return landmark_pos, other_pos, other_vel

        landmark_pos, other_pos, other_vel = _common_stats(self.agent_range)

        def _good(aidx):
            return jnp.concatenate([
                state.p_vel[aidx].flatten(), # 2
                state.p_pos[aidx].flatten(), # 2
                landmark_pos[aidx].flatten(), # 5, 2
                other_pos[aidx].flatten(), # 5, 2
            ])


        def _adversary(aidx):
            return jnp.concatenate([
                state.p_vel[aidx].flatten(), # 2
                state.p_pos[aidx].flatten(), # 2
                landmark_pos[aidx].flatten(), # 5, 2
                other_pos[aidx].flatten(), # 5, 2
                other_vel[aidx,-1:].flatten(), # 2
            ])
        
        obs = {a: _adversary(i) for i, a in enumerate(self.adversaries)}
        obs.update({a: _good(i+self.num_adversaries) for i, a in enumerate(self.good_agents)})
        return obs
    
    def rewards(self, state: State) -> Dict[str, float]:

        @partial(jax.vmap, in_axes=(0, None))
        def _collisions(agent_idx: int, other_idx: int):
            return jax.vmap(self.is_collision, in_axes=(None, 0, None))(
                agent_idx,
                other_idx,
                state,
            )

        c = _collisions(jnp.arange(self.num_good_agents)+self.num_adversaries, 
                        jnp.arange(self.num_adversaries), 
                        )  # [agent, adversary, collison]

        def _good(aidx: int, collisions: chex.Array):

            rew = -10 * jnp.sum(collisions[aidx])

            mr = jnp.sum(self.map_bounds_reward(jnp.abs(state.p_pos[aidx])))
            rew -= mr
            return rew 
        jax.debug.print('collisions {c}', c=c)
        ad_rew = 10 * jnp.sum(c)

        rew = {a: ad_rew for a in self.adversaries}
        rew.update({a: _good(i+self.num_adversaries, c) for i, a in enumerate(self.good_agents)})
        return rew

