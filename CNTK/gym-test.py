# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 18:17:13 2017

@author: Dustin
"""

import gym
from gym import Env
from gym import spaces, utils
from gym.envs.toy_text import discrete
from gym.envs.toy_text.roulette import RouletteEnv
import numpy as np
from gym.utils import seeding


class experiment(Env):

    def __init__(self):
        x=0
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Discrete(25)
        self._seed()
           
    def _step(self, action):
        return 0, 0, True, {}
    
    def _reset(self):
        return 0
    
    def _render(self, mode='human', close=False):
        return
    
    def _close(self):
        return
    
    #def _configure(self):
    #    return
    
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        

        
        
        
if __name__ == '__main__': 

    env = experiment()