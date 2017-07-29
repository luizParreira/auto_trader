# vim: sta:et:sw=4:ts=4:sts=4

import pandas as pd
import numpy as np

class MarketEnv(object):
    VALID_ACTIONS = ["SELL", "BUY"]

    def __init__(self, trial_data, simulator, src, testing = False):
        self.trial_data = trial_data
        self.src = src
        self.simulator = simulator
        self.testing = testing
        self.valid_actions = VALID_ACTIONS

    def reset(self):
        self.src.reset()
        self.simulator.reset()

    def step(self, action):
        assert(action in self.valid_actions)
        obs, done = self.src.step()
        reward, info = self.simulator.step(action)
        return reward, obs, info, done
