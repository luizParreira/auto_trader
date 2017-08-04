# vim: sta:et:sw=4:ts=4:sts=4

import pandas as pd
import numpy as np

class MarketEnv(object):
    def __init__(self, simulator, src, testing = False):
        self.simulator = simulator
        self.src = src
        self.testing = testing
        self.valid_actions = ["SELL", "BUY"]

    def reset(self):
        self.src.reset()
        self.simulator.reset()

    def step(self, action):
        assert(action in self.valid_actions)
        obs, done = self.src.step()
        reward, info = self.simulator.step(action)
        return reward, obs, info, done
