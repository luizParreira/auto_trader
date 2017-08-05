# vim: sta:et:sw=4:ts=4:sts=4

'''
Market Environment
'''

class MarketEnv(object):
    '''
    Market Environment class, it dealls with the simulator and data source
    '''
    def __init__(self, simulator, src, testing=False):
        self.simulator = simulator
        self.src = src
        self.testing = testing
        self.valid_actions = ["SELL", "BUY"]

    def reset(self):
        '''
        Reset source and simulator.
        '''
        self.src.reset()
        self.simulator.reset()

    def step(self, action):
        '''
        Step into another time step with a chosen action by the agent
        '''
        assert action in self.valid_actions

        obs, done = self.src.step()
        reward, info = self.simulator.step(action)
        return reward, obs, info, done
