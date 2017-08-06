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

    def get_current_state(self):
        '''
        Returns a tuple with the state
        (holding, close_mean_ratio, bbands, daily_return) [discretized]
        '''
        data = self.src.get_state_data()
        # close_mean_ratio, bbands, daily_return
        return (False, data[0], data[1], data[2])

    def step(self, action):
        '''
        Step into another time step with a chosen action by the agent
        '''
        assert action in self.valid_actions

        done = self.src.step()
        reward, info = self.simulator.step(action)
        return reward, info, done
