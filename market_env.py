# vim: sta:et:sw=4:ts=4:sts=4

import gym
import pandas as pd
import numpy as np
import math
from poloniex import Poloniex

def sharpe_ratio(returns, freq=365) :
  """Given a set of returns, calculates naive (rfr=0) sharpe """
  return (np.sqrt(freq) * np.mean(returns))/np.std(returns)

def prices_to_returns(prices):
  px = pd.DataFrame(prices)
  nl = px.shift().fillna(0)
  R = ((px - nl)/nl).fillna(0).replace([np.inf, -np.inf], np.nan).dropna()
  R = np.append( R[0].values, 0)
  return R


class DataSource(object):
    '''
    Class responsible for querying data from Poloniex API, prepares it for the trading environment
    and then also acts as as data source for each step on the environment
    '''
    def __init__(self, pair, period, start_date, end_date, days = 365, client = Poloniex()):
        self.pair = pair
        self.days = days
        data = client.returnChartData(pair, period, start_date, end_date)
        df = pd.DataFrame.from_dict(data)
        df['date'] = pd.to_datetime(df['date'], unit='s')
        df.set_index('date', inplace=True)
        df = df.convert_objects(convert_numeric=True)
        df['close_percent_change'] = df['close'].pct_change()
        df['volume_percent_change'] = df['volume'].pct_change()
        self.data = df[['close', 'volume', 'close_percent_change', 'volume_percent_change']]

    def reset(self):
        self.idx = np.random.randint( low = 0, high=len(self.data.index)-self.days )
        self.step = 0

    def _step(self):
        '''
        Step function is responsible for geting the data for the current step and return it,
        it also returns whether we are done with the data as well
        '''
        obs = self.data.iloc[self.idx].as_matrix()
        self.idx += 1
        self.step += 1
        done = self.step >= self.days
        return obs, done

    def data(self):
        return self.data





class MarketEnv(gym.Env):
    '''
    This GYM implements an environment in which an agent can simulate cryptocurrency trading
    '''
    def __init__(self):
        pass

    def reset(self):
        pass

    def _step(self):
        pass
