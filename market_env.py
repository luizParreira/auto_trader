# vim: sta:et:sw=4:ts=4:sts=4

import gym
import pandas as pd
import numpy as np
import math
from poloniex import Poloniex

class PoloniexDataSource(object):
  '''
  DataSource US-based exchange Poloniex: poloniex.com
  '''
  def __init__(self, client = Poloniex, api_key = '', secret_key = ''):
    self.api_key = api_key
    self.secret_key = secret_key
    self.client = client(self.api_key, self.secret_key)

  def _featurize(self, df):
    df['daily_return'] = df['close'].pct_change()
    df.dropna(axis=0, inplace=True)
    rolling_mean = pd.stats.moments.rolling_mean(df['close'], 7)
    df.dropna(axis=0, inplace=True)
    df['rolling_mean'] = rolling_mean
    df['close_mean_ration'] = df['close'] / rolling_mean
    return df[
      [
        'close',
        'volume',
        'daily_return',
        'rolling_mean',
        'close_mean_ration'
      ]
    ]

  def get_historical_data(self, pair, period, start_date, end_date):
    df = pd.DataFrame.from_dict(
      self.client.returnChartData(pair, period, start_date, end_date)
    )
    df['date'] = pd.to_datetime(df['date'], unit='s')
    df.set_index('date', inplace=True)
    df = df.convert_objects(convert_numeric=True)
    return self._featurize(df)


class DataSource(object):
    '''
    Class responsible for querying data from Poloniex API, prepares it for the trading environment
    and then also acts as as data source for each step on the environment
    '''

    def __init__(self, pair, period, start_date, end_date, days = 365, src = PoloniexDataSource()):
        self.pair = pair
        self.days = days
        self.data = src.get_historical_data(pair, period, start_date, end_date)

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
