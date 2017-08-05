# vim: sta:et:sw=4:ts=4:sts=4
'''
Poloniex Exchange data source
'''
from poloniex import Poloniex
import pandas as pd
import numpy as np

class PoloniexDataSource(object):
    """
    DataSource US-based exchange Poloniex: poloniex.com
    """
    def __init__(self, asset_data, client=Poloniex, api_key="", secret_key=""):
        self.api_key = api_key
        self.period = asset_data["period"]
        self.pair = asset_data["pair"]
        self.days = asset_data["days"]
        self.start_date = asset_data["start_date"]
        self.end_date = asset_data["end_date"]
        self.secret_key = secret_key
        self.client = client(self.api_key, self.secret_key)
        self.chart_data = self._get_chart_data()
        self.prices = self._import_prices()
        self.data = self._build_data()
        self._step = 0

    def _build_data(self):
        '''
        Build data that will later be used by the agent based on the data gathered from the exchange
        '''
        data_frame = pd.DataFrame.from_dict(self.chart_data)
        data_frame["date"] = pd.to_datetime(data_frame["date"], unit="s")
        data_frame.set_index("date", inplace=True)
        data_frame = data_frame.convert_objects(convert_numeric=True)
        data_frame["daily_return"] = data_frame["close"].pct_change()
        data_frame.dropna(axis=0, inplace=True)
        rolling_mean = pd.stats.moments.rolling_mean(data_frame["close"], 7)
        data_frame["rolling_mean"] = rolling_mean
        data_frame.dropna(axis=0, inplace=True)
        data_frame["close_mean_ration"] = data_frame["close"] / rolling_mean
        return data_frame[["close", "volume", "daily_return", "rolling_mean", "close_mean_ration"]]

    def _get_chart_data(self):
        return self.client.returnChartData(self.pair, self.period, self.start_date, self.end_date)

    def _import_prices(self):
        data_frame = pd.DataFrame.from_dict(self.chart_data)
        data_frame["date"] = pd.to_datetime(data_frame["date"], unit="s")
        data_frame.set_index("date", inplace=True)
        data_frame = pd.DataFrame(data_frame["close"])
        data_frame = data_frame.convert_objects(convert_numeric=True)
        data_frame["daily_return"] = data_frame["close"].pct_change()
        data_frame[self.pair.split("_")[0]] = np.ones(data_frame.shape[0])
        data_frame.columns = [col.replace("close", self.pair) for col in data_frame.columns]
        data_frame.dropna(axis=0, inplace=True)
        return data_frame

    def get_prices(self):
        '''
        Getter methot to return the prices fetched through this class
        '''
        return self.prices

    def reset(self):
        '''
        Reset data source's step
        '''
        self._step = 0

    def step(self):
        '''
        Step into another episode (day) of trading and return the observed data for that date
        '''
        obs = self.data.iloc[self._step].as_matrix()
        self._step += 1
        done = self._step >= len(self.data.index)
        return obs, done
