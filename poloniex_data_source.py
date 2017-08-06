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
        self.state_data = self._build_state_data()
        self._step = 0

    def _build_state_data(self):
        '''
        Build data that will later be used by the agent based on the data gathered from the exchange
        '''
        data_frame = pd.DataFrame.from_dict(self.chart_data)
        data_frame["date"] = pd.to_datetime(data_frame["date"], unit="s")
        data_frame.set_index("date", inplace=True)
        data_frame = data_frame.convert_objects(convert_numeric=True)
        data_frame["daily_return"] = data_frame["close"].pct_change()
        data_frame.dropna(axis=0, inplace=True)
        rolling_mean = data_frame["close"].rolling(window=7, center=False).mean()
        data_frame["rolling_mean"] = rolling_mean
        data_frame["bbands"] = self._compute_bolinger_bands(data_frame)
        data_frame["close_mean_ratio"] = data_frame["close"] / rolling_mean
        data_frame.fillna(method='ffill', axis=0, inplace=True)
        data_frame.fillna(method='backfill', axis=0, inplace=True)
        data_frame["close_mean_ratio_disc"] = self._discretize(data_frame, "close_mean_ratio")
        data_frame["daily_return_disc"] = self._discretize(data_frame, "daily_return")
        return data_frame[["close_mean_ratio_disc", "bbands", "daily_return_disc"]]

    def _discretize(self, data_frame, col):
        return pd.cut(data_frame[col], 3, labels=["low", "medium", "high"])

    def _discretize_bolinger_bands(self, price, downband, avg, upband):
        n = len(price.index)
        bb_disc = np.zeros(n)
        for i in range(n):
            p = price[i]
            if p > upband[i]:
                bb_disc[i] = 2
                continue
            if p > avg[i]:
                bb_disc[i] = 1
                continue
            if p > downband[i]:
                bb_disc[i] = -1
            else:
                bb_disc[i] = -2
        return bb_disc


    def _compute_bolinger_bands(self, data_frame):
        price = data_frame["close"]
        std_dev = price.rolling(window=7, center=False).std()
        avg = data_frame["rolling_mean"]
        upband = avg + (2 * std_dev)
        downband = avg - (2 * std_dev)

        return self._discretize_bolinger_bands(price, downband, avg, upband)

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

    def get_state_data(self):
        return self.state_data.iloc[self._step].as_matrix()

    def step(self):
        '''
        Step into another episode (day) of trading and return the observed data for that date
        '''
        self._step += 1
        done = self._step >= len(self.state_data.index)
        return done
