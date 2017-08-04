# vim: sta:et:sw=4:ts=4:sts=4
from poloniex import Poloniex
import pandas as pd
import numpy as np

class PoloniexDataSource(object):
    """
    DataSource US-based exchange Poloniex: poloniex.com
    """
    def __init__(self, asset_data, client = Poloniex, api_key = "", secret_key = ""):
        self.api_key = api_key
        self.period = asset_data["period"]
        self.pair = asset_data["pair"]
        self.days = asset_data["days"]
        self.sd = asset_data["start_date"]
        self.ed = asset_data["end_date"]
        self.secret_key = secret_key
        self.client = client(self.api_key, self.secret_key)
        self.chart_data = self._chart_data()
        self.prices = self._import_prices()
        self.data = self._build_data()
        self.step = 0

    def _build_data(self):
        df = pd.DataFrame.from_dict(self.chart_data)
        df["date"] = pd.to_datetime(df["date"], unit="s")
        df.set_index("date", inplace=True)
        df =  df.convert_objects(convert_numeric=True)
        df["daily_return"]= df["close"].pct_change()
        df.dropna(axis=0, inplace=True)
        rolling_mean = pd.stats.moments.rolling_mean(df["close"], 7)
        df["rolling_mean"] = rolling_mean
        df.dropna(axis=0, inplace=True)
        df["close_mean_ration"] = df["close"] / rolling_mean
        return df[["close","volume","daily_return","rolling_mean","close_mean_ration"]]

    def _chart_data(self):
        return self.client.returnChartData(self.pair, self.period, self.sd, self.ed)

    def _import_prices(self):
        df = pd.DataFrame.from_dict(self.chart_data)
        df["date"] = pd.to_datetime(df["date"], unit="s")
        df.set_index("date", inplace=True)
        df = pd.DataFrame(df["close"])
        df =  df.convert_objects(convert_numeric=True)
        df["daily_return"] = df["close"].pct_change()
        df[self.pair.split("_")[0]] = np.ones(df.shape[0])
        df.columns = [col.replace("close", self.pair) for col in df.columns]
        df.dropna(axis=0, inplace=True)
        return df

    def get_prices(self):
        return self.prices

    def reset(self):
        self.step = 0

    def step(self):
        obs = self.data.iloc[self.step].as_matrix()
        self.step += 1
        done = self.step >= len(self.data.index)
        return obs, done
