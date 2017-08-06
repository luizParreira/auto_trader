# vim: sta:et:sw=4:ts=4:sts=4
'''
Trading Simulator
'''
import pandas as pd
import numpy as np

class TradingSimulator(object):
    """
    Trading simulator, responsible for handling the data transitions involved on each step
    of trading. It also keeps track of the data associated with the simulation.
    """
    def __init__(self, value, prices, simulation_data, trading_cost=.0025):
        self.value = value
        self.trading_cost = trading_cost
        self.simulation_data = simulation_data
        self.prices = prices
        self.trades = self._build_zero_filled_df(self.prices)
        self.holdings = self._build_zero_filled_df(self.prices)
        self.values = self._build_zero_filled_df(self.prices)
        self.portfolio_value = self._build_portfolio_value_df(self.prices)
        self.actions = np.zeros(len(self.prices.index))
        self.cumulative_return = np.zeros(len(self.prices.index))
        self._step = 0
        self.dates = self.prices.index

    def reset(self):
        '''
        Reset simulator data
        '''
        self._step = 0
        self.trades = self._build_zero_filled_df(self.prices)
        self.holdings = self._build_zero_filled_df(self.prices)
        self.values = self._build_zero_filled_df(self.prices)
        self.portfolio_value = self._build_portfolio_value_df(self.prices)

    def step(self, action):
        """
        Step another day into the simulation.
        Actions:
        - BUY(1)
        - SELL(-1)
        """
        assert action in ["BUY", "SELL"]

        pair = self.simulation_data["pair"]
        date = self.dates[self._step]
        previous_date = self.dates[0] if self._step == 0 else self.dates[self._step - 1]
        base_symbol, trading_symbol = pair.split("_")
        price = self.prices[pair][date]

        self.actions[self._step] = 1 if action == "BUY" else -1

        if self._step == 0:
            self.trades[base_symbol][date] = self.value

        if action == "BUY":
            if self._step == 0 or self.holdings[trading_symbol][previous_date] == 0:
                self.trades[trading_symbol][date] = self.value / price
                self.trades[base_symbol][date] = -self.value
                self.trading_cost = 0.15 / 100.0
            else:
                self.trades[trading_symbol][date] = self.trades[trading_symbol][previous_date]
                self.trades[base_symbol][date] = self.trades[base_symbol][previous_date]
                self.trading_cost = 0

        if action == "SELL":
            previous_holding = self.holdings[trading_symbol][previous_date]
            if previous_holding > 0:
                self.trades[trading_symbol][date] = -previous_holding
                self.trades[base_symbol][date] = previous_holding * price
                self.trading_cost = 0.25 / 100.0
            else:
                self.trades[trading_symbol][date] = self.trades[trading_symbol][previous_date]
                self.trades[base_symbol][date] = self.trades[base_symbol][previous_date]
                self.trading_cost = 0

        # Update holdings data frame
        if self.trades[trading_symbol][date] <= 0:
            self.holdings[trading_symbol][date] = 0
            self.values[trading_symbol][date] = 0
        else:
            self.holdings[trading_symbol][date] = self.trades[trading_symbol][date]
            self.values[trading_symbol][date] = price * self.trades[trading_symbol][date]

        if self.trades[base_symbol][date] <= 0:
            self.holdings[base_symbol][date] = 0
            self.values[base_symbol][date] = 0
        else:
            self.holdings[base_symbol][date] = self.trades[base_symbol][date]
            self.values[base_symbol][date] = self.holdings[base_symbol][date]
        self.value = self.holdings[base_symbol][date]

        # Update portfolio value
        port_value = self.values[self.values.columns].sum(axis=1)[date]
        self.portfolio_value["portfolio_value"][date] = port_value

        # Compute the _step reward
        currently_holing = self.holdings[trading_symbol][date] > 0
        if currently_holing:
            reward = (1 - self.trading_cost) * self.prices["daily_return"][date]
            if self._step == 0:
                reward = -self.trading_cost
        else:
            reward = (1 - self.trading_cost) * self.prices["daily_return"][date] * -1.0
        info = {
            "reward": reward,
            "port_value": self.portfolio_value,
            "holdings": self.holdings,
            "currently_holding": currently_holing,
            "values": self.values,
            "actions": self.actions
        }
        self._step += 1
        return reward, info

    def _build_zero_filled_df(self, df_base):
        '''
        Utils method used to build a zero-filled df.
        '''
        columns = [col for col in df_base.columns if col != "daily_return"]
        pair = self.simulation_data["pair"]
        pairs = pair.split("_")
        second_pair = pairs[1]
        columns = [second_pair if col == pair else col for col in columns]
        shape = np.zeros((df_base.shape[0], len(columns)))
        return pd.DataFrame(shape, columns=columns, index=df_base.index)

    def _build_portfolio_value_df(self, df_base):
        '''
        Util method used to create portfolio value data frame
        '''
        return pd.DataFrame(
            np.zeros((df_base.shape[0], 1)),
            columns=["portfolio_value"],
            index=df_base.index
        )
