# vim: sta:et:sw=4:ts=4:sts=4
import pandas as pd
import numpy as np

class TradingSimulator(object):
    """
    Trading simulator, responsible for handling the data transitions involved on each step
    of trading. It also keeps track of the data associated with the simulation.
    """
    def __init__(self, value, prices, simulation_data, trading_cost = .0025):
        self.value   = value
        self.trading_cost    = trading_cost
        self.simulation_data = simulation_data
        self.prices          = prices
        self.trades          = self._build_zero_filled_df(self.prices)
        self.holdings        = self._build_zero_filled_df(self.prices)
        self.values          = self._build_zero_filled_df(self.prices)
        self.portfolio_value = self._build_portfolio_value_df(self.prices)
        self.step            = 0
        self.dates           = self.prices.index

    def reset(self):
        self.step = 0
        self.trades = self._build_zero_filled_df(self.prices)
        self.holdings = self._build_zero_filled_df(self.prices)
        self.values = self._build_zero_filled_df(self.prices)
        self.portfolio_value = self._build_portfolio_value_df(self.prices)

    def step(self, action):
        """
        Step another day into the simulation.
        Actions:
        - BUY
        - SELL
        """
        pair = self.simulation_data["pair"]
        date = self.dates[self.step]
        previous_date = self.dates[0] if self.step == 0 else self.dates[self.step - 1]
        pairs = pair.split("_")
        base_symbol, trading_symbol = pairs[0], pairs[1]
        price = self.prices[pair][date]

        if action == "BUY":
            if self.step == 0 or self.holdings[trading_symbol][previous_date] == 0:
                self.trades[trading_symbol][date] = self.value / price
                self.trades[base_symbol][date] = -self.value
            else:
                self.trades[trading_symbol][date] = self.trades[trading_symbol][previous_date]
        if action == "SELL":
            if self.step == 0 or self.holdings[trading_symbol][previous_date] == 0:
                self.trades[trading_symbol][date] = -(self.value / price)
                self.trades[base_symbol][date] = self.value
            else:
                self.trades[trading_symbol][date] = self.trades[trading_symbol][previous_date]

        # Update holdings data frame
        self.holdings[trading_symbol][date] = self.trades[trading_symbol][date]
        self.holdings[base_symbol][date] = self.trades[base_symbol][date] + self.value
        self.value = self.holdings[base_symbol][date]

        # Update the value dataframe
        self.values[trading_symbol][date] = price * self.trades[trading_symbol][date]
        self.values[base_symbol][date] = self.holdings[base_symbol][date]

        # Update portfolio value
        port_value = self.values[self.values.columns].sum(axis=1)[date]
        self.portfolio_value["portfolio_value"][date] = port_value

        # Compute the step reward
        curr_trade =  self.trades[trading_symbol][date]
        prev_trade = self.trades[trading_symbol][previous_date]
        should_charge = self.step != 0 and curr_trade == prev_trade
        trading_cost = 0.0 if should_charge else self.trading_cost
        if self.holdings[trading_symbol][date] > 0:
            reward = (1 - trading_cost) * self.prices["daily_return"][date]
        else:
            reward = (1 - trading_cost) * self.prices["daily_return"][date] * -1.0
        info = {
            "reward": reward,
            "port_value": self.portfolio_value,
            "holdings": self.holdings,
            "values": self.values
        }
        self.step += 1
        return reward, info

    def _build_zero_filled_df(self, df_base):
        columns = filter(lambda col: col not in ["daily_return"], df_base.columns)
        pair = self.simulation_data["pair"]
        pairs = pair.split("_")
        second_pair = pairs[1]
        columns = [second_pair if col == pair else col for col in columns]
        shape = np.zeros((df_base.shape[0], len(columns)))
        return pd.DataFrame(shape, columns=columns, index=df_base.index)

    def _build_portfolio_value_df(self, df_base):
        return pd.DataFrame(
            np.zeros((df_base.shape[0], 1)),
            columns=["portfolio_value"],
            index=df_base.index
        )
